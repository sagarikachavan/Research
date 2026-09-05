#!/usr/bin/env python3
"""
eval/baseline_paper_cnn.py
===========================
Reference baseline: the dual-head "frozen GPT-2 + TextCNN" step/MCP
classifier described in the Pen-Strategist paper (arXiv:2605.04499, Section
4.2.2 / Algorithm 1) and published at
https://github.com/YasodGinige/Pentest-Strategist/blob/main/Model_Training_and_Testing/{train_step_CNN.py,test_step_CNN.py}

This is a *separate, self-contained* baseline model — it is not part of the
GNN → Qwen-SFT → Qwen-GRPO curriculum (stage1/2/3). It exists purely so
stage1/2/3 can be compared against the paper's own reported step-model
architecture, trained on the exact same input the paper uses:

    input text = "New strategy" + "\n" + "Strategy explanation"

Both STEP_LABELS and MCP_LABELS are imported from core.config so the label
space is guaranteed identical to the rest of this pipeline (they already
match the paper's Table 1 / Section 4.1.3 label sets verbatim, so no
relabeling step is needed).

Usage:
    python eval/baseline_paper_cnn.py                 # train (if no cached
                                                        # checkpoint) + evaluate
    python eval/baseline_paper_cnn.py --force-retrain  # ignore cached ckpt
    python eval/baseline_paper_cnn.py --eval-only      # skip training, just
                                                        # score data/test_data.csv
                                                        # with the cached ckpt

Output:
    checkpoints/paper_stepcnn_gpt2/best.pt   (+ tokenizer files)
    output/baseline_paper_cnn.csv            (schema matches stage1/2/3 CSVs
                                               so eval/comparison_report.py
                                               picks it up automatically)
"""
import os
import sys
import csv
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2TokenizerFast, GPT2Model, get_linear_schedule_with_warmup

# ── Path bootstrap (same convention as the other eval/ scripts) ─────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # noqa: E402
    STEP_LABELS, MCP_LABELS, TRAIN_CSV, TEST_CSV, ROOT, CKPT_DIR,
)

BASELINE_CKPT_DIR = os.path.join(CKPT_DIR, "paper_stepcnn_gpt2")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BASELINE_CKPT_DIR, exist_ok=True)


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


STEP_LABEL2ID_NORM = {_norm(s): i for i, s in enumerate(STEP_LABELS)}
MCP_LABEL2ID = {_norm(s): i for i, s in enumerate(MCP_LABELS)}


def step_label_to_id(raw_label: str):
    x = _norm(raw_label)
    return STEP_LABEL2ID_NORM.get(x, None)


def parse_mcp_tasks(mcp_str: str) -> Set[str]:
    """MCP_tasks cells look like "{'Dirbuster': '...', 'Google search': '...'}"."""
    import ast
    if not mcp_str or not isinstance(mcp_str, str):
        return set()
    mcp_str = mcp_str.strip()
    if not mcp_str:
        return set()
    try:
        obj = ast.literal_eval(mcp_str)
        if isinstance(obj, dict):
            servers = set()
            for k in obj.keys():
                k_norm = _norm(k)
                for mcp_label in MCP_LABELS:
                    if k_norm == _norm(mcp_label):
                        servers.add(mcp_label)
                        break
            return servers
    except Exception:
        pass
    return set()


def mcp_to_multihot(servers: Set[str]) -> np.ndarray:
    vec = np.zeros(len(MCP_LABELS), dtype=np.float32)
    for s in servers:
        idx = MCP_LABEL2ID.get(_norm(s))
        if idx is not None:
            vec[idx] = 1.0
    return vec


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
class StepMcpDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts, self.step_labels, self.mcp_vecs = [], [], []
        self.skipped_unknown_step = 0
        self.skipped_empty = 0

        df = df.copy()
        for col in ["New strategy", "Strategy explanation", "New step", "MCP_tasks"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            df[col] = df[col].fillna("").astype(str)

        for _, row in df.iterrows():
            step_id = step_label_to_id(row["New step"])
            if step_id is None:
                self.skipped_unknown_step += 1
                continue
            text = (row["New strategy"].strip() + "\n" + row["Strategy explanation"].strip()).strip()
            if not text:
                self.skipped_empty += 1
                continue
            self.texts.append(text)
            self.step_labels.append(step_id)
            self.mcp_vecs.append(mcp_to_multihot(parse_mcp_tasks(row["MCP_tasks"])))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "step_label": self.step_labels[idx],
            "mcp_multihot": self.mcp_vecs[idx],
        }


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    step_labels: torch.Tensor
    mcp_labels: torch.Tensor


def make_collate(tokenizer: GPT2TokenizerFast, max_len: int):
    def collate(examples: List[Dict[str, Any]]) -> Batch:
        texts = [ex["text"] for ex in examples]
        step_labels = torch.tensor([ex["step_label"] for ex in examples], dtype=torch.long)
        mcp_labels = torch.tensor(np.stack([ex["mcp_multihot"] for ex in examples]), dtype=torch.float32)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return Batch(enc["input_ids"], enc["attention_mask"], step_labels, mcp_labels)
    return collate


# -----------------------------------------------------------------------
# Model — identical architecture to the paper / reference scripts
# -----------------------------------------------------------------------
class TextCNN(nn.Module):
    def __init__(self, hidden_size, num_classes, kernel_sizes=(2, 3, 4, 5), num_filters=128, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))
            p = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            feats.append(p)
        z = self.dropout(torch.cat(feats, dim=1))
        return self.fc(z)


class DualHeadGPT2TextCNN(nn.Module):
    def __init__(self, gpt2, num_step, num_mcp, kernel_sizes=(2, 3, 4, 5), num_filters=128, dropout=0.3):
        super().__init__()
        self.gpt2 = gpt2
        for p in self.gpt2.parameters():
            p.requires_grad = False
        hidden = gpt2.config.hidden_size
        self.step_cnn = TextCNN(hidden, num_step, kernel_sizes, num_filters, dropout)
        self.mcp_cnn = TextCNN(hidden, num_mcp, kernel_sizes, num_filters, dropout)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            token_emb = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.step_cnn(token_emb), self.mcp_cnn(token_emb)


def step_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def mcp_f1(logits, labels, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * labels).sum(dim=1)
    fp = (preds * (1 - labels)).sum(dim=1)
    fn = ((1 - preds) * labels).sum(dim=1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


# -----------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------
def train(
    device, model_name="gpt2", max_len=512, batch_size=16, epochs=30, lr=2e-4,
    weight_decay=0.01, warmup_ratio=0.1, step_loss_weight=1.0, mcp_loss_weight=1.5,
    seed=42, val_ratio=0.2,
):
    set_seed(seed)
    df = pd.read_csv(TRAIN_CSV)

    # Machine-level split, consistent with how Stage 1/2/3 avoid leakage
    # (see CHANGES_AND_FINDINGS.md §3) even though the reference scripts
    # split at the row level.
    if "Machine" in df.columns:
        machines = df["Machine"].dropna().unique().tolist()
        rng = random.Random(seed)
        rng.shuffle(machines)
        n_val_m = max(1, int(val_ratio * len(machines)))
        val_machines = set(machines[:n_val_m])
        val_df = df[df["Machine"].isin(val_machines)].reset_index(drop=True)
        train_df = df[~df["Machine"].isin(val_machines)].reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(val_ratio * len(df))
        val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]

    train_ds = StepMcpDataset(train_df)
    val_ds = StepMcpDataset(val_df)
    print(f"[paper_stepcnn_gpt2] train={len(train_ds)} val={len(val_ds)} "
          f"(skipped unknown-step={train_ds.skipped_unknown_step + val_ds.skipped_unknown_step}, "
          f"empty={train_ds.skipped_empty + val_ds.skipped_empty})")
    if len(train_ds) < 20:
        raise ValueError(f"Training set too small after filtering: {len(train_ds)} rows")

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    collate = make_collate(tok, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    gpt2 = GPT2Model.from_pretrained(model_name).to(device).eval()
    model = DualHeadGPT2TextCNN(gpt2, len(STEP_LABELS), len(MCP_LABELS)).to(device)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps
    )

    best_score = -1.0
    ckpt_path = os.path.join(BASELINE_CKPT_DIR, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            step_labels = batch.step_labels.to(device)
            mcp_labels = batch.mcp_labels.to(device)

            step_logits, mcp_logits = model(input_ids, attn)
            loss = (
                step_loss_weight * F.cross_entropy(step_logits, step_labels)
                + mcp_loss_weight * F.binary_cross_entropy_with_logits(mcp_logits, mcp_labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # ── validation ──
        model.eval()
        va_step_acc, va_mcp_f1, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch.input_ids.to(device)
                attn = batch.attention_mask.to(device)
                step_labels = batch.step_labels.to(device)
                mcp_labels = batch.mcp_labels.to(device)
                step_logits, mcp_logits = model(input_ids, attn)
                va_step_acc += step_accuracy(step_logits, step_labels)
                va_mcp_f1 += mcp_f1(mcp_logits, mcp_labels)
                n += 1
        va_step_acc /= max(n, 1)
        va_mcp_f1 /= max(n, 1)
        combined = va_step_acc + va_mcp_f1
        print(f"[paper_stepcnn_gpt2] epoch {epoch:02d}/{epochs}  "
              f"val_step_acc={va_step_acc:.4f}  val_mcp_f1={va_mcp_f1:.4f}  combined={combined:.4f}")

        if combined > best_score:
            best_score = combined
            torch.save({
                "step_cnn_state_dict": model.step_cnn.state_dict(),
                "mcp_cnn_state_dict": model.mcp_cnn.state_dict(),
                "model_name": model_name,
                "max_len": max_len,
                "val_step_acc": va_step_acc,
                "val_mcp_f1": va_mcp_f1,
            }, ckpt_path)
            tok.save_pretrained(BASELINE_CKPT_DIR)

    print(f"[paper_stepcnn_gpt2] best combined val score: {best_score:.4f}  →  {ckpt_path}")
    return ckpt_path


# -----------------------------------------------------------------------
# Evaluate on data/test_data.csv and write a CSV in the pipeline's schema
# -----------------------------------------------------------------------
@torch.no_grad()
def evaluate_and_write_csv(device, ckpt_path, mcp_threshold=0.5, batch_size=32):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt["model_name"]
    max_len = ckpt["max_len"]

    tok = GPT2TokenizerFast.from_pretrained(BASELINE_CKPT_DIR)
    gpt2 = GPT2Model.from_pretrained(model_name).to(device).eval()
    model = DualHeadGPT2TextCNN(gpt2, len(STEP_LABELS), len(MCP_LABELS)).to(device)
    model.step_cnn.load_state_dict(ckpt["step_cnn_state_dict"])
    model.mcp_cnn.load_state_dict(ckpt["mcp_cnn_state_dict"])
    model.eval()

    test_df = pd.read_csv(TEST_CSV)
    ds = StepMcpDataset(test_df)
    print(f"[paper_stepcnn_gpt2] test set: {len(ds)} usable rows "
          f"(skipped unknown-step={ds.skipped_unknown_step}, empty={ds.skipped_empty})")
    if len(ds) == 0:
        raise ValueError("No usable rows in data/test_data.csv for this baseline.")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate(tok, max_len))

    rows = []
    idx = 0
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attn = batch.attention_mask.to(device)
        step_logits, mcp_logits = model(input_ids, attn)
        step_preds = step_logits.argmax(dim=1).cpu().numpy()
        mcp_preds = (torch.sigmoid(mcp_logits) > mcp_threshold).cpu().numpy()

        for i in range(len(step_preds)):
            gold_step = STEP_LABELS[ds.step_labels[idx]]
            pred_step = STEP_LABELS[int(step_preds[i])]
            gold_mcp = [MCP_LABELS[j] for j in range(len(MCP_LABELS)) if ds.mcp_vecs[idx][j] == 1]
            pred_mcp = [MCP_LABELS[j] for j in range(len(MCP_LABELS)) if mcp_preds[i][j] == 1]
            rows.append({
                "gold_new_step": gold_step,
                "predicted_new_step": pred_step,
                "gold_mcp_tasks": "|".join(gold_mcp),
                "predicted_mcp_tasks": "|".join(pred_mcp),
            })
            idx += 1

    out_path = os.path.join(OUTPUT_DIR, "baseline_paper_cnn.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[paper_stepcnn_gpt2] predictions written to: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mcp_threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                     help="Skip training; requires an existing checkpoint at "
                          "checkpoints/paper_stepcnn_gpt2/best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paper_stepcnn_gpt2] device: {device}")

    ckpt_path = os.path.join(BASELINE_CKPT_DIR, "best.pt")

    if args.eval_only:
        if not os.path.exists(ckpt_path):
            print(f"[paper_stepcnn_gpt2] ERROR: --eval-only given but no checkpoint at {ckpt_path}", file=sys.stderr)
            sys.exit(1)
    elif args.force_retrain or not os.path.exists(ckpt_path):
        ckpt_path = train(
            device, model_name=args.model, max_len=args.max_len,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
    else:
        print(f"[paper_stepcnn_gpt2] found cached checkpoint at {ckpt_path}, skipping training "
              f"(use --force-retrain to retrain).")

    evaluate_and_write_csv(device, ckpt_path, mcp_threshold=args.mcp_threshold)


if __name__ == "__main__":
    main()