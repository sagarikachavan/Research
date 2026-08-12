"""
Stage 1: Supervised training of the GNN + context-fusion classifier for
  - Next Step Type (single-label, 10-way)   -> Accuracy / Macro-F1
  - MCP tool type   (multi-label, 11-way)   -> subset accuracy, micro/macro-F1

Training input: stepmodelv2/input/train.json
Evaluation input: stepmodelv2/input/test.json

Run:
    python stage1_gnn_train.py
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score, f1_score,
)

from config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, STAGE1_CKPT, STAGE1_LR, STAGE1_EPOCHS,
    STAGE1_BATCH_SIZE, STEP_LOSS_WEIGHT, MCP_LOSS_WEIGHT, RANDOM_SEED, MCP_LABELS,
)
from data_utils import load_from_input_json, _embed_texts, CONTEXT_COLUMNS
from graph_encoder import Stage1Classifier
from mcp_threshold_search import search_per_class_thresholds

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class Stage1Dataset(Dataset):
    def __init__(self, json_path, split="train"):
        self.examples = load_from_input_json(json_path, split)
        self.split = split
        # pre-embed all context text once (frozen encoder, no grad needed)
        self._embed_cache()

    def _embed_cache(self):
        for ex in self.examples:
            texts = [ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]
            ex["field_embs"] = _embed_texts(texts)  # (5, TEXT_EMB_DIM)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Graph is already a torch_geometric Data object built at load time
        return {
            "graph": ex["graph"],
            "field_embs": torch.tensor(ex["field_embs"], dtype=torch.float32),
            "step_idx": torch.tensor(ex["step_idx"], dtype=torch.long),
            "mcp_vec": torch.tensor(ex["mcp_vec"], dtype=torch.float32),
        }


def collate(batch):
    graphs = Batch.from_data_list([b["graph"] for b in batch])
    field_embs = torch.stack([b["field_embs"] for b in batch])
    step_idx = torch.stack([b["step_idx"] for b in batch])
    mcp_vec = torch.stack([b["mcp_vec"] for b in batch])
    return graphs, field_embs, step_idx, mcp_vec


def evaluate(model, loader, device, threshold=0.5, return_probs=False):
    model.eval()
    step_preds, step_gold = [], []
    mcp_preds, mcp_gold = [], []
    mcp_probs = []  # Store raw probabilities for threshold optimization
    with torch.no_grad():
        for graphs, field_embs, step_idx, mcp_vec in loader:
            graphs = graphs.to(device)
            field_embs, step_idx, mcp_vec = (
                field_embs.to(device), step_idx.to(device), mcp_vec.to(device)
            )
            step_logits, mcp_logits, _ = model(
                graphs.x, graphs.edge_index, graphs.batch, field_embs
            )
            step_preds.append(step_logits.argmax(-1).cpu().numpy())
            step_gold.append(step_idx.cpu().numpy())
            mcp_probs.append(torch.sigmoid(mcp_logits).cpu().numpy())
            mcp_preds.append((torch.sigmoid(mcp_logits) >= threshold).float().cpu().numpy())
            mcp_gold.append(mcp_vec.cpu().numpy())

    step_preds = np.concatenate(step_preds)
    step_gold = np.concatenate(step_gold)
    mcp_preds = np.concatenate(mcp_preds)
    mcp_gold = np.concatenate(mcp_gold)
    mcp_probs = np.concatenate(mcp_probs) if return_probs else None

    metrics = {
        "step_accuracy": accuracy_score(step_gold, step_preds),
        "step_macro_f1": f1_score(step_gold, step_preds, average="macro", zero_division=0),
        "step_weighted_f1": f1_score(step_gold, step_preds, average="weighted", zero_division=0),
        "mcp_subset_accuracy": accuracy_score(mcp_gold, mcp_preds),  # exact set match
        "mcp_micro_f1": f1_score(mcp_gold, mcp_preds, average="micro", zero_division=0),
        "mcp_macro_f1": f1_score(mcp_gold, mcp_preds, average="macro", zero_division=0),
        "mcp_samples_f1": f1_score(mcp_gold, mcp_preds, average="samples", zero_division=0),
    }
    
    if return_probs:
        return metrics, mcp_probs, mcp_gold
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 1] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 1] Test input     : {INPUT_TEST_JSON}")

    full_ds = Stage1Dataset(INPUT_TRAIN_JSON, split="train")
    n = len(full_ds)
    val_n = max(1, int(0.1 * n))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)

    # ── Calculate MCP class weights for imbalanced data ─────────────────
    print("[Stage 1] Calculating MCP class weights for imbalanced data...")
    mcp_counts = np.zeros(len(MCP_LABELS))
    for idx in train_idx:
        mcp_counts += full_ds[idx]["mcp_vec"].numpy()
    
    # Calculate inverse frequency weights (higher weight for rare classes)
    total_samples = mcp_counts.sum()
    class_frequencies = mcp_counts / (total_samples + 1e-8)
    # Use inverse frequency with smoothing to avoid extreme weights
    mcp_class_weights = 1.0 / (class_frequencies + 1e-6)
    # Normalize weights to have mean of 1.0
    mcp_class_weights = mcp_class_weights / mcp_class_weights.mean()
    mcp_class_weights = torch.tensor(mcp_class_weights, dtype=torch.float32, device=device)
    
    print(f"[Stage 1] MCP class weights: {mcp_class_weights.cpu().numpy()}")
    for i, label in enumerate(MCP_LABELS):
        print(f"  {label}: {mcp_class_weights[i].item():.3f} (count: {int(mcp_counts[i])})")

    model = Stage1Classifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE1_EPOCHS)

    best_f1 = -1
    for epoch in range(STAGE1_EPOCHS):
        model.train()
        total_loss = 0.0
        for graphs, field_embs, step_idx, mcp_vec in train_loader:
            graphs = graphs.to(device)
            field_embs, step_idx, mcp_vec = (
                field_embs.to(device), step_idx.to(device), mcp_vec.to(device)
            )
            step_logits, mcp_logits, _ = model(
                graphs.x, graphs.edge_index, graphs.batch, field_embs
            )
            loss, step_l, mcp_l = model.loss(
                step_logits, mcp_logits, step_idx, mcp_vec,
                step_w=STEP_LOSS_WEIGHT, mcp_w=MCP_LOSS_WEIGHT,
                mcp_class_weights=mcp_class_weights,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | train_loss {total_loss/len(train_loader):.4f} | "
              f"val_step_acc {val_metrics['step_accuracy']:.3f} "
              f"val_step_macroF1 {val_metrics['step_macro_f1']:.3f} | "
              f"val_mcp_microF1 {val_metrics['mcp_micro_f1']:.3f} "
              f"val_mcp_subsetAcc {val_metrics['mcp_subset_accuracy']:.3f}")

        score = val_metrics["step_macro_f1"] + val_metrics["mcp_micro_f1"]
        if score > best_f1:
            best_f1 = score
            torch.save(model.state_dict(), STAGE1_CKPT)
            print(f"  -> saved new best checkpoint to {STAGE1_CKPT}")

    print("Stage 1 training complete. Best combined score:", best_f1)
    
    # ── Optimize per-class MCP thresholds on validation set ─────────────
    print("\n[Stage 1] Optimizing per-class MCP thresholds on validation set...")
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    val_metrics, mcp_probs, mcp_gold = evaluate(model, val_loader, device, return_probs=True)
    mcp_thresholds = search_per_class_thresholds(mcp_probs, mcp_gold)
    print(f"[Stage 1] Optimized thresholds: {[round(t, 2) for t in mcp_thresholds]}")
    
    # Re-evaluate with optimized thresholds
    print("[Stage 1] Re-evaluating with optimized thresholds...")
    # Apply optimized thresholds
    mcp_preds_opt = (mcp_probs >= np.array(mcp_thresholds)).astype(int)
    opt_micro_f1 = f1_score(mcp_gold, mcp_preds_opt, average="micro", zero_division=0)
    opt_macro_f1 = f1_score(mcp_gold, mcp_preds_opt, average="macro", zero_division=0)
    print(f"[Stage 1] Optimized MCP Micro F1: {opt_micro_f1:.4f} (was {val_metrics['mcp_micro_f1']:.4f})")
    print(f"[Stage 1] Optimized MCP Macro F1: {opt_macro_f1:.4f} (was {val_metrics['mcp_macro_f1']:.4f})")
    
    # Save checkpoint with optimized thresholds
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mcp_thresholds": [float(t) for t in mcp_thresholds],
        "mcp_class_weights": [float(w) for w in mcp_class_weights.cpu().numpy()],
        "best_epoch": best_f1,
        "best_score": best_f1,
    }
    torch.save(checkpoint, STAGE1_CKPT)
    print(f"[Stage 1] Saved checkpoint with optimized thresholds to {STAGE1_CKPT}")


if __name__ == "__main__":
    main()
