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
import csv
import os

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
    STEP_LABELS, ROOT, STEP_LABEL_SMOOTHING, STAGE1_WARMUP_EPOCHS, STAGE1_GRAD_CLIP,
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


def evaluate(model, loader, device, threshold=0.5, return_probs=False, save_csv=False, csv_path=None, dataset=None):
    model.eval()
    step_preds, step_gold = [], []
    mcp_preds, mcp_gold = [], []
    mcp_probs = []  # Store raw probabilities for threshold optimization
    csv_rows = []  # Store rows for CSV output
    global_idx = 0  # Track global index for matching with dataset
    
    with torch.no_grad():
        for graphs, field_embs, step_idx, mcp_vec in loader:
            graphs = graphs.to(device)
            field_embs, step_idx, mcp_vec = (
                field_embs.to(device), step_idx.to(device), mcp_vec.to(device)
            )
            edge_attr = getattr(graphs, 'edge_attr', None)
            step_logits, mcp_logits, _ = model(
                graphs.x, graphs.edge_index, graphs.batch, field_embs,
                edge_attr=edge_attr,
            )
            step_preds.append(step_logits.argmax(-1).cpu().numpy())
            step_gold.append(step_idx.cpu().numpy())
            mcp_probs.append(torch.sigmoid(mcp_logits).cpu().numpy())
            # Handle threshold as list for per-class thresholds
            if isinstance(threshold, (list, np.ndarray)):
                threshold_tensor = torch.tensor(threshold, dtype=torch.float32, device=device)
                mcp_preds.append((torch.sigmoid(mcp_logits) >= threshold_tensor).float().cpu().numpy())
            else:
                mcp_preds.append((torch.sigmoid(mcp_logits) >= threshold).float().cpu().numpy())
            mcp_gold.append(mcp_vec.cpu().numpy())
            
            # Collect data for CSV output
            if save_csv and dataset is not None:
                batch_size = step_idx.shape[0]
                for i in range(batch_size):
                    if global_idx < len(dataset):
                        ex = dataset[global_idx]
                        pred_step_idx = step_preds[-1][i]
                        gold_step_idx = step_gold[-1][i]
                        pred_mcp = mcp_preds[-1][i]
                        gold_mcp = mcp_gold[-1][i]
                        
                        # Convert indices to labels
                        pred_step_label = STEP_LABELS[pred_step_idx] if 0 <= pred_step_idx < len(STEP_LABELS) else "UNPARSEABLE"
                        gold_step_label = STEP_LABELS[gold_step_idx] if 0 <= gold_step_idx < len(STEP_LABELS) else "UNPARSEABLE"
                        
                        # Convert MCP vectors to tool names
                        pred_mcp_tools = [MCP_LABELS[j] for j in range(len(MCP_LABELS)) if pred_mcp[j] == 1]
                        gold_mcp_tools = [MCP_LABELS[j] for j in range(len(MCP_LABELS)) if gold_mcp[j] == 1]
                        
                        csv_rows.append({
                            "machine": ex.get("machine", ""),
                            "new_strategy": ex["context"].get("New strategy", ""),
                            "strategy_explanation": ex["context"].get("Strategy explanation", ""),
                            "step_prediction": pred_step_label,
                            "gold_new_step": gold_step_label,
                            "mcp_tool_prediction": "|".join(pred_mcp_tools),
                            "mcp_tool_gold": "|".join(gold_mcp_tools),
                        })
                    global_idx += 1

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
    
    # Save CSV if requested
    if save_csv and csv_path and csv_rows:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Stage 1] Evaluation CSV saved to: {csv_path}")
    
    if return_probs:
        return metrics, mcp_probs, mcp_gold
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 1] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 1] Test input     : {INPUT_TEST_JSON}")
    print(f"[Stage 1] Device         : {device}")
    print(f"[Stage 1] Epochs         : {STAGE1_EPOCHS} (warmup {STAGE1_WARMUP_EPOCHS})")
    print(f"[Stage 1] Label smoothing: {STEP_LABEL_SMOOTHING}")
    print(f"[Stage 1] Grad clip      : {STAGE1_GRAD_CLIP}")

    full_ds = Stage1Dataset(INPUT_TRAIN_JSON, split="train")
    n = len(full_ds)

    # DATA LEAKAGE PREVENTION: split by MACHINE ID (not by example index)
    # This ensures no machine has data in BOTH train AND validation sets
    all_examples_for_split = full_ds.examples
    all_machines = sorted(set(e["machine"] for e in all_examples_for_split))
    rng_split = np.random.default_rng(RANDOM_SEED + 1)
    perm_machines = rng_split.permutation(len(all_machines))
    # Use STAGE2_VAL_SPLIT ratio for consistency across stages, default 15%
    from config import STAGE2_VAL_SPLIT
    val_split_ratio = STAGE2_VAL_SPLIT if STAGE2_VAL_SPLIT else 0.15
    n_val_machines = max(1, int(len(all_machines) * val_split_ratio))
    val_machine_set = set(all_machines[i] for i in perm_machines[:n_val_machines])
    train_machine_set = set(all_machines) - val_machine_set

    # Map machines back to example indices
    train_idx = [i for i, e in enumerate(all_examples_for_split) if e["machine"] in train_machine_set]
    val_idx = [i for i, e in enumerate(all_examples_for_split) if e["machine"] in val_machine_set]

    # ── Data leakage pre-check: train machines vs TEST machines ──
    test_examples_precheck = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = set(e["machine"] for e in test_examples_precheck)
    train_test_overlap = train_machine_set & test_machines
    val_test_overlap = val_machine_set & test_machines
    if train_test_overlap:
        print(f"[Stage 1] ⚠  WARNING: TRAIN/TEST machine overlap: {sorted(train_test_overlap)}")
    if val_test_overlap:
        print(f"[Stage 1] ⚠  WARNING: VAL/TEST machine overlap: {sorted(val_test_overlap)}")
    if not train_test_overlap and not val_test_overlap:
        print(f"[Stage 1] ✓ No machine overlap between (train ∪ val) and test sets")
    del test_examples_precheck

    print(f"[Stage 1] Train machines  : {len(train_machine_set)}")
    print(f"[Stage 1] Val machines    : {len(val_machine_set)}")
    print(f"[Stage 1] Train examples  : {len(train_idx)}")
    print(f"[Stage 1] Val examples    : {len(val_idx)}")

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    val_loader = DataLoader(val_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)
    # train_loader is built further below (after class weights are computed)
    # using a WeightedRandomSampler instead of plain shuffle=True.

    # ── Calculate MCP class weights for imbalanced data ─────────────────
    print("[Stage 1] Calculating MCP class weights for imbalanced data...")
    mcp_counts = np.zeros(len(MCP_LABELS))
    for idx in train_idx:
        mcp_counts += full_ds[idx]["mcp_vec"].numpy()

    total_samples_mcp = mcp_counts.sum()
    class_frequencies = mcp_counts / (total_samples_mcp + 1e-8)
    mcp_class_weights = 1.0 / (class_frequencies + 1e-6)
    rare_class_indices = [i for i, count in enumerate(mcp_counts) if count < 15]
    for idx in rare_class_indices:
        mcp_class_weights[idx] *= 2.5
    mcp_class_weights = mcp_class_weights / mcp_class_weights.mean()
    mcp_class_weights = torch.tensor(mcp_class_weights, dtype=torch.float32, device=device)

    print(f"[Stage 1] MCP class weights:")
    for i, label in enumerate(MCP_LABELS):
        print(f"  {label:<22}: w={mcp_class_weights[i].item():.3f}  count={int(mcp_counts[i])}")

    # ── Calculate STEP class weights for imbalanced data ─────────────────
    print("\n[Stage 1] Calculating STEP class weights for imbalanced data...")
    step_counts = np.zeros(len(STEP_LABELS))
    for idx in train_idx:
        step_counts[full_ds[idx]["step_idx"].item()] += 1
    total_samples_step = step_counts.sum()
    step_freq = step_counts / (total_samples_step + 1e-8)
    step_class_weights = 1.0 / (step_freq + 1e-6)
    rare_step_idx = [i for i, c in enumerate(step_counts) if c < 10]
    for idx in rare_step_idx:
        step_class_weights[idx] *= 2.0
    # Handle zero-count classes: set weight to 0 (they shouldn't contribute to loss)
    zero_count_mask = step_counts == 0
    step_class_weights[zero_count_mask] = 0.0
    # Normalize only non-zero weights
    non_zero_mean = step_class_weights[~zero_count_mask].mean()
    if non_zero_mean > 0:
        step_class_weights[~zero_count_mask] = step_class_weights[~zero_count_mask] / non_zero_mean
    step_class_weights = torch.tensor(step_class_weights, dtype=torch.float32, device=device)

    print(f"[Stage 1] STEP class weights:")
    for i, label in enumerate(STEP_LABELS):
        print(f"  [{i}] {label[:50]:<50}: w={step_class_weights[i].item():.3f}  count={int(step_counts[i])}")

    # ── Rare-class oversampling ──────────────────────────────────────────
    # Loss reweighting (above) changes how much a rare-class example counts
    # toward the gradient, but every example still appears exactly once per
    # epoch regardless of shuffle=True. For very thin classes (e.g. SQLmap
    # ~24 rows, hydra ~23 rows) that's not enough signal per epoch. Build a
    # WeightedRandomSampler so rare-class rows are actually drawn more often.
    print("\n[Stage 1] Building rare-class-aware sampler for training...")
    mcp_w_np = mcp_class_weights.detach().cpu().numpy()
    step_w_np = step_class_weights.detach().cpu().numpy()
    sample_weights = np.ones(len(train_idx), dtype=np.float64)
    for pos, idx in enumerate(train_idx):
        ex = full_ds[idx]
        step_i = ex["step_idx"].item()
        mcp_vec_i = ex["mcp_vec"].numpy()
        w = float(step_w_np[step_i])
        active = mcp_vec_i > 0
        if active.any():
            w = max(w, float(mcp_w_np[active].max()))
        sample_weights[pos] = w

    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_idx),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=STAGE1_BATCH_SIZE, sampler=train_sampler,
        collate_fn=collate, drop_last=False,
    )

    model = Stage1Classifier().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Stage 1] Trainable parameters: {n_params:,}")

    from config import STAGE1_WEIGHT_DECAY
    opt = torch.optim.AdamW(
        model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8,
    )

    # Cosine annealing with warmup
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * STAGE1_EPOCHS
    warmup_steps = steps_per_epoch * STAGE1_WARMUP_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_f1 = -1
    best_epoch = -1
    no_improve = 0
    early_stop_patience = 8

    # ── Top-K checkpoint averaging (SWA-lite) ────────────────────────────
    # The val set is small (239 examples / 26 machines), so val_score is
    # noisy epoch-to-epoch (see run log: 0.013 -> 0.373 -> 0.426 -> 0.572
    # -> ...). Trusting a single "best" epoch means trusting whichever
    # epoch happened to land on a lucky val batch. Instead, keep the K
    # checkpoints with the highest val_score seen during training and
    # average their weights at the end. This is cheap (K small state
    # dicts on CPU) and, unlike raising model capacity, doesn't add
    # overfitting risk. It's checked against the single-best checkpoint
    # on validation before being trusted -- see below.
    TOPK_SWA = 3
    topk_checkpoints = []  # list of (val_score, epoch, cpu_state_dict)

    train_losses, val_scores = [], []

    for epoch in range(STAGE1_EPOCHS):
        model.train()
        total_loss = 0.0
        step_losses, mcp_losses = 0.0, 0.0
        n_batches = 0

        for graphs, field_embs, step_idx, mcp_vec in train_loader:
            graphs = graphs.to(device)
            field_embs, step_idx, mcp_vec = (
                field_embs.to(device), step_idx.to(device), mcp_vec.to(device)
            )
            edge_attr = getattr(graphs, 'edge_attr', None)
            step_logits, mcp_logits, _ = model(
                graphs.x, graphs.edge_index, graphs.batch, field_embs,
                edge_attr=edge_attr,
            )
            loss, step_l, mcp_l = model.loss(
                step_logits, mcp_logits, step_idx, mcp_vec,
                step_w=STEP_LOSS_WEIGHT, mcp_w=MCP_LOSS_WEIGHT,
                mcp_class_weights=mcp_class_weights,
                use_focal=True, focal_gamma=2.0,
                label_smoothing=STEP_LABEL_SMOOTHING,
                step_class_weights=step_class_weights,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), STAGE1_GRAD_CLIP)
            opt.step()
            sched.step()
            total_loss += loss.item()
            step_losses += step_l.item()
            mcp_losses += mcp_l.item()
            n_batches += 1

        current_lr = sched.get_last_lr()[0]
        val_metrics = evaluate(model, val_loader, device)
        train_losses.append(total_loss / n_batches)
        val_score = val_metrics["step_macro_f1"] + val_metrics["mcp_micro_f1"]
        val_scores.append(val_score)

        print(
            f"epoch {epoch+1:02d}/{STAGE1_EPOCHS} | "
            f"lr {current_lr:.2e} | "
            f"train_loss {total_loss/n_batches:.4f} "
            f"(step={step_losses/n_batches:.4f}, mcp={mcp_losses/n_batches:.4f}) | "
            f"val_step_acc {val_metrics['step_accuracy']:.3f} "
            f"val_step_macroF1 {val_metrics['step_macro_f1']:.3f} | "
            f"val_mcp_microF1 {val_metrics['mcp_micro_f1']:.3f} "
            f"val_mcp_subsetAcc {val_metrics['mcp_subset_accuracy']:.3f} | "
            f"score {val_score:.4f}"
        )

        # Maintain the top-K checkpoints by val_score for later SWA averaging.
        # CPU clone so we're not holding K copies of the model on GPU.
        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        topk_checkpoints.append((val_score, epoch + 1, cpu_state))
        topk_checkpoints.sort(key=lambda t: t[0], reverse=True)
        topk_checkpoints = topk_checkpoints[:TOPK_SWA]

        if val_score > best_f1:
            best_f1 = val_score
            best_epoch = epoch + 1
            no_improve = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_score": best_f1,
                "train_losses": train_losses,
                "val_scores": val_scores,
            }
            torch.save(checkpoint, STAGE1_CKPT)
            print(f"  -> saved new best checkpoint (score={best_f1:.4f}) to {STAGE1_CKPT}")
        else:
            no_improve += 1
            print(f"  -> no improvement ({no_improve}/{early_stop_patience})")
            if no_improve >= early_stop_patience:
                print(f"\n[Stage 1] Early stopping at epoch {epoch+1}. Best was epoch {best_epoch} (score={best_f1:.4f})")
                break

    print(f"\n[Stage 1] Training complete. Best combined score: {best_f1:.4f} (epoch {best_epoch})")

    # ── Try SWA: average the top-K checkpoints' weights ──────────────────
    swa_epochs = [e for _, e, _ in topk_checkpoints]
    print(f"\n[Stage 1] Averaging top-{len(topk_checkpoints)} checkpoints (epochs {swa_epochs}) for SWA candidate...")
    swa_state = {}
    for key in topk_checkpoints[0][2].keys():
        stacked = torch.stack([sd[key].float() for _, _, sd in topk_checkpoints], dim=0)
        swa_state[key] = stacked.mean(dim=0)

    swa_model = Stage1Classifier().to(device)
    swa_model.load_state_dict(swa_state)
    swa_val_metrics = evaluate(swa_model, val_loader, device)
    swa_val_score = swa_val_metrics["step_macro_f1"] + swa_val_metrics["mcp_micro_f1"]

    print(f"[Stage 1] SWA val score: {swa_val_score:.4f}  (single-best val score: {best_f1:.4f})")

    if swa_val_score > best_f1:
        print(f"[Stage 1] SWA beats single-best checkpoint -> using SWA weights.")
        best_f1 = swa_val_score
        checkpoint = {
            "model_state_dict": swa_state,
            "best_epoch": f"swa({swa_epochs})",
            "best_score": best_f1,
            "train_losses": train_losses,
            "val_scores": val_scores,
        }
        torch.save(checkpoint, STAGE1_CKPT)
    else:
        print(f"[Stage 1] Single-best checkpoint (epoch {best_epoch}) still wins on val -> keeping it.")

    # ── Optimize per-class MCP thresholds on validation set ─────────────
    print("\n[Stage 1] Optimizing per-class MCP thresholds on validation set...")
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics, mcp_probs, mcp_gold = evaluate(model, val_loader, device, return_probs=True)

    print(f"[Stage 1] Rare class indices for aggressive threshold tuning: {rare_class_indices}")
    mcp_thresholds = search_per_class_thresholds(mcp_probs, mcp_gold, rare_class_indices=rare_class_indices)
    print(f"[Stage 1] Optimized thresholds: {[round(t, 2) for t in mcp_thresholds]}")

    mcp_preds_opt = (mcp_probs >= np.array(mcp_thresholds)).astype(int)
    opt_micro_f1 = f1_score(mcp_gold, mcp_preds_opt, average="micro", zero_division=0)
    opt_macro_f1 = f1_score(mcp_gold, mcp_preds_opt, average="macro", zero_division=0)
    print(f"[Stage 1] Optimized MCP Micro F1: {opt_micro_f1:.4f} (was {val_metrics['mcp_micro_f1']:.4f})")
    print(f"[Stage 1] Optimized MCP Macro F1: {opt_macro_f1:.4f} (was {val_metrics['mcp_macro_f1']:.4f})")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mcp_thresholds": [float(t) for t in mcp_thresholds],
        "mcp_class_weights": [float(w) for w in mcp_class_weights.cpu().numpy()],
        "step_class_weights": [float(w) for w in step_class_weights.cpu().numpy()],
        "best_epoch": best_epoch,
        "best_score": best_f1,
        "train_losses": ckpt.get("train_losses", []),
        "val_scores": ckpt.get("val_scores", []),
    }
    torch.save(checkpoint, STAGE1_CKPT)
    print(f"[Stage 1] Saved checkpoint with optimized thresholds to {STAGE1_CKPT}")

    # ── Evaluate on test set and save CSV ─────────────────────────────────────
    print("\n[Stage 1] Evaluating on test set...")
    test_ds = Stage1Dataset(INPUT_TEST_JSON, split="test")
    test_loader = DataLoader(test_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_examples = test_ds.examples

    model.load_state_dict(ckpt["model_state_dict"])

    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "stage1.csv")

    test_metrics = evaluate(
        model, test_loader, device,
        threshold=mcp_thresholds,
        save_csv=True,
        csv_path=csv_path,
        dataset=test_examples
    )

    print(f"\n[Stage 1] ═══════════ TEST SET RESULTS ═══════════")
    print(f"  Step Accuracy     : {test_metrics['step_accuracy']:.4f}  ({test_metrics['step_accuracy']*100:.2f}%)")
    print(f"  Step Macro F1     : {test_metrics['step_macro_f1']:.4f}")
    print(f"  Step Weighted F1  : {test_metrics['step_weighted_f1']:.4f}")
    print(f"  MCP Micro F1      : {test_metrics['mcp_micro_f1']:.4f}")
    print(f"  MCP Macro F1      : {test_metrics['mcp_macro_f1']:.4f}")
    print(f"  MCP Subset Acc    : {test_metrics['mcp_subset_accuracy']:.4f}")
    print(f"  MCP Samples F1    : {test_metrics['mcp_samples_f1']:.4f}")
    combined = test_metrics['step_accuracy'] * 0.5 + test_metrics['mcp_micro_f1'] * 0.5
    print(f"  Combined Score    : {combined:.4f}  (target >= 0.80 for ~80%)")
    print(f"[Stage 1] ════════════════════════════════════════")


if __name__ == "__main__":
    main()