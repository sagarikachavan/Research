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
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score, f1_score,
)

# ── Path bootstrap (folder was restructured into core/ data_prep/ training/ eval/) ──
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "core"), _os.path.join(_ROOT, "data_prep"), _os.path.join(_ROOT, "training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

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
torch.cuda.manual_seed_all(RANDOM_SEED)

# Use seeded reproducibility instead of strict deterministic algorithms.
# The latter triggers CuBLAS warnings on modern CUDA while providing little
# practical benefit for this project when the data split and sampler are fixed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)


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


def compute_count_based_class_weights(counts, beta=0.999, min_weight=0.2, max_weight=2.8):
    """Deterministic class weights from the effective sample count.

    This follows the class-imbalance correction used in effective number weighting:
      w_c = (1 - beta) / (1 - beta**n_c)
    where n_c is the count for class c. It is a direct function of the dataset
    counts, not of random sampling or runtime state, so the assignment stays fixed
    across runs for the same data.

    We keep a moderate floor and cap so rare classes remain boosted without
    making the step head systematically underfit the dominant step classes.
    """
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.full(counts.shape, min_weight, dtype=np.float64)
    positive = counts > 0
    if positive.any():
        effective = (1.0 - np.power(beta, counts[positive])) / (1.0 - beta)
        inverse_eff = 1.0 / effective
        inverse_eff = inverse_eff / np.mean(inverse_eff)
        weights[positive] = inverse_eff
    weights = np.clip(weights, min_weight, max_weight)
    return weights


def adaptive_cost_sensitive_loss(step_logits, step_labels, base_weights, 
                                  class_counts, alpha=0.1, beta=0.5):
    """
    Adaptive cost-sensitive loss that adjusts class weights based on performance.
    
    Args:
        step_logits: Model predictions (B, num_classes)
        step_labels: Ground truth labels (B,)
        base_weights: Base class weights from frequency
        class_counts: Number of samples per class
        alpha: Learning rate for weight adaptation
        beta: Balance between frequency-based and performance-based weights
    
    Returns:
        loss: Computed loss
        updated_weights: Updated class weights
    """
    # Compute per-class accuracy
    with torch.no_grad():
        preds = step_logits.argmax(dim=-1)
        class_correct = torch.zeros(len(base_weights), device=step_logits.device)
        class_total = torch.zeros(len(base_weights), device=step_logits.device)
        
        for c in range(len(base_weights)):
            mask = (step_labels == c)
            if mask.sum() > 0:
                class_correct[c] = (preds[mask] == c).float().sum()
                class_total[c] = mask.sum()
        
        # Compute per-class accuracy
        class_acc = class_correct / (class_total + 1e-8)
        
        # Performance-based weights: lower accuracy = higher weight
        perf_weights = 1.0 / (class_acc + 1e-8)
        perf_weights = perf_weights / perf_weights.mean()
        
        # Combine frequency-based and performance-based weights
        adaptive_weights = beta * base_weights + (1 - beta) * perf_weights
        
        # Smooth update from base weights to adaptive weights
        updated_weights = (1 - alpha) * base_weights + alpha * adaptive_weights
    
    # Compute weighted cross-entropy loss
    loss = F.cross_entropy(step_logits, step_labels, weight=updated_weights)
    
    return loss, updated_weights


def evaluate(model, loader, device, threshold=0.5, return_probs=False, save_csv=False, csv_path=None, dataset=None):
    model.eval()
    step_preds, step_gold = [], []
    mcp_preds, mcp_gold = [], []
    mcp_probs = []  # Store raw probabilities for threshold optimization
    csv_rows = []  # Store rows for CSV output
    global_idx = 0  # Track global index for matching with dataset
    if threshold is None:
        threshold = 0.5
    
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
            if isinstance(threshold, (list, tuple, np.ndarray)):
                threshold_tensor = torch.as_tensor(threshold, dtype=mcp_logits.dtype, device=mcp_logits.device)
                mcp_preds.append((torch.sigmoid(mcp_logits) >= threshold_tensor).float().cpu().numpy())
            else:
                threshold_value = 0.5 if threshold is None else float(threshold)
                mcp_preds.append((torch.sigmoid(mcp_logits) >= threshold_value).float().cpu().numpy())
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


def tune_mcp_thresholds(model, loader, device):
    """Tune per-class MCP thresholds on validation data with support-gated search."""
    model.eval()
    all_probs, all_targets = [], []
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
            all_probs.append(torch.sigmoid(mcp_logits).cpu().numpy())
            all_targets.append(mcp_vec.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    thresholds = search_per_class_thresholds(
        probs,
        targets,
        min_val_positives=10,
        candidate_floor=0.25,
        candidate_ceil=0.75,
        n_bootstrap=20,
        bootstrap_seed=RANDOM_SEED,
        auto_fallback=True,
        verbose=True,
    )
    return thresholds


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

    # Keep a small machine-level validation split. This reduces training size a bit,
    # but gives a much more reliable checkpoint-selection signal for this small,
    # imbalanced dataset.
    machine_to_idx = {}
    for idx, ex in enumerate(full_ds.examples):
        machine_to_idx.setdefault(ex["machine"], []).append(idx)
    all_machines = sorted(machine_to_idx)
    val_machine_count = min(max(2, len(all_machines) // 10), max(2, len(all_machines) - 1))
    rng = random.Random(RANDOM_SEED)
    val_machines = set(rng.sample(all_machines, k=val_machine_count))
    val_idx = sorted(i for m in val_machines for i in machine_to_idx[m])
    train_idx = [i for i in range(n) if i not in set(val_idx)]

    # ── Data leakage pre-check: train machines vs TEST machines ──
    test_examples_precheck = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = set(e["machine"] for e in test_examples_precheck)
    train_machine_set = set(e["machine"] for e in full_ds.examples)
    train_test_overlap = train_machine_set & test_machines
    if train_test_overlap:
        print(f"[Stage 1] ⚠  WARNING: TRAIN/TEST machine overlap: {sorted(train_test_overlap)}")
    else:
        print(f"[Stage 1] ✓ No machine overlap between train and test sets")
    del test_examples_precheck

    print(f"[Stage 1] Train machines  : {len(set(e['machine'] for e in [full_ds.examples[i] for i in train_idx]))}")
    print(f"[Stage 1] Val machines    : {len(val_machines)}")
    print(f"[Stage 1] Train examples  : {len(train_idx)}")
    print(f"[Stage 1] Val examples    : {len(val_idx)}")
    print(f"[Stage 1] Using machine-based validation split")

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    # ── Calculate MCP class weights for imbalanced data ─────────────────
    print("[Stage 1] Calculating MCP class weights for imbalanced data...")
    mcp_counts = np.zeros(len(MCP_LABELS))
    for idx in train_idx:
        mcp_counts += full_ds[idx]["mcp_vec"].numpy()

    # Use the effective sample count formula, which depends only on the actual
    # class counts for the fixed dataset. This removes random drift and keeps the
    # balanced weighting behavior stable across runs.
    mcp_class_weights = compute_count_based_class_weights(mcp_counts, beta=0.999, min_weight=0.2, max_weight=2.8)
    mcp_class_weights = torch.tensor(mcp_class_weights, dtype=torch.float32, device=device)

    print(f"[Stage 1] MCP class weights:")
    for i, label in enumerate(MCP_LABELS):
        print(f"  {label:<22}: w={mcp_class_weights[i].item():.3f}  count={int(mcp_counts[i])}")

    # ── Calculate STEP class weights for imbalanced data ─────────────────
    print("\n[Stage 1] Calculating STEP class weights for imbalanced data...")
    step_counts = np.zeros(len(STEP_LABELS))
    for idx in train_idx:
        step_counts[full_ds[idx]["step_idx"].item()] += 1
    step_class_weights = compute_count_based_class_weights(step_counts, beta=0.999, min_weight=0.2, max_weight=2.8)
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
            w = max(w, float(mcp_w_np[active].max()) * 0.70)
        # Keep the sampler mild: the class-loss weights already provide class balancing,
        # so aggressive oversampling on the step head suppresses the dominant classes and
        # drives the model away from the target step accuracy.
        if step_counts[step_i] < 80:
            w *= 1.0
        elif step_counts[step_i] > 250:
            w *= 0.98
        sample_weights[pos] = w

    sampler_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_idx),
        replacement=True,
        generator=sampler_generator,
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

    # Validation-driven checkpointing and early stopping for this small, noisy graph dataset.
    # Best checkpointing should be driven by the actual validation objective (combined score),
    # while the target thresholds are used as a secondary signal for whether the model is in the
    # desired operating window. Requiring the target thresholds at every checkpoint candidate is
    # too strict for early training and prevents any valid best checkpoint from ever being saved.
    early_stop_patience = 12
    tolerance = 1e-4
    step_target = 0.80
    mcp_target = 0.70
    patience_counter = 0
    best_val_score = -float("inf")
    best_epoch = -1
    best_state_dict = None
    train_losses, val_scores = [], []
    run_tag = f"seed{RANDOM_SEED}_run{len(os.listdir(os.path.dirname(STAGE1_CKPT))) if os.path.exists(os.path.dirname(STAGE1_CKPT)) else 0}"
    best_checkpoint_path = STAGE1_CKPT.replace(".pt", f"_{run_tag}_best.pt")
    # Remove stale training artifacts from previous runs so evaluation always loads the
    # current run's checkpoint instead of an older file with a matching name.
    for stale_path in [STAGE1_CKPT, best_checkpoint_path]:
        if os.path.exists(stale_path):
            os.remove(stale_path)
    val_loader = DataLoader(val_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)

    for epoch in range(STAGE1_EPOCHS):
        model.train()
        total_loss = 0.0
        step_losses, mcp_losses = 0.0, 0.0
        n_batches = 0
        
        # Enable adaptive cost-sensitive learning after warmup
        use_adaptive_loss = epoch >= STAGE1_WARMUP_EPOCHS

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
            
            # Stronger rare-class handling: focal-style step loss plus weighted BCE.
            # This is more robust than plain CE on the small and imbalanced class set.
            step_l = F.cross_entropy(step_logits, step_idx, weight=step_class_weights, reduction='none')
            step_pt = torch.exp(-step_l)
            step_focal = (1.0 - step_pt) ** 1.0
            step_l = (step_focal * step_l).mean()

            if mcp_class_weights is not None:
                mcp_l = F.binary_cross_entropy_with_logits(mcp_logits, mcp_vec, reduction='none')
                mcp_pt = torch.exp(-mcp_l)
                mcp_focal = (1.0 - mcp_pt) ** 1.3
                mcp_l = (mcp_focal * mcp_l * mcp_class_weights).mean()
            else:
                mcp_l = F.binary_cross_entropy_with_logits(mcp_logits, mcp_vec)

            loss = STEP_LOSS_WEIGHT * step_l + MCP_LOSS_WEIGHT * mcp_l
            
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
        train_losses.append(total_loss / n_batches)

        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        val_combined = 0.5 * val_metrics["step_accuracy"] + 0.5 * val_metrics["mcp_micro_f1"]
        val_scores.append(val_combined)

        meets_target = (
            val_metrics["step_accuracy"] >= step_target and
            val_metrics["mcp_micro_f1"] >= mcp_target
        )
        if val_combined > best_val_score + tolerance:
            best_val_score = val_combined
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "mcp_class_weights": [float(w) for w in mcp_class_weights.cpu().numpy()],
                "step_class_weights": [float(w) for w in step_class_weights.cpu().numpy()],
                "train_losses": train_losses,
                "val_scores": val_scores,
                "best_val_epoch": best_epoch,
                "best_val_score": best_val_score,
                "random_seed": RANDOM_SEED,
                "run_tag": run_tag,
                "target_met": bool(meets_target),
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            torch.save(checkpoint, STAGE1_CKPT)
            torch.save(checkpoint, best_checkpoint_path)
            if meets_target:
                print(f"[Stage 1] Saved target checkpoint to {STAGE1_CKPT} and {best_checkpoint_path} (step={val_metrics['step_accuracy']:.4f}, mcp={val_metrics['mcp_micro_f1']:.4f})")
            else:
                print(f"[Stage 1] Saved best validation checkpoint to {STAGE1_CKPT} and {best_checkpoint_path} (val_combined={best_val_score:.4f}, step={val_metrics['step_accuracy']:.4f}, mcp={val_metrics['mcp_micro_f1']:.4f})")
        else:
            patience_counter += 1

        print(
            f"epoch {epoch+1:02d}/{STAGE1_EPOCHS} | "
            f"lr {current_lr:.2e} | "
            f"train_loss {total_loss/n_batches:.4f} "
            f"(step={step_losses/n_batches:.4f}, mcp={mcp_losses/n_batches:.4f}) | "
            f"val_combined {val_combined:.4f}"
        )

        if patience_counter >= early_stop_patience:
            print(f"[Stage 1] Early stopping triggered after {epoch + 1} epochs; best validation score was {best_val_score:.4f} at epoch {best_epoch}.")
            break

    print(f"\n[Stage 1] Training complete. Trained for {epoch + 1} epochs.")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[Stage 1] Restored best validation checkpoint from epoch {best_epoch} (val_combined={best_val_score:.4f})")
    else:
        print("[Stage 1] No validation improvement detected; keeping final model state.")

    # Tune per-class MCP thresholds on validation data and use them for test evaluation.
    # This avoids the common failure mode where a uniform 0.5 threshold is too harsh on
    # imbalanced multi-label heads, especially for sparse classes.
    mcp_thresholds = tune_mcp_thresholds(model, val_loader, device)
    print(f"[Stage 1] Tuned MCP thresholds: {mcp_thresholds}")

    # Save final best checkpoint metadata in the canonical checkpoint path.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mcp_class_weights": [float(w) for w in mcp_class_weights.cpu().numpy()],
        "step_class_weights": [float(w) for w in step_class_weights.cpu().numpy()],
        "train_losses": train_losses,
        "val_scores": val_scores,
        "best_val_epoch": best_epoch,
        "best_val_score": best_val_score,
        "random_seed": RANDOM_SEED,
        "run_tag": run_tag,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    torch.save(checkpoint, STAGE1_CKPT)
    torch.save(checkpoint, best_checkpoint_path)
    print(f"[Stage 1] Final saved checkpoint: {STAGE1_CKPT}")

    # ── Evaluate on test set and save CSV ─────────────────────────────────────
    print("\n[Stage 1] Evaluating on test set...")
    test_ds = Stage1Dataset(INPUT_TEST_JSON, split="test")
    test_loader = DataLoader(test_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_examples = test_ds.examples

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