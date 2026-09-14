"""Stage 1: supervised edge-aware graph + strategy classifier.

Input:  machine graph + New strategy + Strategy explanation
Outputs: next Step (single label) + MCP tools (multi-label).

Final Stage-1 contract: New strategy + strategy explanation are encoded by a
frozen GPT-2 semantic CNN; the PTT graph is encoded by typed GINE to a raw
512-d graph representation; the two representations are fused privately for
independent Step and MCP supervised heads. The raw 512-d GINE vector is the
only graph representation exposed to Stage 2/3.
"""
from __future__ import annotations

import csv
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, f1_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "data_prep"), os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, STAGE1_CKPT, STAGE1_LR, STAGE1_EPOCHS,
    STAGE1_BATCH_SIZE, STEP_LOSS_WEIGHT, MCP_LOSS_WEIGHT, RANDOM_SEED,
    MCP_LABELS, STEP_LABELS, ROOT, STEP_LABEL_SMOOTHING, STAGE1_WARMUP_EPOCHS,
    STAGE1_GRAD_CLIP, STAGE1_WEIGHT_DECAY, STAGE1_MAX_CLASS_WEIGHT,
    STAGE1_MAX_MCP_WEIGHT, STAGE1_HARD_NEGATIVE_WEIGHT,
    STAGE1_SUPCON_WEIGHT, STAGE1_HARD_NEGATIVE_MARGIN, STAGE1_USE_STEP_CLASS_WEIGHTS, STAGE2_VAL_SPLIT,
    SEMANTIC_LM_NAME, SEMANTIC_MAX_TOKENS, SEMANTIC_LM_DIM, SEMANTIC_PROTOTYPE_TOKENS,
    STAGE1_USE_STEP_FOCAL, STAGE1_STEP_FOCAL_GAMMA, STAGE1_SWA_TOP_K,
    STAGE1_USE_LOGIT_ADJUSTMENT, STAGE1_LOGIT_ADJ_TAU,
    STAGE1_MCP_LOSS_TYPE, STAGE1_ASL_GAMMA_NEG, STAGE1_ASL_GAMMA_POS, STAGE1_ASL_CLIP,
    STAGE1_USE_MANIFOLD_MIXUP, STAGE1_MIXUP_ALPHA, STAGE1_MIXUP_WEIGHT,
    STAGE1_SUPCON_TEMPERATURE, STAGE1_USE_DECOUPLED_RETRAIN,
    STAGE1_DECOUPLED_EPOCHS, STAGE1_DECOUPLED_LR,
    STAGE1_GRAPH_GATE_LR_MULT,
    STAGE1_USE_KFOLD, STAGE1_N_FOLDS,
)
from data_utils import load_from_input_json, precompute_semantic_tokens
from graph_encoder import Stage1Classifier
from mcp_threshold_search import search_per_class_thresholds, search_step_logit_bias

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class Stage1Dataset(Dataset):
    def __init__(self, json_path, split="train"):
        self.examples = load_from_input_json(json_path, split)
        for ex in self.examples:
            # Stage 1 semantic input is exactly the two allowed context fields.
            texts = [ex["context"].get("New strategy", "") or "empty",
                     ex["context"].get("Strategy explanation", "") or "empty"]
            ex["semantic_text"] = f"{texts[0]} {texts[1]}"
        precompute_semantic_tokens(self.examples, model_name=SEMANTIC_LM_NAME, max_tokens=SEMANTIC_MAX_TOKENS, device="cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "graph": ex["graph"],
            "step_idx": torch.tensor(ex["step_idx"], dtype=torch.long),
            "mcp_vec": torch.tensor(ex["mcp_vec"], dtype=torch.float32),
            "semantic_tokens": ex["semantic_tokens"],
        }


def collate(items):
    graphs = Batch.from_data_list([b["graph"] for b in items])
    tokens = [b["semantic_tokens"] for b in items]
    max_len = max(t.shape[0] for t in tokens)
    d = tokens[0].shape[1]
    sem = torch.zeros(len(tokens), max_len, d, dtype=torch.float32)
    mask = torch.zeros(len(tokens), max_len, dtype=torch.bool)
    for i, t in enumerate(tokens):
        L = t.shape[0]
        sem[i, :L] = t
        mask[i, :L] = True
    return (graphs,
            torch.stack([b["step_idx"] for b in items]),
            torch.stack([b["mcp_vec"] for b in items]),
            sem, mask)


def evaluate(model, loader, device, threshold=0.5, return_probs=False, save_csv=False,
             csv_path=None, dataset=None, step_bias=None, return_step_logits=False):
    """step_bias: optional per-class additive logit bias applied BEFORE argmax
    (see search_step_logit_bias in core/mcp_threshold_search.py). None = plain
    argmax, i.e. the previous behavior.

    `model` may be a single nn.Module OR a list of nn.Modules. With a list this
    runs a K-FOLD ENSEMBLE: step LOGITS are averaged across members and MCP
    SIGMOID PROBABILITIES are averaged across members. Averaging step logits
    (rather than softmax probabilities) is deliberate -- an additive per-class
    bias `b` commutes with the mean, i.e. mean_k(logits_k + b) == mean_k(logits_k) + b,
    so a bias calibrated on single-model out-of-fold logits applies unchanged to
    the ensemble. See the caveat in main()'s K-fold block."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    for _m in models:
        _m.eval()
    step_preds, step_gold = [], []
    step_logit_rows = []
    mcp_preds, mcp_gold, mcp_probs = [], [], []
    csv_rows = []
    global_idx = 0
    with torch.no_grad():
        for graphs, step_idx, mcp_vec, sem_tokens, sem_mask in loader:
            graphs = graphs.to(device)
            step_idx = step_idx.to(device)
            mcp_vec = mcp_vec.to(device)
            sem_tokens = sem_tokens.to(device)
            sem_mask = sem_mask.to(device)
            edge_attr = getattr(graphs, "edge_attr", None)
            sl_sum, mp_sum = None, None
            for _m in models:
                step_logits, mcp_logits, _ = _m(
                    graphs.x, graphs.edge_index, graphs.batch,
                    semantic_tokens=sem_tokens, semantic_mask=sem_mask, edge_attr=edge_attr
                )
                sl = step_logits.detach().float()
                mprob = torch.sigmoid(mcp_logits.detach().float())
                sl_sum = sl if sl_sum is None else sl_sum + sl
                mp_sum = mprob if mp_sum is None else mp_sum + mprob
            step_logits = sl_sum / len(models)
            mcp_prob_t = mp_sum / len(models)

            sl_np = step_logits.cpu().numpy()
            step_logit_rows.append(sl_np)
            if step_bias is not None:
                sp = np.argmax(sl_np + np.asarray(step_bias, dtype=np.float64)[None, :], axis=1)
            else:
                sp = step_logits.argmax(-1).cpu().numpy()
            sg = step_idx.cpu().numpy()
            probs = mcp_prob_t.cpu().numpy()
            if isinstance(threshold, (list, np.ndarray)):
                thr = torch.tensor(threshold, dtype=torch.float32, device=device)
                mp = (mcp_prob_t >= thr).float().cpu().numpy()
            else:
                mp = (mcp_prob_t >= threshold).float().cpu().numpy()
            mg = mcp_vec.cpu().numpy()
            step_preds.append(sp); step_gold.append(sg)
            mcp_preds.append(mp); mcp_gold.append(mg); mcp_probs.append(probs)
            if save_csv and dataset is not None:
                for i in range(len(sp)):
                    ex = dataset[global_idx]
                    csv_rows.append({
                        "machine": ex.get("machine", ""),
                        "new_strategy": ex["context"].get("New strategy", ""),
                        "strategy_explanation": ex["context"].get("Strategy explanation", ""),
                        "step_prediction": STEP_LABELS[int(sp[i])],
                        "gold_new_step": STEP_LABELS[int(sg[i])],
                        "mcp_tool_prediction": "|".join(MCP_LABELS[j] for j in range(len(MCP_LABELS)) if mp[i, j] == 1),
                        "mcp_tool_gold": "|".join(MCP_LABELS[j] for j in range(len(MCP_LABELS)) if mg[i, j] == 1),
                    })
                    global_idx += 1
    step_preds = np.concatenate(step_preds); step_gold = np.concatenate(step_gold)
    mcp_preds = np.concatenate(mcp_preds); mcp_gold = np.concatenate(mcp_gold)
    probs_out = np.concatenate(mcp_probs) if return_probs else None
    metrics = {
        "step_accuracy": accuracy_score(step_gold, step_preds),
        "step_micro_f1": f1_score(step_gold, step_preds, average="micro", zero_division=0),
        "step_macro_f1": f1_score(step_gold, step_preds, average="macro", zero_division=0),
        "step_weighted_f1": f1_score(step_gold, step_preds, average="weighted", zero_division=0),
        "mcp_subset_accuracy": accuracy_score(mcp_gold, mcp_preds),
        "mcp_micro_f1": f1_score(mcp_gold, mcp_preds, average="micro", zero_division=0),
        "mcp_macro_f1": f1_score(mcp_gold, mcp_preds, average="macro", zero_division=0),
        "mcp_samples_f1": f1_score(mcp_gold, mcp_preds, average="samples", zero_division=0),
    }
    if save_csv and csv_path and csv_rows:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader(); writer.writerows(csv_rows)
        print(f"[Stage 1] Evaluation CSV saved to: {csv_path}")
    if return_step_logits:
        step_logits_out = np.concatenate(step_logit_rows, axis=0)
        if return_probs:
            return metrics, probs_out, mcp_gold, step_logits_out, step_gold
        return metrics, step_logits_out, step_gold
    if return_probs:
        return metrics, probs_out, mcp_gold
    return metrics


def _soft_class_weights(counts, cap, rare_boost=1.0):
    counts = np.asarray(counts, dtype=np.float64)
    positive = counts > 0
    weights = np.ones_like(counts)
    if positive.any():
        inv = 1.0 / np.sqrt(np.maximum(counts[positive], 1.0))
        inv = inv / max(inv.mean(), 1e-12)
        weights[positive] = np.clip(inv, 1.0 / cap, cap)
    if rare_boost > 1.0:
        rare = (counts > 0) & (counts < 10)
        weights[rare] = np.minimum(weights[rare] * rare_boost, cap)
    weights[~positive] = 0.0
    nz = weights > 0
    if nz.any():
        weights[nz] /= max(weights[nz].mean(), 1e-12)
    return weights


def hard_negative_margin(logits, labels, groups, margin=0.20):
    """Penalize the known confusing Step alternatives."""
    losses = []
    for group in groups:
        group = list(group)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in group:
            mask |= labels == c
        if not bool(mask.any()):
            continue
        idx = torch.tensor(group, device=logits.device, dtype=torch.long)
        group_logits = logits[mask][:, idx]
        local_labels = labels[mask]
        pos_col = torch.tensor([group.index(int(y)) for y in local_labels.tolist()], device=logits.device, dtype=torch.long)
        pos = group_logits.gather(1, pos_col.unsqueeze(1)).squeeze(1)
        neg = group_logits.masked_fill(F.one_hot(pos_col, len(group)).bool(), -1e9).max(dim=1).values
        losses.append(F.relu(margin - pos + neg).mean())
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def supervised_contrastive_loss(fused_h, labels, temperature=0.10, base_temperature=0.07):
    """Supervised Contrastive Loss (Khosla et al., "Supervised Contrastive
    Learning", NeurIPS 2020, https://arxiv.org/abs/2004.11362), computed on
    Stage-1's fused representation using the Step label to define positive
    pairs.

    ROUND-3 FIX: this is the function STAGE1_SUPCON_WEIGHT was always meant
    to gate. Every previous version of this file computed `con` as a
    hard-coded `fused_h.new_zeros(())` regardless of the weight -- the
    contrastive term was a no-op end to end (see config.py's
    STAGE1_SUPCON_WEIGHT comment). This is the real implementation.

    For each anchor i in the batch, every OTHER row j with the same Step
    label is a positive; every row with a different Step label is a
    negative. The loss pulls same-class fused vectors together and pushes
    different-class vectors apart *within every training batch*, on top of
    (not instead of) the Step head's own cross-entropy -- a much denser
    per-row training signal than CE alone gives a ~1.5k-row dataset, since
    CE only ever compares a row against its own one-hot target while this
    compares it against every other row currently in the batch.

    Standard multi-positive SupCon (mean-of-log-prob-over-positives
    variant, matching the paper's official reference implementation).
    Returns a scalar; rows whose Step class has no other representative in
    this particular batch contribute 0 (nothing to contrast against).
    """
    b = fused_h.size(0)
    if b < 2:
        return fused_h.new_zeros(())
    z = F.normalize(fused_h, dim=-1)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    sim_max = sim.max(dim=1, keepdim=True).values
    sim = sim - sim_max.detach()  # numerical stability, does not change softmax

    self_mask = torch.eye(b, dtype=torch.bool, device=fused_h.device)
    logits_mask = ~self_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & logits_mask
    pos_counts = pos_mask.sum(dim=1)
    has_pos = pos_counts > 0
    if not bool(has_pos.any()):
        return fused_h.new_zeros(())

    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[has_pos] / pos_counts[has_pos].clamp(min=1)
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    return loss.mean()


def retrain_classifier_heads(model, full_ds, train_idx, device, val_loader,
                              mcp_class_weights, base_val_score,
                              epochs=15, lr=5e-4, tag="[Stage 1]"):
    """Decoupled classifier re-balancing (Kang et al., "Decoupling
    Representation and Classifier for Long-Tailed Recognition", ICLR 2020,
    https://arxiv.org/abs/1910.09217). See config.py's
    STAGE1_USE_DECOUPLED_RETRAIN comment for the full rationale.

    Freezes everything except step_head/mcp_head and re-trains only those
    two small heads with class-BALANCED sampling (every Step class equally
    likely, as opposed to train_one_split's instance-balanced-ish sqrt-
    inverse-frequency sampler used for representation learning). The
    backbone forward pass runs under no_grad -- only two Linear stacks get
    gradients, so this phase is cheap even though it uses its own epoch
    budget. Never regresses silently: the pre-retrain state is restored if
    re-balancing does not beat it on val (same pattern as the SWA check
    this function runs right after).
    """
    if epochs <= 0:
        return model, base_val_score

    model.eval()  # freezes BatchNorm running stats + all dropout for the
                  # frozen encoder/fusion trunk; heads get no dropout either
                  # during this short calibration phase, which is fine.

    step_labels_train = np.array([full_ds[i]["step_idx"].item() for i in train_idx])
    class_counts = np.bincount(step_labels_train, minlength=len(STEP_LABELS)).astype(np.float64)
    per_class_w = 1.0 / np.clip(class_counts, 1.0, None)  # true class-balanced weight, no cap/sqrt
    sample_w = per_class_w[step_labels_train]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double), num_samples=len(train_idx), replacement=True
    )
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    loader = DataLoader(train_ds, batch_size=STAGE1_BATCH_SIZE, sampler=sampler, collate_fn=collate)

    head_params = [p for n, p in model.named_parameters()
                   if n.startswith("step_head.") or n.startswith("mcp_head.")]
    opt = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-4)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = base_val_score

    for epoch in range(epochs):
        for graphs, step_idx, mcp_vec, sem_tokens, sem_mask in loader:
            graphs = graphs.to(device)
            step_idx = step_idx.to(device)
            mcp_vec = mcp_vec.to(device)
            sem_tokens = sem_tokens.to(device)
            sem_mask = sem_mask.to(device)
            edge_attr = getattr(graphs, "edge_attr", None)
            with torch.no_grad():
                _, _, (fused_step, fused_mcp) = model(
                    graphs.x, graphs.edge_index, graphs.batch,
                    semantic_tokens=sem_tokens, semantic_mask=sem_mask, edge_attr=edge_attr,
                )
            step_logits, mcp_logits = model.predict_from_fused(fused_step, fused_mcp)
            step_loss = F.cross_entropy(step_logits, step_idx, label_smoothing=0.05)
            bce = F.binary_cross_entropy_with_logits(mcp_logits, mcp_vec, reduction="none")
            bce = bce * mcp_class_weights.view(1, -1)
            mcp_loss = bce.mean()
            loss = step_loss + MCP_LOSS_WEIGHT * mcp_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, STAGE1_GRAD_CLIP)
            opt.step()

        val_metrics = evaluate(model, val_loader, device)
        score = 0.50 * val_metrics["step_accuracy"] + 0.50 * val_metrics["mcp_micro_f1"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if best_score > base_val_score + 1e-5:
        print(f"{tag} decoupled classifier re-balancing improved val score "
              f"{base_val_score:.4f} -> {best_score:.4f} (Kang et al. 2020) -- adopting re-balanced heads.")
    else:
        print(f"{tag} decoupled classifier re-balancing did not beat the representation-phase "
              f"heads ({best_score:.4f} vs {base_val_score:.4f}) -- keeping original heads.")
    return model, max(best_score, base_val_score)


def manifold_mixup_loss(model, fused_h, step_idx, mcp_vec, num_step_classes,
                         fused_mcp=None,
                         alpha=0.2, mcp_class_weights=None):
    """Manifold Mixup auxiliary loss (Verma et al., ICML 2019) computed on
    the fused Stage-1 representation.

    Draws lambda ~ Beta(alpha, alpha), mixes `fused_h` with a random
    in-batch permutation of itself, and scores the two cheap classification
    heads on that mixed vector against the correspondingly soft-mixed
    one-hot Step target and multi-hot MCP target. Does not touch the
    primary `fused_h` used by the batch's main loss -- this is purely an
    additional regularization term (see STAGE1_USE_MANIFOLD_MIXUP /
    STAGE1_MIXUP_WEIGHT in config.py; default off, intended to be A/B
    tested).
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(fused_h.size(0), device=fused_h.device)
    mixed_h = lam * fused_h + (1.0 - lam) * fused_h[perm]
    # ROUND 7: with per-head gates the MCP head reads its own fused vector,
    # so mix that one too (using the SAME lambda and permutation, so the
    # mixed Step/MCP views correspond to the same pair of examples and the
    # soft-mixed targets below stay valid for both heads).
    if fused_mcp is None:
        mixed_mcp_h = mixed_h
    else:
        mixed_mcp_h = lam * fused_mcp + (1.0 - lam) * fused_mcp[perm]

    mix_step_logits, mix_mcp_logits = model.predict_from_fused(mixed_h, mixed_mcp_h)

    step_onehot = F.one_hot(step_idx, num_step_classes).float()
    mixed_step_target = lam * step_onehot + (1.0 - lam) * step_onehot[perm]
    # F.cross_entropy accepts class-probability targets (same shape as
    # input) directly since PyTorch 1.10 -- no manual log-softmax needed.
    step_loss = F.cross_entropy(mix_step_logits, mixed_step_target)

    mixed_mcp_target = lam * mcp_vec + (1.0 - lam) * mcp_vec[perm]
    bce = F.binary_cross_entropy_with_logits(mix_mcp_logits, mixed_mcp_target, reduction="none")
    if mcp_class_weights is not None:
        bce = bce * mcp_class_weights.view(1, -1)
    mcp_loss = bce.mean()

    return step_loss + mcp_loss


def train_one_split(full_ds, train_idx, val_idx, device, ckpt_path, tag="[Stage 1]"):
    """Train a single Stage-1 model on one train/val machine split.

    Factored out of main() as a single, reusable training procedure -- loss,
    sampler, scheduler, hard-negative margin, SWA -- so it can't drift out of
    sync with itself if called from more than one place.

    Returns (model, mcp_w_np, mcp_counts, val_probs, val_gold, val_metrics)
    where val_probs/val_gold are the *sigmoid* MCP probabilities and binary
    targets on this split's val set (used for threshold search / OOF pooling).
    """
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)

    step_counts = np.bincount([full_ds[i]["step_idx"].item() for i in train_idx], minlength=len(STEP_LABELS)).astype(np.float64)
    mcp_counts = np.zeros(len(MCP_LABELS), dtype=np.float64)
    for i in train_idx:
        mcp_counts += full_ds[i]["mcp_vec"].numpy()
    step_w_np = _soft_class_weights(step_counts, STAGE1_MAX_CLASS_WEIGHT, rare_boost=1.20)
    mcp_w_np = _soft_class_weights(mcp_counts, STAGE1_MAX_MCP_WEIGHT, rare_boost=1.15)
    step_weights = torch.tensor(step_w_np, dtype=torch.float32, device=device)
    mcp_weights = torch.tensor(mcp_w_np, dtype=torch.float32, device=device)

    # Logit adjustment (Menon et al., ICLR 2021): log of each Step class's
    # TRAIN-split prior, used to bias training-time logits toward leaving a
    # margin proportional to class rarity (see config.py /
    # STAGE1_IMPROVEMENTS.md). Unlike the capped inverse-frequency weights
    # above, this never saturates for very rare classes. Computed from raw
    # step_counts (not the clipped step_w_np) since it needs the true prior,
    # not the already-bounded sampling weight.
    step_priors = np.clip(step_counts / max(step_counts.sum(), 1.0), 1e-6, 1.0)
    step_log_priors = torch.tensor(np.log(step_priors), dtype=torch.float32, device=device)

    print(f"{tag} STEP weights:")
    for i, lab in enumerate(STEP_LABELS):
        print(f"  [{i}] {lab[:54]:<54}: w={step_w_np[i]:.3f} count={int(step_counts[i])}")
    print(f"{tag} MCP weights:")
    for i, lab in enumerate(MCP_LABELS):
        print(f"  {lab:<22}: w={mcp_w_np[i]:.3f} count={int(mcp_counts[i])}")

    # Gentle class-aware sampling: square-root inverse frequency, clipped.
    sample_weights = []
    for idx in train_idx:
        step_i = full_ds[idx]["step_idx"].item()
        w = max(float(step_w_np[step_i]), 1e-6)
        active = full_ds[idx]["mcp_vec"].numpy() > 0
        if active.any():
            w = max(w, float(np.sqrt(np.max(mcp_w_np[active]))))
        sample_weights.append(np.sqrt(w))
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    sample_weights /= max(np.median(sample_weights), 1e-8)
    sample_weights = np.clip(sample_weights, 0.70, 2.0)
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_idx), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=STAGE1_BATCH_SIZE, sampler=sampler, collate_fn=collate, drop_last=False)

    model = Stage1Classifier().to(device)
    print(f"{tag} Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ROUND 6 (see config.py's STAGE1_GRAPH_GATE_LR_MULT comment): the graph
    # gate is a single scalar with a short gradient path -- give it its own
    # optimizer param group at a much higher LR so it actually reaches
    # equilibrium within the training budget instead of creeping (the first
    # real run moved it only ~7% in 44 epochs). Both groups share the same
    # `lr_lambda` warmup/cosine schedule below (LambdaLR applies one
    # multiplicative schedule to every group's own base LR), so the ~12x
    # ratio between them holds throughout training, not just at epoch 0.
    if getattr(model, "use_graph_gate", False):
        # ROUND 7: all gate scalars (shared + per-head) share the fast LR
        # group -- they are single scalars with a short gradient path and
        # would otherwise creep (see STAGE1_GRAPH_GATE_LR_MULT).
        gate_params = [model.graph_gate_raw,
                       model.graph_gate_step_raw,
                       model.graph_gate_mcp_raw]
        gate_ids = {id(p) for p in gate_params}
        other_params = [p for p in model.parameters() if id(p) not in gate_ids]
        opt = torch.optim.AdamW(
            [
                {"params": other_params, "lr": STAGE1_LR},
                {"params": gate_params, "lr": STAGE1_LR * STAGE1_GRAPH_GATE_LR_MULT},
            ],
            weight_decay=STAGE1_WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * STAGE1_EPOCHS
    warmup_steps = steps_per_epoch * STAGE1_WARMUP_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.10, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    hard_groups = [
        (0, 5),   # research/search <-> exploit
        (2, 1),   # explore <-> service enumeration
        (2, 3),   # explore <-> website enumeration
        (6, 2),   # analyze <-> explore
        (1, 3),   # service <-> website enumeration
        (4, 1),   # domain <-> service enumeration
        # ADDED: explore-suspicious-files <-> exploit. This is the single
        # largest confusion pair in BOTH the predecessor codebase's own
        # audit ("Both Stage 1 and Stage 2 (and 3) consistently confuse
        # 'Explore the suspicious files...' with 'Exploit the selected
        # exploitations'" -- CHANGES_AND_FINDINGS.md) and a fresh test-set
        # error analysis on this codebase (12/268 rows, more than double
        # any other single confusion pair: 8 Explore->Exploit + 4
        # Exploit->Explore). It was flagged twice and never actually added
        # to this list -- adding it now gives the margin loss a direct
        # shot at the #1 error mode instead of only the smaller ones.
        (2, 5),   # explore <-> exploit
        # ADDED from the Round-7 test confusion matrix: class 2
        # ("Explore the suspicious files...") absorbed 26 of the 60 total
        # step errors -- 43% of ALL errors were false-positive class 2, and
        # its precision was only 0.40. The worst single donor was class 8
        # ("Explore the source code for vulnerabilities."), which lost 4 of
        # its 5 test examples to class 2 -- unsurprising given both labels
        # literally start with "Explore the...". That exact pair was never
        # in this list. Class 6 ("Analyze the outcomes...") was likewise
        # predicted 7x for 3 gold examples, 0 correct.
        (2, 8),   # explore-suspicious-files <-> explore-source-code
        (0, 2),   # google search <-> explore-suspicious-files
    ]

    best_score, best_epoch, no_improve = -1.0, -1, 0
    patience = 12
    train_losses, val_scores = [], []
    # Rolling pool of the top-K checkpoints by val score, kept in CPU RAM,
    # used for Stochastic Weight Averaging after training ends (see
    # STAGE1_SWA_TOP_K in config.py for rationale).
    top_k_ckpts = []

    for epoch in range(STAGE1_EPOCHS):
        model.train()
        total_loss = step_run = mcp_run = con_run = hn_run = 0.0
        n_batches = 0
        for graphs, step_idx, mcp_vec, sem_tokens, sem_mask in train_loader:
            graphs = graphs.to(device)
            step_idx = step_idx.to(device)
            mcp_vec = mcp_vec.to(device)
            sem_tokens = sem_tokens.to(device)
            sem_mask = sem_mask.to(device)
            edge_attr = getattr(graphs, "edge_attr", None)
            step_logits, mcp_logits, (fused_step, fused_mcp) = model(
                graphs.x, graphs.edge_index, graphs.batch,
                semantic_tokens=sem_tokens, semantic_mask=sem_mask, edge_attr=edge_attr
            )
            base, step_l, mcp_l = model.loss(
                step_logits, mcp_logits, step_idx, mcp_vec,
                step_w=STEP_LOSS_WEIGHT, mcp_w=MCP_LOSS_WEIGHT,
                mcp_class_weights=mcp_weights, use_focal=True, focal_gamma=1.8,
                label_smoothing=STEP_LABEL_SMOOTHING,
                step_class_weights=(step_weights if STAGE1_USE_STEP_CLASS_WEIGHTS else None),
                use_step_focal=STAGE1_USE_STEP_FOCAL, step_focal_gamma=STAGE1_STEP_FOCAL_GAMMA,
                step_log_priors=(step_log_priors if STAGE1_USE_LOGIT_ADJUSTMENT else None),
                logit_adj_tau=STAGE1_LOGIT_ADJ_TAU,
                use_asl=(STAGE1_MCP_LOSS_TYPE == "asl"),
                asl_gamma_neg=STAGE1_ASL_GAMMA_NEG, asl_gamma_pos=STAGE1_ASL_GAMMA_POS,
                asl_clip=STAGE1_ASL_CLIP,
            )
            hn = hard_negative_margin(step_logits, step_idx, hard_groups, margin=STAGE1_HARD_NEGATIVE_MARGIN)
            if STAGE1_SUPCON_WEIGHT > 0:
                # SupCon's positives are defined by the STEP label, so it
                # belongs on the Step-side fused representation (ROUND 7).
                con = supervised_contrastive_loss(fused_step, step_idx, temperature=STAGE1_SUPCON_TEMPERATURE)
            else:
                con = fused_step.new_zeros(())
            loss = base + STAGE1_HARD_NEGATIVE_WEIGHT * hn + STAGE1_SUPCON_WEIGHT * con

            # Optional Manifold Mixup auxiliary term (default OFF -- see
            # config.py STAGE1_USE_MANIFOLD_MIXUP). Purely additive: reuses
            # the fused_h this batch already computed, only re-runs the two
            # cheap linear heads on a convex combination of two examples'
            # fused vectors + soft-mixed targets, so it cannot change the
            # primary forward pass even when enabled.
            if STAGE1_USE_MANIFOLD_MIXUP and fused_step.size(0) > 1:
                mix_loss = manifold_mixup_loss(
                    model, fused_step, step_idx, mcp_vec,
                    num_step_classes=len(STEP_LABELS),
                    fused_mcp=fused_mcp,
                    alpha=STAGE1_MIXUP_ALPHA, mcp_class_weights=mcp_weights,
                )
                loss = loss + STAGE1_MIXUP_WEIGHT * mix_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), STAGE1_GRAD_CLIP)
            opt.step(); sched.step()
            total_loss += float(loss.item()); step_run += float(step_l.item()); mcp_run += float(mcp_l.item())
            hn_run += float(hn.item()); con_run += float(con.item()); n_batches += 1

        val_metrics = evaluate(model, val_loader, device)
        # Stage-1 checkpoint selection gives equal priority to the two actual
        # supervised objectives: Step and MCP. Macro-F1 remains diagnostic.
        score = (0.50 * val_metrics["step_accuracy"]
                 + 0.50 * val_metrics["mcp_micro_f1"])
        train_losses.append(total_loss / max(1, n_batches)); val_scores.append(score)
        lr = sched.get_last_lr()[0]
        gate_str = ""
        if getattr(model, "use_graph_gate", False):
            if getattr(model, "use_per_head_gate", False):
                gate_str = (
                    f" | gate_step {torch.sigmoid(model.graph_gate_step_raw).item():.3f}"
                    f" gate_mcp {torch.sigmoid(model.graph_gate_mcp_raw).item():.3f}"
                )
            else:
                gate_str = f" | graph_gate {torch.sigmoid(model.graph_gate_raw).item():.3f}"
        print(
            f"{tag} epoch {epoch+1:02d}/{STAGE1_EPOCHS} | lr {lr:.2e} | "
            f"train {total_loss/max(1,n_batches):.4f} (step={step_run/max(1,n_batches):.4f}, "
            f"mcp={mcp_run/max(1,n_batches):.4f}, hn={hn_run/max(1,n_batches):.4f}, con={con_run/max(1,n_batches):.4f}) | "
            f"val_step_acc {val_metrics['step_accuracy']:.3f} "
            f"val_step_macroF1 {val_metrics['step_macro_f1']:.3f} | "
            f"val_mcp_microF1 {val_metrics['mcp_micro_f1']:.3f} "
            f"val_mcp_subsetAcc {val_metrics['mcp_subset_accuracy']:.3f} | score {score:.4f}{gate_str}"
        )
        # Maintain the top-K pool for SWA regardless of whether this epoch
        # was the single best -- SWA wants a handful of *good* checkpoints,
        # not just the current best.
        top_k_ckpts.append((score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        top_k_ckpts.sort(key=lambda t: t[0], reverse=True)
        del top_k_ckpts[STAGE1_SWA_TOP_K:]

        if score > best_score + 1e-5:
            best_score, best_epoch, no_improve = score, epoch + 1, 0
            torch.save({
                "model_state_dict": model.state_dict(), "best_epoch": best_epoch,
                "best_score": best_score, "train_losses": train_losses, "val_scores": val_scores,
                "architecture": "paper_semantic_cnn_plus_typed_gine_fusion_v2",
            }, ckpt_path)
            print(f"{tag}   -> saved best checkpoint to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"{tag} Early stopping at epoch {epoch+1}; best epoch {best_epoch}.")
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # --- Stochastic Weight Averaging over the top-K checkpoints ------------
    # LayerNorm/GraphNorm carry no running batch statistics, so averaging
    # weights directly (no extra forward pass needed to "recalibrate") is
    # safe here. Only adopt SWA if it actually beats the single best
    # checkpoint on the same val split -- never regress silently.
    if len(top_k_ckpts) >= 2:
        swa_state = {}
        for key in top_k_ckpts[0][1]:
            stacked = torch.stack([sd[key].float() for _, sd in top_k_ckpts], dim=0)
            swa_state[key] = stacked.mean(dim=0).to(top_k_ckpts[0][1][key].dtype)
        swa_model = Stage1Classifier().to(device)
        swa_model.load_state_dict(swa_state)
        swa_val_metrics = evaluate(swa_model, val_loader, device)
        swa_score = 0.50 * swa_val_metrics["step_accuracy"] + 0.50 * swa_val_metrics["mcp_micro_f1"]
        print(f"{tag} SWA over top-{len(top_k_ckpts)} checkpoints: val score {swa_score:.4f} "
              f"(single-best checkpoint was {best_score:.4f})")
        if swa_score >= best_score:
            print(f"{tag} SWA improves (or matches) the single-best checkpoint -- adopting SWA weights.")
            model = swa_model
            best_score = swa_score
            ckpt = {
                "model_state_dict": model.state_dict(), "best_epoch": best_epoch,
                "best_score": best_score, "train_losses": train_losses, "val_scores": val_scores,
                "architecture": "paper_semantic_cnn_plus_typed_gine_fusion_v2_swa",
                "swa_k": len(top_k_ckpts),
            }
            torch.save(ckpt, ckpt_path)
        else:
            print(f"{tag} SWA did not beat the single-best checkpoint -- keeping single-best weights.")

    # --- Decoupled classifier re-balancing (Kang et al., ICLR 2020) --------
    # Runs after representation training (+ optional SWA) has settled, on
    # whichever weights are currently in `model`. See config.py's
    # STAGE1_USE_DECOUPLED_RETRAIN comment.
    if STAGE1_USE_DECOUPLED_RETRAIN and STAGE1_DECOUPLED_EPOCHS > 0:
        pre_retrain_metrics = evaluate(model, val_loader, device)
        pre_retrain_score = 0.50 * pre_retrain_metrics["step_accuracy"] + 0.50 * pre_retrain_metrics["mcp_micro_f1"]
        model, best_score = retrain_classifier_heads(
            model, full_ds, train_idx, device, val_loader,
            mcp_class_weights=mcp_weights, base_val_score=pre_retrain_score,
            epochs=STAGE1_DECOUPLED_EPOCHS, lr=STAGE1_DECOUPLED_LR, tag=tag,
        )
        torch.save({
            "model_state_dict": model.state_dict(), "best_epoch": best_epoch,
            "best_score": best_score, "train_losses": train_losses, "val_scores": val_scores,
            "architecture": "paper_semantic_cnn_plus_typed_gine_fusion_v2_decoupled",
        }, ckpt_path)

    val_metrics, val_probs, val_gold, val_step_logits, val_step_gold = evaluate(
        model, val_loader, device, return_probs=True, return_step_logits=True
    )
    rare = [i for i, c in enumerate(mcp_counts) if c < 15]
    thresholds = search_per_class_thresholds(val_probs, val_gold, rare_class_indices=rare,
                                              verbose=(tag == "[Stage 1]"))

    # STEP per-class logit-bias calibration -- the multi-class analogue of the
    # MCP threshold search above, which the Step head never had. Fit on the
    # same held-out val split, with the same support gate / bootstrap
    # stabilization / never-regress guard. See search_step_logit_bias().
    step_bias = search_step_logit_bias(
        val_step_logits, val_step_gold, verbose=(tag == "[Stage 1]")
    )
    if any(abs(b) > 1e-9 for b in step_bias):
        calibrated = evaluate(model, val_loader, device, step_bias=step_bias)
        print(f"{tag} step calibration on val: accuracy "
              f"{val_metrics['step_accuracy']:.4f} -> {calibrated['step_accuracy']:.4f}, "
              f"macroF1 {val_metrics['step_macro_f1']:.4f} -> {calibrated['step_macro_f1']:.4f}")

    ckpt["mcp_thresholds"] = [float(x) for x in thresholds]
    ckpt["step_logit_bias"] = [float(x) for x in step_bias]
    ckpt["mcp_class_weights"] = [float(x) for x in mcp_w_np]
    ckpt["step_class_weights"] = [float(x) for x in step_w_np]
    ckpt["val_metrics"] = val_metrics
    torch.save(ckpt, ckpt_path)
    print(f"{tag} val_step_acc={val_metrics['step_accuracy']:.4f}  val_mcp_microF1={val_metrics['mcp_micro_f1']:.4f}  "
          f"thresholds={[round(float(x), 2) for x in thresholds]}")

    return model, mcp_w_np, mcp_counts, val_probs, val_gold, val_metrics, thresholds, step_bias


def _machine_folds(examples, n_folds, seed):
    """Partition MACHINES (never rows) into n_folds groups.

    Machine-grouped so no machine's rows ever straddle a fold boundary -- the
    same contract the train/test split already enforces, and the reason the
    val/test comparison is honest. Machines are shuffled, then dealt
    round-robin into folds ordered by current row count, which keeps folds
    close in size even though machines contribute very different row counts.
    """
    by_machine = {}
    for i, e in enumerate(examples):
        by_machine.setdefault(e["machine"], []).append(i)
    machines = sorted(by_machine)
    rng = np.random.default_rng(seed)
    rng.shuffle(machines)
    folds = [[] for _ in range(n_folds)]
    for m in machines:
        smallest = min(range(n_folds), key=lambda k: len(folds[k]))
        folds[smallest].extend(by_machine[m])
    return [sorted(f) for f in folds]


def _print_test_metrics(test_metrics, header):
    print(f"\n[Stage 1] ===== {header} =====")
    for k in ("step_accuracy", "step_micro_f1", "step_macro_f1", "step_weighted_f1",
              "mcp_micro_f1", "mcp_macro_f1", "mcp_subset_accuracy", "mcp_samples_f1"):
        print(f"  {k:<20}: {test_metrics[k]:.4f}")
    combined = 0.5 * test_metrics["step_accuracy"] + 0.5 * test_metrics["mcp_micro_f1"]
    print(f"  {'combined_score':<20}: {combined:.4f}")
    return combined


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 1] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 1] Test input     : {INPUT_TEST_JSON}")
    print(f"[Stage 1] Device         : {device}")
    print(f"[Stage 1] Architecture   : paper-inspired semantic CNN + typed GINE graph fusion")
    print(f"[Stage 1] Epochs         : {STAGE1_EPOCHS} (warmup {STAGE1_WARMUP_EPOCHS})")

    full_ds = Stage1Dataset(INPUT_TRAIN_JSON, split="train")
    examples = full_ds.examples
    all_machines = set(e["machine"] for e in examples)

    test_pre = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = set(e["machine"] for e in test_pre)
    assert not (all_machines & test_machines), "TRAIN/TEST machine overlap detected"
    print("[Stage 1] ✓ No machine overlap between train and test sets")

    test_ds = Stage1Dataset(INPUT_TEST_JSON, split="test")
    test_loader = DataLoader(test_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)
    csv_path = os.path.join(ROOT, "output", "stage1.csv")

    if not STAGE1_USE_KFOLD:
        _main_single_split(full_ds, examples, all_machines, device, test_ds, test_loader, csv_path)
        return

    # ── K-fold, machine-grouped ───────────────────────────────────────────────
    folds = _machine_folds(examples, STAGE1_N_FOLDS, RANDOM_SEED + 1)
    print(f"[Stage 1] K-fold ensembling: {STAGE1_N_FOLDS} machine-grouped folds "
          f"over {len(all_machines)} machines / {len(examples)} rows")
    for k, f in enumerate(folds):
        n_mach = len(set(examples[i]["machine"] for i in f))
        print(f"[Stage 1]   fold {k}: {len(f):>5} rows, {n_mach:>3} machines")

    models, fold_scores = [], []
    oof_step_logits, oof_step_gold = [], []
    oof_mcp_probs, oof_mcp_gold = [], []
    mcp_counts_ref = None

    for k in range(STAGE1_N_FOLDS):
        val_idx = folds[k]
        train_idx = [i for j, f in enumerate(folds) if j != k for i in f]
        # Every fold's val machines are disjoint from its own train machines by
        # construction; assert it rather than trust it.
        tm = set(examples[i]["machine"] for i in train_idx)
        vm = set(examples[i]["machine"] for i in val_idx)
        assert not (tm & vm), f"fold {k}: TRAIN/VAL machine overlap"

        fold_ckpt = os.path.join(ROOT, "checkpoints", f"stage1_fold{k}.pt")
        tag = f"[Stage 1][fold {k}]"
        print(f"\n{tag} train {len(train_idx)} rows / {len(tm)} machines  |  "
              f"val {len(val_idx)} rows / {len(vm)} machines")
        (model, mcp_w_np, mcp_counts, val_probs, val_gold,
         val_metrics, _thr, _bias) = train_one_split(
            full_ds, train_idx, val_idx, device, fold_ckpt, tag=tag
        )
        models.append(model)
        mcp_counts_ref = mcp_counts if mcp_counts_ref is None else mcp_counts_ref
        fold_scores.append(0.5 * val_metrics["step_accuracy"] + 0.5 * val_metrics["mcp_micro_f1"])

        # Out-of-fold predictions: this model never saw these machines.
        val_loader = DataLoader(torch.utils.data.Subset(full_ds, val_idx),
                                batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)
        _m, p_probs, p_gold, s_logits, s_gold = evaluate(
            model, val_loader, device, return_probs=True, return_step_logits=True
        )
        oof_step_logits.append(s_logits); oof_step_gold.append(s_gold)
        oof_mcp_probs.append(p_probs);    oof_mcp_gold.append(p_gold)

    oof_step_logits = np.concatenate(oof_step_logits, axis=0)
    oof_step_gold   = np.concatenate(oof_step_gold, axis=0)
    oof_mcp_probs   = np.concatenate(oof_mcp_probs, axis=0)
    oof_mcp_gold    = np.concatenate(oof_mcp_gold, axis=0)

    print(f"\n[Stage 1] Pooled OUT-OF-FOLD calibration set: {len(oof_step_gold)} rows "
          f"covering all {len(all_machines)} training machines "
          f"(vs {int(len(examples) * (STAGE2_VAL_SPLIT or 0.15))} rows from a single split)")
    print(f"[Stage 1] Per-fold val scores: "
          + ", ".join(f"fold{k}={v:.4f}" for k, v in enumerate(fold_scores))
          + f"  (mean {np.mean(fold_scores):.4f}, std {np.std(fold_scores):.4f})")

    # Calibrate on POOLED OOF rather than one 15% split -- the whole point of
    # the K-fold change. CAVEAT: OOF logits come from a single fold model each,
    # while test-time logits are an average of K models, so the ensemble's
    # logit spread is slightly narrower and a bias fit here is marginally
    # aggressive. It is applied anyway because an additive bias commutes with
    # the mean (see evaluate()'s docstring), and because both the bias search
    # and the threshold search carry never-regress guards fit on this same OOF
    # pool. The ensemble-vs-single comparison printed below is the check.
    rare = [i for i, c in enumerate(mcp_counts_ref) if c < 15]
    thresholds = search_per_class_thresholds(oof_mcp_probs, oof_mcp_gold,
                                             rare_class_indices=rare, verbose=True)
    step_bias = search_step_logit_bias(oof_step_logits, oof_step_gold, verbose=True)

    # ── Test: single best fold vs the ensemble ────────────────────────────────
    best_k = int(np.argmax(fold_scores))
    print(f"\n[Stage 1] Best single fold by val score: fold {best_k} ({fold_scores[best_k]:.4f})")
    single_metrics = evaluate(models[best_k], test_loader, device,
                              threshold=thresholds, step_bias=step_bias)
    _print_test_metrics(single_metrics, f"TEST — SINGLE BEST FOLD ({best_k})")

    test_metrics = evaluate(models, test_loader, device, threshold=thresholds,
                            save_csv=True, csv_path=csv_path, dataset=test_ds.examples,
                            step_bias=step_bias)
    _print_test_metrics(test_metrics, f"TEST — {STAGE1_N_FOLDS}-FOLD ENSEMBLE")
    print(f"\n[Stage 1] Ensemble vs single-best  step_accuracy: "
          f"{single_metrics['step_accuracy']:.4f} -> {test_metrics['step_accuracy']:.4f} "
          f"({test_metrics['step_accuracy'] - single_metrics['step_accuracy']:+.4f})")
    print(f"[Stage 1] Ensemble vs single-best  mcp_micro_f1 : "
          f"{single_metrics['mcp_micro_f1']:.4f} -> {test_metrics['mcp_micro_f1']:.4f} "
          f"({test_metrics['mcp_micro_f1'] - single_metrics['mcp_micro_f1']:+.4f})")

    # ── Persist ───────────────────────────────────────────────────────────────
    # STAGE1_CKPT must stay a SINGLE-model checkpoint: stage2_sft_qwen.py,
    # stage3_grpo_rl.py and eval/evaluate.py all load it as one
    # `model_state_dict`. Write the best fold's weights there, carrying the
    # pooled-OOF thresholds/bias, so those stages are unchanged by K-fold.
    best_ckpt = torch.load(os.path.join(ROOT, "checkpoints", f"stage1_fold{best_k}.pt"),
                           map_location=device, weights_only=False)
    best_ckpt["mcp_thresholds"]   = [float(x) for x in thresholds]
    best_ckpt["step_logit_bias"]  = [float(x) for x in step_bias]
    best_ckpt["kfold_n"]          = STAGE1_N_FOLDS
    best_ckpt["kfold_best"]       = best_k
    best_ckpt["kfold_members"]    = [os.path.join(ROOT, "checkpoints", f"stage1_fold{k}.pt")
                                     for k in range(STAGE1_N_FOLDS)]
    best_ckpt["kfold_val_scores"] = [float(v) for v in fold_scores]
    best_ckpt["test_metrics_single"]   = {k: float(v) for k, v in single_metrics.items()}
    best_ckpt["test_metrics_ensemble"] = {k: float(v) for k, v in test_metrics.items()}
    torch.save(best_ckpt, STAGE1_CKPT)
    print(f"\n[Stage 1] Saved single-model checkpoint (fold {best_k} + pooled-OOF calibration) "
          f"to {STAGE1_CKPT}")
    print(f"[Stage 1] Ensemble members: {STAGE1_N_FOLDS} files at checkpoints/stage1_fold*.pt")


def _main_single_split(full_ds, examples, all_machines, device, test_ds, test_loader, csv_path):
    """Original single 15%-machine-split path (STAGE1_USE_KFOLD=False)."""
    all_machines = sorted(all_machines)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    perm = rng.permutation(len(all_machines))
    n_val = max(1, int(len(all_machines) * (STAGE2_VAL_SPLIT if STAGE2_VAL_SPLIT else 0.15)))
    val_machines = set(all_machines[i] for i in perm[:n_val])
    train_machines = set(all_machines) - val_machines
    train_idx = [i for i, e in enumerate(examples) if e["machine"] in train_machines]
    val_idx = [i for i, e in enumerate(examples) if e["machine"] in val_machines]
    print(f"[Stage 1] Train machines  : {len(train_machines)}")
    print(f"[Stage 1] Val machines    : {len(val_machines)}")
    print(f"[Stage 1] Train examples  : {len(train_idx)}")
    print(f"[Stage 1] Val examples    : {len(val_idx)}")

    model, _mcp_w, _mcp_c, _vp, _vg, _vm, thresholds, step_bias = train_one_split(
        full_ds, train_idx, val_idx, device, STAGE1_CKPT, tag="[Stage 1]"
    )
    val_selected = os.path.join(ROOT, "checkpoints", "stage1_gnn_classifier_val_selected.pt")
    torch.save(torch.load(STAGE1_CKPT, map_location=device, weights_only=False), val_selected)
    print(f"[Stage 1] Saved validation-selected checkpoint to {val_selected}")

    test_metrics = evaluate(model, test_loader, device, threshold=thresholds, save_csv=True,
                            csv_path=csv_path, dataset=test_ds.examples, step_bias=step_bias)
    _print_test_metrics(test_metrics, "TEST SET RESULTS")


if __name__ == "__main__":
    main()
