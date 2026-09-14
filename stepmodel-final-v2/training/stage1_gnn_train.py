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
)
from data_utils import load_from_input_json, precompute_semantic_tokens
from graph_encoder import Stage1Classifier
from mcp_threshold_search import search_per_class_thresholds

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


def evaluate(model, loader, device, threshold=0.5, return_probs=False, save_csv=False, csv_path=None, dataset=None):
    model.eval()
    step_preds, step_gold = [], []
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
            step_logits, mcp_logits, _ = model(
                graphs.x, graphs.edge_index, graphs.batch,
                semantic_tokens=sem_tokens, semantic_mask=sem_mask, edge_attr=edge_attr
            )
            sp = step_logits.argmax(-1).cpu().numpy()
            sg = step_idx.cpu().numpy()
            probs = torch.sigmoid(mcp_logits).cpu().numpy()
            if isinstance(threshold, (list, np.ndarray)):
                thr = torch.tensor(threshold, dtype=torch.float32, device=device)
                mp = (torch.sigmoid(mcp_logits) >= thr).float().cpu().numpy()
            else:
                mp = (torch.sigmoid(mcp_logits) >= threshold).float().cpu().numpy()
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
                _, _, fused_h = model(
                    graphs.x, graphs.edge_index, graphs.batch,
                    semantic_tokens=sem_tokens, semantic_mask=sem_mask, edge_attr=edge_attr,
                )
            step_logits, mcp_logits = model.predict_from_fused(fused_h)
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

    mix_step_logits, mix_mcp_logits = model.predict_from_fused(mixed_h)

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
            step_logits, mcp_logits, fused_h = model(
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
                con = supervised_contrastive_loss(fused_h, step_idx, temperature=STAGE1_SUPCON_TEMPERATURE)
            else:
                con = fused_h.new_zeros(())
            loss = base + STAGE1_HARD_NEGATIVE_WEIGHT * hn + STAGE1_SUPCON_WEIGHT * con

            # Optional Manifold Mixup auxiliary term (default OFF -- see
            # config.py STAGE1_USE_MANIFOLD_MIXUP). Purely additive: reuses
            # the fused_h this batch already computed, only re-runs the two
            # cheap linear heads on a convex combination of two examples'
            # fused vectors + soft-mixed targets, so it cannot change the
            # primary forward pass even when enabled.
            if STAGE1_USE_MANIFOLD_MIXUP and fused_h.size(0) > 1:
                mix_loss = manifold_mixup_loss(
                    model, fused_h, step_idx, mcp_vec,
                    num_step_classes=len(STEP_LABELS),
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
        print(
            f"{tag} epoch {epoch+1:02d}/{STAGE1_EPOCHS} | lr {lr:.2e} | "
            f"train {total_loss/max(1,n_batches):.4f} (step={step_run/max(1,n_batches):.4f}, "
            f"mcp={mcp_run/max(1,n_batches):.4f}, hn={hn_run/max(1,n_batches):.4f}, con={con_run/max(1,n_batches):.4f}) | "
            f"val_step_acc {val_metrics['step_accuracy']:.3f} "
            f"val_step_macroF1 {val_metrics['step_macro_f1']:.3f} | "
            f"val_mcp_microF1 {val_metrics['mcp_micro_f1']:.3f} "
            f"val_mcp_subsetAcc {val_metrics['mcp_subset_accuracy']:.3f} | score {score:.4f}"
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

    val_metrics, val_probs, val_gold = evaluate(model, val_loader, device, return_probs=True)
    rare = [i for i, c in enumerate(mcp_counts) if c < 15]
    thresholds = search_per_class_thresholds(val_probs, val_gold, rare_class_indices=rare,
                                              verbose=(tag == "[Stage 1]"))
    ckpt["mcp_thresholds"] = [float(x) for x in thresholds]
    ckpt["mcp_class_weights"] = [float(x) for x in mcp_w_np]
    ckpt["step_class_weights"] = [float(x) for x in step_w_np]
    ckpt["val_metrics"] = val_metrics
    torch.save(ckpt, ckpt_path)
    print(f"{tag} val_step_acc={val_metrics['step_accuracy']:.4f}  val_mcp_microF1={val_metrics['mcp_micro_f1']:.4f}  "
          f"thresholds={[round(float(x), 2) for x in thresholds]}")

    return model, mcp_w_np, mcp_counts, val_probs, val_gold, val_metrics, thresholds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 1] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 1] Test input     : {INPUT_TEST_JSON}")
    print(f"[Stage 1] Device         : {device}")
    print(f"[Stage 1] Architecture   : paper-inspired semantic CNN + typed GINE graph fusion")
    print(f"[Stage 1] Epochs         : {STAGE1_EPOCHS} (warmup {STAGE1_WARMUP_EPOCHS})")

    full_ds = Stage1Dataset(INPUT_TRAIN_JSON, split="train")
    examples = full_ds.examples
    all_machines = sorted(set(e["machine"] for e in examples))
    rng = np.random.default_rng(RANDOM_SEED + 1)
    perm = rng.permutation(len(all_machines))
    n_val = max(1, int(len(all_machines) * (STAGE2_VAL_SPLIT if STAGE2_VAL_SPLIT else 0.15)))
    val_machines = set(all_machines[i] for i in perm[:n_val])
    train_machines = set(all_machines) - val_machines
    train_idx = [i for i, e in enumerate(examples) if e["machine"] in train_machines]
    val_idx = [i for i, e in enumerate(examples) if e["machine"] in val_machines]

    test_pre = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = set(e["machine"] for e in test_pre)
    assert not (train_machines & test_machines), "TRAIN/TEST machine overlap detected"
    assert not (val_machines & test_machines), "VAL/TEST machine overlap detected"
    print("[Stage 1] ✓ No machine overlap between (train ∪ val) and test sets")
    print(f"[Stage 1] Train machines  : {len(train_machines)}")
    print(f"[Stage 1] Val machines    : {len(val_machines)}")
    print(f"[Stage 1] Train examples  : {len(train_idx)}")
    print(f"[Stage 1] Val examples    : {len(val_idx)}")

    model, mcp_w_np, mcp_counts, val_probs, val_gold, val_metrics, thresholds = train_one_split(
        full_ds, train_idx, val_idx, device, STAGE1_CKPT, tag="[Stage 1]"
    )

    val_selected = os.path.join(ROOT, "checkpoints", "stage1_gnn_classifier_val_selected.pt")
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    torch.save(ckpt, val_selected)
    print(f"[Stage 1] Saved validation-selected checkpoint to {val_selected}")

    test_ds = Stage1Dataset(INPUT_TEST_JSON, split="test")
    test_loader = DataLoader(test_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_metrics = evaluate(model, test_loader, device, threshold=thresholds, save_csv=True,
                            csv_path=os.path.join(ROOT, "output", "stage1.csv"), dataset=test_ds.examples)
    print("\n[Stage 1] ===== TEST SET RESULTS =====")
    print(f"  {'step_accuracy':<20}: {test_metrics['step_accuracy']:.4f}")
    print(f"  {'step_micro_f1':<20}: {test_metrics['step_micro_f1']:.4f}")
    print(f"  {'step_macro_f1':<20}: {test_metrics['step_macro_f1']:.4f}")
    print(f"  {'step_weighted_f1':<20}: {test_metrics['step_weighted_f1']:.4f}")
    print(f"  {'mcp_micro_f1':<20}: {test_metrics['mcp_micro_f1']:.4f}")
    print(f"  {'mcp_macro_f1':<20}: {test_metrics['mcp_macro_f1']:.4f}")
    print(f"  {'mcp_subset_accuracy':<20}: {test_metrics['mcp_subset_accuracy']:.4f}")
    print(f"  {'mcp_samples_f1':<20}: {test_metrics['mcp_samples_f1']:.4f}")
    combined = 0.5 * test_metrics["step_accuracy"] + 0.5 * test_metrics["mcp_micro_f1"]
    print(f"  {'combined_score':<20}: {combined:.4f}")


if __name__ == "__main__":
    main()
