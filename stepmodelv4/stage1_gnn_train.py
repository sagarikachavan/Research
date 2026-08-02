"""
Stage 1: Supervised training of the GNN encoder for stepmodelv4
Trains graph encoder to produce embeddings for graph prefix adapter

Training input: stepmodelv4/input/train.json
Evaluation input: stepmodelv4/input/test.json

Run:
    python stage1_gnn_train.py
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, f1_score

from config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, GNN_CKPT, GNN_LR, GNN_EPOCHS,
    GNN_BATCH_SIZE, RANDOM_SEED, STEP_LABELS, MCP_LABELS,
)
from data_utils import load_json_data
from graph_encoder import Stage1Classifier

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class Stage1Dataset(Dataset):
    def __init__(self, json_path):
        self.examples = load_json_data(json_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "graph": ex["graph"],
            "field_embs": torch.tensor(ex["field_embs"], dtype=torch.float32),
            "step_idx": torch.tensor(ex["gold_step_idx"], dtype=torch.long),
            "mcp_vec": torch.tensor(ex["gold_mcp_vec"], dtype=torch.float32),
        }


def collate(batch):
    graphs = Batch.from_data_list([b["graph"] for b in batch])
    field_embs = torch.stack([b["field_embs"] for b in batch])
    step_idx = torch.stack([b["step_idx"] for b in batch])
    mcp_vec = torch.stack([b["mcp_vec"] for b in batch])
    return graphs, field_embs, step_idx, mcp_vec


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    step_preds, step_gold = [], []
    mcp_preds, mcp_gold = [], []
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
            mcp_preds.append((torch.sigmoid(mcp_logits) >= threshold).float().cpu().numpy())
            mcp_gold.append(mcp_vec.cpu().numpy())

    step_preds = np.concatenate(step_preds)
    step_gold = np.concatenate(step_gold)
    mcp_preds = np.concatenate(mcp_preds)
    mcp_gold = np.concatenate(mcp_gold)

    metrics = {
        "step_accuracy": accuracy_score(step_gold, step_preds),
        "step_macro_f1": f1_score(step_gold, step_preds, average="macro", zero_division=0),
        "step_weighted_f1": f1_score(step_gold, step_preds, average="weighted", zero_division=0),
        "mcp_subset_accuracy": accuracy_score(mcp_gold, mcp_preds),
        "mcp_micro_f1": f1_score(mcp_gold, mcp_preds, average="micro", zero_division=0),
        "mcp_macro_f1": f1_score(mcp_gold, mcp_preds, average="macro", zero_division=0),
        "mcp_samples_f1": f1_score(mcp_gold, mcp_preds, average="samples", zero_division=0),
    }
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 1] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 1] Test input     : {INPUT_TEST_JSON}")
    print(f"[Stage 1] Device        : {device}")

    full_ds = Stage1Dataset(INPUT_TRAIN_JSON)
    n = len(full_ds)
    val_n = max(1, int(0.1 * n))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=GNN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=GNN_BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = Stage1Classifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=GNN_EPOCHS)

    best_f1 = -1
    for epoch in range(GNN_EPOCHS):
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
                step_w=1.0, mcp_w=1.0,
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
            torch.save(model.state_dict(), GNN_CKPT)
            print(f"  -> saved new best checkpoint to {GNN_CKPT}")

    print("Stage 1 training complete. Best combined score:", best_f1)


if __name__ == "__main__":
    main()
