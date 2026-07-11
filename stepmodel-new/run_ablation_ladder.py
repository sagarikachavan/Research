#!/usr/bin/env python3
"""Run the StepModel v2 ablation ladder across seeds and summarize metrics."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from statistics import mean, stdev

import torch


CONFIGS = [
    {
        "name": "baseline_static_lora_reference",
        "description": "Reference slot for the previous static-graph LoRA architecture.",
        "external_command_arg": "baseline_command",
        "external_metrics_arg": "baseline_metrics",
    },
    {
        "name": "dynamic_graph_only",
        "description": "Dynamic per-step graph, legacy LoRA path disabled in v2 codebase.",
        "phase0": False,
        "phase2": True,
        "freeze_llm": True,
    },
    {
        "name": "frozen_llm_no_phase0",
        "description": "Dynamic graph, frozen Qwen, random GNN, supervised only.",
        "phase0": False,
        "phase2": False,
        "freeze_llm": True,
    },
    {
        "name": "phase0_pretraining",
        "description": "Dynamic graph, frozen Qwen, Phase 0 init, supervised only.",
        "phase0": True,
        "phase2": False,
        "freeze_llm": True,
    },
    {
        "name": "phase2_grpo_fixed",
        "description": "Phase 0 init plus corrected GRPO.",
        "phase0": True,
        "phase2": True,
        "freeze_llm": True,
    },
    {
        "name": "paper_cnn_reference",
        "description": "Reserved slot for the paper architecture retraining baseline.",
        "external_command_arg": "paper_command",
        "external_metrics_arg": "paper_metrics",
    },
]

METRIC_KEYS = ["step_acc", "step_micro_f1", "mcp_acc", "mcp_micro_f1", "mcp_threshold"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run(cmd, cwd):
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print("+ " + printable)
    subprocess.run(cmd, cwd=cwd, check=True, shell=isinstance(cmd, str))


def prepare_config(base_config, output_root, name, seed, overrides):
    config = deepcopy(base_config)
    config["paths"]["output_dir"] = os.path.join(output_root, name, f"seed_{seed}", "checkpoints")
    config["paths"]["log_dir"] = os.path.join(output_root, name, f"seed_{seed}", "logs")
    config["training"]["seed"] = seed
    config["training"]["num_grpo_epochs"] = (
        int(base_config["training"].get("num_grpo_epochs", 0)) if overrides.get("phase2") else 0
    )
    config["training"]["phase0_checkpoint"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        config["paths"]["output_dir"],
        "phase0_gnn_projector.pt",
    )
    return config


def checkpoint_metrics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "step_acc": float(checkpoint.get("test_step_acc", checkpoint.get("val_step_acc", 0.0))),
        "step_micro_f1": float(checkpoint.get("test_step_micro_f1", checkpoint.get("val_step_acc", 0.0))),
        "mcp_acc": float(checkpoint.get("test_mcp_acc", 0.0)),
        "mcp_micro_f1": float(checkpoint.get("test_mcp_micro_f1", checkpoint.get("test_mcp_f1", checkpoint.get("val_mcp_f1", 0.0)))),
        "mcp_threshold": float(checkpoint.get("mcp_threshold", 0.5)),
    }


def metrics_json(path):
    payload = load_json(path)
    return {key: float(payload.get(key, 0.0)) for key in METRIC_KEYS}


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["config"], []).append(row)
    summary = []
    for name, items in grouped.items():
        out = {"config": name, "seeds": len(items)}
        for key in METRIC_KEYS:
            values = [float(item[key]) for item in items]
            out[f"{key}_mean"] = mean(values)
            out[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-root", default="ablation_runs")
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--baseline-command", default=None)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--paper-command", default=None)
    parser.add_argument("--paper-metrics", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_config = load_json(os.path.join(base_dir, args.config))
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    output_root = os.path.join(base_dir, args.output_root)
    os.makedirs(output_root, exist_ok=True)

    rows = []
    for cfg in CONFIGS:
        if cfg.get("external_command_arg"):
            command = getattr(args, cfg["external_command_arg"])
            metrics_path = getattr(args, cfg["external_metrics_arg"])
            if args.dry_run:
                print(f"Would run external reference `{cfg['name']}` via {cfg['external_command_arg']}.")
                continue
            if not command or not metrics_path:
                raise SystemExit(
                    f"`{cfg['name']}` is required by the ladder. Provide "
                    f"--{cfg['external_command_arg'].replace('_', '-')} and "
                    f"--{cfg['external_metrics_arg'].replace('_', '-')} as a JSON file containing {METRIC_KEYS}."
                )
            run(command, base_dir)
            metrics = metrics_json(metrics_path)
            rows.append({"config": cfg["name"], "seed": -1, **metrics})
            continue
        for seed in seeds:
            run_dir = os.path.join(output_root, cfg["name"], f"seed_{seed}")
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
            os.makedirs(run_dir, exist_ok=True)
            config = prepare_config(base_config, args.output_root, cfg["name"], seed, cfg)
            config_path = os.path.join(run_dir, "config.json")
            dump_json(config_path, config)

            if args.dry_run:
                print(f"Would run {cfg['name']} seed {seed} with {config_path}")
                continue

            if cfg.get("phase0"):
                run([
                    sys.executable, "pretrain_gnn.py",
                    "--config", config_path,
                    "--epochs", str(args.pretrain_epochs),
                    "--seed", str(seed),
                ], base_dir)

            run([sys.executable, "train_gnn_rl.py", "--config", config_path], base_dir)
            final_ckpt = os.path.join(base_dir, config["paths"]["output_dir"], "final_checkpoint.pt")
            metrics = checkpoint_metrics(final_ckpt)
            row = {"config": cfg["name"], "seed": seed, **metrics}
            rows.append(row)

    detail_path = os.path.join(output_root, "ablation_results_by_seed.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "seed"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows)
    summary_path = os.path.join(output_root, "ablation_summary.csv")
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Wrote per-seed results to {detail_path}")
    print(f"Wrote summary results to {summary_path}")


if __name__ == "__main__":
    main()
