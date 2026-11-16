"""Entry point for reproducing GSL-GCRNN."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from data_loader import apply_zscore, fit_zscore, load_csv, make_dataloader
from graph import correlation_matrix, load_adjacency_csv, to_tensor, topk_binarize
from models import GSLGCRNN
from train import evaluate, train_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GSL-GCRNN spectrum prediction")
    p.add_argument("--sensor", type=str, default=None, help="run a single sensor")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--device", type=str, default=config.get_device())
    p.add_argument("--graph-method", type=str, default=config.GRAPH_METHOD,
                   choices=["pearson", "spearman", "load_spearman"])
    return p.parse_args()


def build_graph(train_data: np.ndarray, method: str, sensor: str) -> np.ndarray:
    if method == "load_spearman":
        path = config.TRAIN_DATA / sensor / f"101_adj_matrix_{sensor}_0.5_spearman.csv"
        if not path.exists():
            path = config.TRAIN_DATA / sensor / "101_adj_bcn_L__0.5_spearman.csv"
        return load_adjacency_csv(path)
    s = correlation_matrix(train_data, method)
    return topk_binarize(s, config.TOP_K)


def spectrum_csv_name(sensor: str) -> str:
    """Map a sensor id to its CSV filename.

    Three of the four sensors use an underscore in the filename, but the bcn_L
    sensor is stored as ``bcn-L_...`` with a hyphen.
    """
    name = sensor.replace("bcn_L", "bcn-L")
    return f"{name}_202106010800-202106080800_1mins_2MHz.csv"


def run_sensor(sensor: str, args: argparse.Namespace) -> dict:
    train_csv = config.TRAIN_DATA / sensor / spectrum_csv_name(sensor)
    test_csv = config.TEST_DATA / sensor / spectrum_csv_name(sensor)

    train_raw = load_csv(train_csv)   # (T_train, N)
    test_raw = load_csv(test_csv)     # (T_test, N)

    mean, std = fit_zscore(train_raw)
    train_norm = apply_zscore(train_raw, mean, std)
    test_norm = apply_zscore(test_raw, mean, std)

    # Static adjacency is built from training data only (no test leakage).
    static_a = to_tensor(build_graph(train_raw, args.graph_method, sensor)).to(args.device)

    train_dl = make_dataloader(train_norm, config.WINDOW, config.HORIZON, config.BATCH_SIZE, True)
    test_dl = make_dataloader(test_norm, config.WINDOW, config.HORIZON, config.BATCH_SIZE, False)

    model = GSLGCRNN(
        n_nodes=config.N_BANDS,
        horizon=config.HORIZON,
        static_a=static_a,
        gcn_hidden=config.GCN_HIDDEN,
        gcn_layers=config.GCN_LAYERS,
        embed_dim=config.EMBED_DIM,
        gru_hidden=config.GRU_HIDDEN,
        top_k=config.TOP_K,
        mu=config.MU,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best = {"rmse": float("inf"), "epoch": -1}
    for epoch in range(args.epochs):
        # Paper: static A in the first epoch, learned A~ from the second epoch on.
        use_estimator = epoch >= 1
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, criterion, args.device, use_estimator)

        if epoch == 0 or (epoch + 1) % config.LOG_EVERY == 0:
            print(f"[{sensor}] epoch {epoch + 1}/{args.epochs} "
                  f"train_loss={train_loss:.6f} ({time.time() - t0:.1f}s)")

        if (epoch + 1) % config.EVAL_EVERY == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, test_dl, criterion, args.device, mean, std)
            print(f"[{sensor}] eval@{epoch + 1} "
                  f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}")
            if metrics["rmse"] < best["rmse"]:
                best = {"rmse": metrics["rmse"], "mae": metrics["mae"],
                        "r2": metrics["r2"], "epoch": epoch + 1}

    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt = config.OUTPUT_ROOT / f"gsl_gcrnn_{sensor}_best.pt"
    torch.save(model.state_dict(), ckpt)
    best["sensor"] = sensor
    return best


def main() -> None:
    args = parse_args()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    sensors = [args.sensor] if args.sensor else config.SENSORS
    results = {}
    for sensor in sensors:
        print(f"=== training on {sensor} ===")
        results[sensor] = run_sensor(sensor, args)

    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_ROOT / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults saved to {out_path}")
    for sensor, r in results.items():
        print(f"{sensor}: rmse={r['rmse']:.4f} mae={r['mae']:.4f} r2={r['r2']:.4f}")


if __name__ == "__main__":
    main()
