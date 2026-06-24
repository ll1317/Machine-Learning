from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset import FEATURE_COLUMNS, prepare_task_data, set_seed
from models import LSTMForecaster


DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def choose_device(requested: str) -> torch.device:
    """选择训练设备。"""
    requested = requested.lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境未检测到可用 CUDA，请改用 --device cpu。")

    return torch.device(requested)


def configure_reproducibility(seed: int) -> None:
    """设置随机种子，并尽量保证实验可重复。"""
    set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """在原始尺度上计算 MSE 和 MAE。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"真实值和预测值形状不一致：{y_true.shape} vs {y_pred.shape}"
        )

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, mae


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = criterion(prediction, y)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"训练损失出现非有限值：{loss.item()}")

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        prediction = model(x)
        loss = criterion(prediction, y)

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def predict_scaled(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """得到标准化尺度下的预测值与真实值。"""
    model.eval()

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        prediction = model(x)

        predictions.append(prediction.cpu().numpy())
        targets.append(y.numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def plot_training_history(
    history: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (standardized scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_prediction(
    dates: pd.Series,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 5))
    plt.plot(dates, ground_truth, label="Ground Truth")
    plt.plot(dates, prediction, label="LSTM Prediction")
    plt.xlabel("Date")
    plt.ylabel("Daily Global Active Power")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_mean_prediction(
    dates: pd.Series,
    ground_truth: np.ndarray,
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lower = prediction_mean - prediction_std
    upper = prediction_mean + prediction_std

    plt.figure(figsize=(13, 5))
    plt.plot(dates, ground_truth, label="Ground Truth")
    plt.plot(dates, prediction_mean, label="Mean LSTM Prediction")
    plt.fill_between(
        dates,
        lower,
        upper,
        alpha=0.25,
        label="±1 Standard Deviation",
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Global Active Power")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_one_run(
    data_dir: Path,
    output_root: Path,
    horizon: int,
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    grad_clip: float,
    num_workers: int,
) -> Dict:
    configure_reproducibility(seed)

    bundle = prepare_task_data(
        train_csv=data_dir / "train.csv",
        test_csv=data_dir / "test.csv",
        horizon=horizon,
        batch_size=batch_size,
        stride=1,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = LSTMForecaster(
        input_dim=len(FEATURE_COLUMNS),
        horizon=horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, patience // 4),
        min_lr=1e-6,
    )

    checkpoint_dir = output_root / "checkpoints"
    history_dir = output_root / "history"
    prediction_dir = output_root / "predictions"
    figure_dir = output_root / "figures"

    checkpoint_path = checkpoint_dir / f"lstm_h{horizon}_seed{seed}.pt"
    history_path = history_dir / f"lstm_h{horizon}_seed{seed}_history.csv"
    prediction_path = (
        prediction_dir / f"lstm_h{horizon}_seed{seed}_prediction.csv"
    )

    config = {
        "model": "LSTM",
        "horizon": horizon,
        "seed": seed,
        "input_dim": len(FEATURE_COLUMNS),
        "input_length": bundle.input_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "grad_clip": grad_clip,
    }

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: List[Dict] = []

    start_time = time.time()

    print(
        f"\n开始训练：LSTM | horizon={horizon} | seed={seed} | "
        f"device={device}"
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=bundle.val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        improved = val_loss < best_val_loss - 1e-8

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                config=config,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f} | "
                f"lr={current_lr:.2e} | "
                f"best={best_val_loss:.6f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"触发早停：连续 {patience} 轮验证损失未改善。"
            )
            break

    history = pd.DataFrame(history_rows)
    history_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False, encoding="utf-8-sig")

    plot_training_history(
        history=history,
        output_path=figure_dir
        / f"lstm_h{horizon}_seed{seed}_loss.png",
        title=f"LSTM Training Curve (Horizon={horizon}, Seed={seed})",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    prediction_scaled, target_scaled = predict_scaled(
        model=model,
        loader=bundle.test_loader,
        device=device,
    )

    prediction = bundle.inverse_transform_target(prediction_scaled)[0]
    ground_truth = bundle.inverse_transform_target(target_scaled)[0]

    mse, mae = compute_metrics(ground_truth, prediction)

    test_frame = pd.read_csv(
        data_dir / "test.csv",
        parse_dates=["date"],
    ).iloc[:horizon]

    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "date": test_frame["date"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "absolute_error": np.abs(ground_truth - prediction),
            "squared_error": (ground_truth - prediction) ** 2,
        }
    )
    prediction_frame.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_single_prediction(
        dates=test_frame["date"],
        ground_truth=ground_truth,
        prediction=prediction,
        output_path=figure_dir
        / f"lstm_h{horizon}_seed{seed}_prediction.png",
        title=f"LSTM Forecast (Horizon={horizon}, Seed={seed})",
    )

    elapsed_seconds = time.time() - start_time

    print(
        f"完成：horizon={horizon}, seed={seed}, "
        f"MSE={mse:.6f}, MAE={mae:.6f}, "
        f"best_epoch={best_epoch}"
    )

    return {
        "model": "LSTM",
        "horizon": horizon,
        "seed": seed,
        "mse": mse,
        "mae": mae,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed_seconds,
        "prediction": prediction,
        "ground_truth": ground_truth,
    }


def aggregate_horizon_results(
    data_dir: Path,
    output_root: Path,
    horizon: int,
    run_results: List[Dict],
) -> Dict:
    result_dir = output_root / "results"
    prediction_dir = output_root / "predictions"
    figure_dir = output_root / "figures"

    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {"prediction", "ground_truth"}
            }
            for result in run_results
        ]
    )

    metrics_path = result_dir / f"lstm_h{horizon}_five_runs.csv"
    metrics_frame.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_matrix = np.stack(
        [result["prediction"] for result in run_results],
        axis=0,
    )
    ground_truth = run_results[0]["ground_truth"]

    prediction_mean = prediction_matrix.mean(axis=0)
    prediction_std = prediction_matrix.std(axis=0, ddof=1)

    test_frame = pd.read_csv(
        data_dir / "test.csv",
        parse_dates=["date"],
    ).iloc[:horizon]

    mean_prediction_frame = pd.DataFrame(
        {
            "date": test_frame["date"],
            "ground_truth": ground_truth,
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "prediction_lower": prediction_mean - prediction_std,
            "prediction_upper": prediction_mean + prediction_std,
        }
    )

    mean_prediction_path = (
        prediction_dir / f"lstm_h{horizon}_mean_prediction.csv"
    )
    mean_prediction_frame.to_csv(
        mean_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_mean_prediction(
        dates=test_frame["date"],
        ground_truth=ground_truth,
        prediction_mean=prediction_mean,
        prediction_std=prediction_std,
        output_path=figure_dir
        / f"lstm_h{horizon}_mean_prediction.png",
        title=f"LSTM Mean Forecast over {len(run_results)} Runs "
        f"(Horizon={horizon})",
    )

    summary = {
        "model": "LSTM",
        "horizon": horizon,
        "num_runs": len(run_results),
        "mse_mean": float(metrics_frame["mse"].mean()),
        "mse_std": float(metrics_frame["mse"].std(ddof=1)),
        "mae_mean": float(metrics_frame["mae"].mean()),
        "mae_std": float(metrics_frame["mae"].std(ddof=1)),
        "best_epoch_mean": float(metrics_frame["best_epoch"].mean()),
        "elapsed_seconds_mean": float(
            metrics_frame["elapsed_seconds"].mean()
        ),
    }

    print("\n" + "-" * 72)
    print(f"LSTM horizon={horizon} 的 {len(run_results)} 轮汇总：")
    print(
        f"MSE = {summary['mse_mean']:.6f} "
        f"± {summary['mse_std']:.6f}"
    )
    print(
        f"MAE = {summary['mae_mean']:.6f} "
        f"± {summary['mae_std']:.6f}"
    )
    print("-" * 72)

    return summary


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="训练并评估家庭电力消耗 LSTM 基线模型。"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir,
        help="train.csv、test.csv 和 dataset.py 所在目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "lstm_outputs",
        help="模型、结果和图像输出目录。",
    )
    parser.add_argument(
        "--horizon",
        choices=["90", "365", "both"],
        default="both",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs 必须大于0。")
    if args.patience <= 0:
        raise ValueError("patience 必须大于0。")
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于0。")
    if len(args.seeds) == 0:
        raise ValueError("至少需要提供一个随机种子。")

    device = choose_device(args.device)

    horizons = (
        [90, 365]
        if args.horizon == "both"
        else [int(args.horizon)]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict] = []

    run_config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "horizons": horizons,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "grad_clip": args.grad_clip,
        "device": str(device),
    }

    with (args.output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    print("运行配置：")
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    for horizon in horizons:
        run_results: List[Dict] = []

        for seed in args.seeds:
            result = train_one_run(
                data_dir=args.data_dir,
                output_root=args.output_dir,
                horizon=horizon,
                seed=seed,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                grad_clip=args.grad_clip,
                num_workers=args.num_workers,
            )
            run_results.append(result)

        summary = aggregate_horizon_results(
            data_dir=args.data_dir,
            output_root=args.output_dir,
            horizon=horizon,
            run_results=run_results,
        )
        all_summaries.append(summary)

    summary_frame = pd.DataFrame(all_summaries)
    summary_path = args.output_dir / "results" / "lstm_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n全部 LSTM 实验完成。")
    print("汇总结果：", summary_path)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
