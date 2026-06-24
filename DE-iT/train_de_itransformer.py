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
from de_itransformer_model import DEITransformer


DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用CUDA，请使用 --device cpu。")

    return torch.device(requested)


def configure_reproducibility(seed: int) -> None:
    set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"真实值与预测值形状不同：{y_true.shape} vs {y_pred.shape}"
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
            raise FloatingPointError(
                f"训练损失出现非有限值：{loss.item()}"
            )

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
def predict_components(
    model: DEITransformer,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()

    final_predictions: List[np.ndarray] = []
    itransformer_predictions: List[np.ndarray] = []
    dlinear_predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        outputs = model(
            x,
            return_components=True,
        )

        final_predictions.append(
            outputs["prediction"].cpu().numpy()
        )
        itransformer_predictions.append(
            outputs["itransformer_prediction"].cpu().numpy()
        )
        dlinear_predictions.append(
            outputs["dlinear_prediction"].cpu().numpy()
        )
        targets.append(y.numpy())

    return {
        "prediction": np.concatenate(
            final_predictions,
            axis=0,
        ),
        "itransformer_prediction": np.concatenate(
            itransformer_predictions,
            axis=0,
        ),
        "dlinear_prediction": np.concatenate(
            dlinear_predictions,
            axis=0,
        ),
        "target": np.concatenate(targets, axis=0),
    }


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
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
    )
    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (standardized scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_prediction(
    dates: pd.Series,
    ground_truth_kwh: np.ndarray,
    prediction_kwh: np.ndarray,
    output_path: Path,
    title: str,
    model_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 5))
    plt.plot(
        dates,
        ground_truth_kwh,
        label="Ground Truth",
    )
    plt.plot(
        dates,
        prediction_kwh,
        label=f"{model_label} Prediction",
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Energy Consumption (kWh)")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_mean_prediction(
    dates: pd.Series,
    ground_truth_kwh: np.ndarray,
    prediction_mean_kwh: np.ndarray,
    prediction_std_kwh: np.ndarray,
    output_path: Path,
    title: str,
    model_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lower = prediction_mean_kwh - prediction_std_kwh
    upper = prediction_mean_kwh + prediction_std_kwh

    plt.figure(figsize=(13, 5))
    plt.plot(
        dates,
        ground_truth_kwh,
        label="Ground Truth",
    )
    plt.plot(
        dates,
        prediction_mean_kwh,
        label=f"Mean {model_label} Prediction",
    )
    plt.fill_between(
        dates,
        lower,
        upper,
        alpha=0.25,
        label="±1 Standard Deviation",
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Energy Consumption (kWh)")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_one_run(
    data_dir: Path,
    output_root: Path,
    mode: str,
    horizon: int,
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    moving_average_kernel: int,
    gate_init: float,
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

    model = DEITransformer(
        input_length=bundle.input_length,
        num_features=len(FEATURE_COLUMNS),
        horizon=horizon,
        target_index=0,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        moving_average_kernel=moving_average_kernel,
        mode=mode,
        gate_init=gate_init,
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

    checkpoint_path = (
        checkpoint_dir
        / f"{mode}_h{horizon}_seed{seed}.pt"
    )
    history_path = (
        history_dir
        / f"{mode}_h{horizon}_seed{seed}_history.csv"
    )
    prediction_path = (
        prediction_dir
        / f"{mode}_h{horizon}_seed{seed}_prediction.csv"
    )

    config = {
        "model": mode,
        "horizon": horizon,
        "seed": seed,
        "input_length": bundle.input_length,
        "num_features": len(FEATURE_COLUMNS),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "moving_average_kernel": moving_average_kernel,
        "gate_init": gate_init,
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
        f"\n开始训练：{mode} | horizon={horizon} | "
        f"seed={seed} | device={device}"
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
                "fusion_weight": float(
                    model.fusion_weight.detach().cpu().item()
                ),
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
                f"gate={model.fusion_weight.item():.4f} | "
                f"best={best_val_loss:.6f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"触发早停：连续 {patience} 轮验证损失未改善。"
            )
            break

    history = pd.DataFrame(history_rows)
    history_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_training_history(
        history=history,
        output_path=figure_dir
        / f"{mode}_h{horizon}_seed{seed}_loss.png",
        title=f"{mode} Training Curve "
        f"(Horizon={horizon}, Seed={seed})",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    outputs_scaled = predict_components(
        model=model,
        loader=bundle.test_loader,
        device=device,
    )

    prediction_raw = bundle.inverse_transform_target(
        outputs_scaled["prediction"]
    )[0]
    itransformer_raw = bundle.inverse_transform_target(
        outputs_scaled["itransformer_prediction"]
    )[0]
    dlinear_raw = bundle.inverse_transform_target(
        outputs_scaled["dlinear_prediction"]
    )[0]
    ground_truth_raw = bundle.inverse_transform_target(
        outputs_scaled["target"]
    )[0]

    prediction_kwh = prediction_raw / 60.0
    itransformer_kwh = itransformer_raw / 60.0
    dlinear_kwh = dlinear_raw / 60.0
    ground_truth_kwh = ground_truth_raw / 60.0

    mse_raw, mae_raw = compute_metrics(
        ground_truth_raw,
        prediction_raw,
    )
    mse_kwh2, mae_kwh = compute_metrics(
        ground_truth_kwh,
        prediction_kwh,
    )

    test_frame = pd.read_csv(
        data_dir / "test.csv",
        parse_dates=["date"],
    ).iloc[:horizon]

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_frame = pd.DataFrame(
        {
            "date": test_frame["date"],
            "ground_truth_raw": ground_truth_raw,
            "prediction_raw": prediction_raw,
            "ground_truth_kwh": ground_truth_kwh,
            "prediction_kwh": prediction_kwh,
            "itransformer_prediction_kwh": itransformer_kwh,
            "dlinear_prediction_kwh": dlinear_kwh,
            "absolute_error_kwh": np.abs(
                ground_truth_kwh - prediction_kwh
            ),
            "squared_error_kwh2": (
                ground_truth_kwh - prediction_kwh
            ) ** 2,
        }
    )
    prediction_frame.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_single_prediction(
        dates=test_frame["date"],
        ground_truth_kwh=ground_truth_kwh,
        prediction_kwh=prediction_kwh,
        output_path=figure_dir
        / f"{mode}_h{horizon}_seed{seed}_prediction.png",
        title=f"{mode} Forecast "
        f"(Horizon={horizon}, Seed={seed})",
        model_label=mode,
    )

    elapsed_seconds = time.time() - start_time
    fusion_weight = float(
        model.fusion_weight.detach().cpu().item()
    )

    print(
        f"完成：mode={mode}, horizon={horizon}, seed={seed}, "
        f"MSE(kWh²)={mse_kwh2:.6f}, "
        f"MAE(kWh)={mae_kwh:.6f}, "
        f"gate={fusion_weight:.4f}, "
        f"best_epoch={best_epoch}"
    )

    return {
        "model": mode,
        "horizon": horizon,
        "seed": seed,
        "mse_raw": mse_raw,
        "mae_raw": mae_raw,
        "mse_kwh2": mse_kwh2,
        "mae_kwh": mae_kwh,
        "fusion_weight": fusion_weight,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed_seconds,
        "prediction_kwh": prediction_kwh,
        "ground_truth_kwh": ground_truth_kwh,
    }


def aggregate_horizon_results(
    data_dir: Path,
    output_root: Path,
    mode: str,
    horizon: int,
    run_results: List[Dict],
) -> Dict:
    result_dir = output_root / "results"
    prediction_dir = output_root / "predictions"
    figure_dir = output_root / "figures"

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "prediction_kwh",
                    "ground_truth_kwh",
                }
            }
            for result in run_results
        ]
    )

    metrics_path = (
        result_dir
        / f"{mode}_h{horizon}_five_runs.csv"
    )
    metrics_frame.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_matrix = np.stack(
        [
            result["prediction_kwh"]
            for result in run_results
        ],
        axis=0,
    )

    ground_truth_kwh = (
        run_results[0]["ground_truth_kwh"]
    )

    prediction_mean_kwh = prediction_matrix.mean(
        axis=0
    )

    if len(run_results) > 1:
        prediction_std_kwh = prediction_matrix.std(
            axis=0,
            ddof=1,
        )
        mse_std = float(
            metrics_frame["mse_kwh2"].std(ddof=1)
        )
        mae_std = float(
            metrics_frame["mae_kwh"].std(ddof=1)
        )
        mse_raw_std = float(
            metrics_frame["mse_raw"].std(ddof=1)
        )
        mae_raw_std = float(
            metrics_frame["mae_raw"].std(ddof=1)
        )
        gate_std = float(
            metrics_frame["fusion_weight"].std(ddof=1)
        )
    else:
        prediction_std_kwh = np.zeros_like(
            prediction_mean_kwh
        )
        mse_std = 0.0
        mae_std = 0.0
        mse_raw_std = 0.0
        mae_raw_std = 0.0
        gate_std = 0.0

    test_frame = pd.read_csv(
        data_dir / "test.csv",
        parse_dates=["date"],
    ).iloc[:horizon]

    mean_prediction_frame = pd.DataFrame(
        {
            "date": test_frame["date"],
            "ground_truth_kwh": ground_truth_kwh,
            "prediction_mean_kwh": prediction_mean_kwh,
            "prediction_std_kwh": prediction_std_kwh,
            "prediction_lower_kwh": (
                prediction_mean_kwh
                - prediction_std_kwh
            ),
            "prediction_upper_kwh": (
                prediction_mean_kwh
                + prediction_std_kwh
            ),
        }
    )

    mean_prediction_path = (
        prediction_dir
        / f"{mode}_h{horizon}_mean_prediction.csv"
    )
    mean_prediction_frame.to_csv(
        mean_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_mean_prediction(
        dates=test_frame["date"],
        ground_truth_kwh=ground_truth_kwh,
        prediction_mean_kwh=prediction_mean_kwh,
        prediction_std_kwh=prediction_std_kwh,
        output_path=figure_dir
        / f"{mode}_h{horizon}_mean_prediction.png",
        title=f"{mode} Mean Forecast over "
        f"{len(run_results)} Runs "
        f"(Horizon={horizon})",
        model_label=mode,
    )

    summary = {
        "model": mode,
        "horizon": horizon,
        "num_runs": len(run_results),
        "mse_raw_mean": float(
            metrics_frame["mse_raw"].mean()
        ),
        "mse_raw_std": mse_raw_std,
        "mae_raw_mean": float(
            metrics_frame["mae_raw"].mean()
        ),
        "mae_raw_std": mae_raw_std,
        "mse_kwh2_mean": float(
            metrics_frame["mse_kwh2"].mean()
        ),
        "mse_kwh2_std": mse_std,
        "mae_kwh_mean": float(
            metrics_frame["mae_kwh"].mean()
        ),
        "mae_kwh_std": mae_std,
        "fusion_weight_mean": float(
            metrics_frame["fusion_weight"].mean()
        ),
        "fusion_weight_std": gate_std,
        "best_epoch_mean": float(
            metrics_frame["best_epoch"].mean()
        ),
        "elapsed_seconds_mean": float(
            metrics_frame["elapsed_seconds"].mean()
        ),
    }

    print("\n" + "-" * 72)
    print(
        f"{mode} horizon={horizon} 的 "
        f"{len(run_results)} 轮汇总："
    )
    print(
        f"MSE(kWh²) = "
        f"{summary['mse_kwh2_mean']:.6f} "
        f"± {summary['mse_kwh2_std']:.6f}"
    )
    print(
        f"MAE(kWh) = "
        f"{summary['mae_kwh_mean']:.6f} "
        f"± {summary['mae_kwh_std']:.6f}"
    )
    print(
        f"融合权重 alpha = "
        f"{summary['fusion_weight_mean']:.4f} "
        f"± {summary['fusion_weight_std']:.4f}"
    )
    print("-" * 72)

    return summary


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "训练分解增强倒置Transformer，"
            "并支持iTransformer与DLinear消融。"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        choices=[
            "deitransformer",
            "itransformer",
            "dlinear",
        ],
        default="deitransformer",
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--moving-average-kernel",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--gate-init",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    if args.d_model % args.nhead != 0:
        raise ValueError(
            "d_model 必须能被 nhead 整除。"
        )
    if args.moving_average_kernel % 2 == 0:
        raise ValueError(
            "moving_average_kernel 必须为奇数。"
        )
    if len(args.seeds) == 0:
        raise ValueError(
            "至少提供一个随机种子。"
        )

    if args.output_dir is None:
        args.output_dir = (
            script_dir / f"{args.model}_outputs"
        )

    device = choose_device(args.device)

    horizons = (
        [90, 365]
        if args.horizon == "both"
        else [int(args.horizon)]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "horizons": horizons,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "moving_average_kernel": (
            args.moving_average_kernel
        ),
        "gate_init": args.gate_init,
        "grad_clip": args.grad_clip,
        "device": str(device),
    }

    with (
        args.output_dir / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("运行配置：")
    print(
        json.dumps(
            run_config,
            ensure_ascii=False,
            indent=2,
        )
    )

    all_summaries: List[Dict] = []

    for horizon in horizons:
        run_results: List[Dict] = []

        for seed in args.seeds:
            result = train_one_run(
                data_dir=args.data_dir,
                output_root=args.output_dir,
                mode=args.model,
                horizon=horizon,
                seed=seed,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                moving_average_kernel=(
                    args.moving_average_kernel
                ),
                gate_init=args.gate_init,
                grad_clip=args.grad_clip,
                num_workers=args.num_workers,
            )
            run_results.append(result)

        summary = aggregate_horizon_results(
            data_dir=args.data_dir,
            output_root=args.output_dir,
            mode=args.model,
            horizon=horizon,
            run_results=run_results,
        )
        all_summaries.append(summary)

    summary_frame = pd.DataFrame(
        all_summaries
    )
    summary_path = (
        args.output_dir
        / "results"
        / f"{args.model}_summary.csv"
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_frame.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n全部 {args.model} 实验完成。"
    )
    print(
        "汇总结果：",
        summary_path,
    )
    print(
        summary_frame.to_string(index=False)
    )


if __name__ == "__main__":
    main()
