from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def draw_one(
    csv_path: Path,
    output_path: Path,
    horizon: int,
    unit: str,
) -> None:
    data = pd.read_csv(csv_path, parse_dates=["date"])

    required = {
        "date",
        "ground_truth",
        "prediction_mean",
        "prediction_std",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少字段：{sorted(missing)}")

    ground_truth = data["ground_truth"].to_numpy()
    prediction_mean = data["prediction_mean"].to_numpy()
    prediction_std = data["prediction_std"].to_numpy()

    if unit == "kwh":
        ground_truth = ground_truth / 60.0
        prediction_mean = prediction_mean / 60.0
        prediction_std = prediction_std / 60.0
        ylabel = "Daily Energy Consumption (kWh)"
    else:
        ylabel = "Daily Sum of Global Active Power"

    lower = prediction_mean - prediction_std
    upper = prediction_mean + prediction_std

    plt.figure(figsize=(13, 5))
    plt.plot(data["date"], ground_truth, label="Ground Truth")
    plt.plot(data["date"], prediction_mean, label="Mean LSTM Prediction")
    plt.fill_between(
        data["date"],
        lower,
        upper,
        alpha=0.25,
        label="±1 Standard Deviation",
    )
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(f"LSTM Mean Forecast over 5 Runs (Horizon={horizon})")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("已保存：", output_path)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="使用已有LSTM预测CSV重新绘图，不重新训练模型。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "lstm_outputs",
        help="lstm_outputs所在目录。",
    )
    parser.add_argument(
        "--horizon",
        choices=["90", "365", "both"],
        default="both",
    )
    parser.add_argument(
        "--unit",
        choices=["sum", "kwh"],
        default="sum",
        help="sum仅修正纵轴名称；kwh会将数值除以60。",
    )
    args = parser.parse_args()

    horizons = [90, 365] if args.horizon == "both" else [int(args.horizon)]

    for horizon in horizons:
        csv_path = (
            args.output_dir
            / "predictions"
            / f"lstm_h{horizon}_mean_prediction.csv"
        )

        suffix = "kwh" if args.unit == "kwh" else "relabeled"
        output_path = (
            args.output_dir
            / "figures"
            / f"lstm_h{horizon}_mean_prediction_{suffix}.png"
        )

        if not csv_path.exists():
            print("跳过，未找到：", csv_path)
            continue

        draw_one(
            csv_path=csv_path,
            output_path=output_path,
            horizon=horizon,
            unit=args.unit,
        )


if __name__ == "__main__":
    main()
