from pathlib import Path

import pandas as pd


WEATHER_PATH = Path(r"D:\\Users\\ll\\桌面\\机器学习\\data\\weather_data.csv")


def read_weather(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        for separator in [";", ",", "\t"]:
            try:
                df = pd.read_csv(
                    path,
                    sep=separator,
                    encoding=encoding,
                    low_memory=False,
                )

                if df.shape[1] > 1:
                    print(
                        f"读取成功：encoding={encoding}, "
                        f"sep={repr(separator)}"
                    )
                    return df
            except Exception:
                continue

    raise RuntimeError("无法正确读取天气文件。")


weather = read_weather(WEATHER_PATH)

print("\n天气数据形状：", weather.shape)

print("\n全部列名：")
for index, column in enumerate(weather.columns):
    print(index, repr(column))

print("\n前5行：")
print(weather.head().to_string())

print("\n可能相关的列：")
keywords = [
    "POSTE",
    "NOM",
    "AAAAMM",
    "ANNEE",
    "MOIS",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
]

for column in weather.columns:
    upper = str(column).upper()
    if any(keyword in upper for keyword in keywords):
        print(column)