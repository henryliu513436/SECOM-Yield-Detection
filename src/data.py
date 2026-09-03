"""SECOM 原始資料的載入與時間切分。

依照「切分優先」原則，清理（欄位篩選、補值）完全不放在這裡——凡是
要從資料算出來的統計量，只能發生在切分之後、只在訓練集上。那些邏輯
在 src/features.py 裡以 sklearn transformer 呈現。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TIME_COLUMN = "Time"
LABEL_COLUMN = "Pass/Fail"
LABEL_NAME = "fail"

# 原始標籤 -1 = Pass、1 = Fail
PASS_RAW_VALUE = -1
FAIL_RAW_VALUE = 1

TRAIN_RATIO = 0.7
MIN_TEST_FAIL_WARNING = 20


def load_raw(csv_path: Path) -> pd.DataFrame:
    """載入原始 CSV 並依 Time 排序。

    原始檔有 6 筆列序與時間序不一致，且有 32 個重複時間戳；用 stable
    排序讓同時間戳的相對順序固定，切分邊界才有再現性。
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到原始資料：{csv_path}")

    frame = pd.read_csv(csv_path, parse_dates=[TIME_COLUMN])
    _validate_raw(frame)
    return frame.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


def _validate_raw(frame: pd.DataFrame) -> None:
    """在邊界擋掉格式不符的輸入，不要讓錯誤延後到訓練階段才爆。"""
    required = {TIME_COLUMN, LABEL_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"原始資料缺少必要欄位：{sorted(missing)}")

    unexpected = set(frame[LABEL_COLUMN].dropna().unique()) - {PASS_RAW_VALUE, FAIL_RAW_VALUE}
    if unexpected:
        raise ValueError(
            f"{LABEL_COLUMN} 出現非預期的值：{sorted(unexpected)}，"
            f"預期只有 {PASS_RAW_VALUE}（Pass）與 {FAIL_RAW_VALUE}（Fail）"
        )

    if frame[TIME_COLUMN].isna().any():
        raise ValueError(f"{TIME_COLUMN} 有缺失值，無法可靠排序與切分")


def extract_labels(frame: pd.DataFrame) -> pd.Series:
    """把 -1/1 轉成 0/1，Fail 為 1。"""
    return (frame[LABEL_COLUMN] == FAIL_RAW_VALUE).astype("int8").rename(LABEL_NAME)


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    """取出 590 個感測器欄。

    Time 只用於排序與切分，絕不進特徵，所以在這裡就丟掉。
    """
    features = frame.drop(columns=[TIME_COLUMN, LABEL_COLUMN])

    non_numeric = features.columns[~features.dtypes.map(pd.api.types.is_numeric_dtype)]
    if len(non_numeric) > 0:
        raise ValueError(f"感測器欄出現非數值型別：{list(non_numeric)[:10]}")

    return features


@dataclass(frozen=True)
class TimeSplit:
    """時間切分後的兩個子集，連同切分點本身。

    保留 cut_index 是因為報告與 test_no_time_leakage 這類測試都需要
    引用切分點，不只是兩個已經切好的 DataFrame。
    """

    train: pd.DataFrame
    test: pd.DataFrame
    cut_index: int


def split_by_time(frame: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> TimeSplit:
    """按時間排序後的前 train_ratio 切訓練集，其餘為測試集。

    frame 必須是 load_raw 排序過的結果；這裡只做位置切片、不重新排序，
    切分本身不該藏著隱性的重新運算。
    """
    if not frame[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError("frame 必須已依 Time 遞增排序，請先呼叫 load_raw")

    cut_index = int(len(frame) * train_ratio)
    return TimeSplit(
        train=frame.iloc[:cut_index].reset_index(drop=True),
        test=frame.iloc[cut_index:].reset_index(drop=True),
        cut_index=cut_index,
    )


def format_split_report(split: TimeSplit) -> str:
    """切分結果報告，含 CLAUDE.md 要求的測試集 Fail 數量警告。"""
    train_labels = extract_labels(split.train)
    test_labels = extract_labels(split.test)

    lines = [
        f"訓練集：{len(split.train)} 筆，"
        f"{split.train[TIME_COLUMN].min()} ~ {split.train[TIME_COLUMN].max()}，"
        f"Fail={int(train_labels.sum())}（{train_labels.mean():.2%}）",
        f"測試集：{len(split.test)} 筆，"
        f"{split.test[TIME_COLUMN].min()} ~ {split.test[TIME_COLUMN].max()}，"
        f"Fail={int(test_labels.sum())}（{test_labels.mean():.2%}）",
    ]

    if test_labels.sum() < MIN_TEST_FAIL_WARNING:
        lines.append(
            f"警告：測試集只有 {int(test_labels.sum())} 筆 Fail，"
            f"低於門檻 {MIN_TEST_FAIL_WARNING}，PR-AUC 等指標會很不穩定。"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"

    raw = load_raw(csv_path)
    split = split_by_time(raw)

    print(f"原始資料：{raw.shape[0]} 列 × {raw.shape[1]} 欄")
    print(f"時間範圍：{raw[TIME_COLUMN].min()} ~ {raw[TIME_COLUMN].max()}")
    print()
    print(format_split_report(split))
