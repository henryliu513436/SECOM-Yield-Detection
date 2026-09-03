"""CLAUDE.md 階段6指定的兩個洩漏防範測試：時間切分與補值統計量的來源。"""

from pathlib import Path

import data as D
from features import SecomColumnCleaner

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"


def test_no_time_leakage() -> None:
    """訓練集最晚時間必須早於測試集最早時間，否則兩者在時間上有重疊。"""
    raw = D.load_raw(CSV_PATH)
    split = D.split_by_time(raw)

    train_latest = split.train[D.TIME_COLUMN].max()
    test_earliest = split.test[D.TIME_COLUMN].min()

    assert train_latest < test_earliest


def test_imputation_uses_train_only() -> None:
    """補值中位數必須等於訓練集中位數、且不等於全體中位數，才能確認補值沒看過測試集。"""
    raw = D.load_raw(CSV_PATH)
    split = D.split_by_time(raw)
    X_train = D.extract_features(split.train)
    X_all = D.extract_features(raw)

    cleaner = SecomColumnCleaner().fit(X_train)

    # 找一欄訓練集中位數確實跟全體中位數不同，測試才有意義（不是巧合相等）。
    column = next(
        c for c in cleaner.columns_to_keep_ if X_train[c].median() != X_all[c].median()
    )

    assert cleaner.fill_values_[column] == X_train[column].median()
    assert cleaner.fill_values_[column] != X_all[column].median()
