"""欄位清理：包成 sklearn transformer，fit 只能接觸訓練集。

四個步驟（常數欄、高缺失欄、高相關欄、中位數補值）包在同一個
transformer 裡，放進 Pipeline 後，TimeSeriesSplit 的每一折都會各自
正確地只 fit 到當折的訓練部分，不必依賴呼叫端自律「只傳訓練集」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING_RATE_THRESHOLD = 0.5
CORRELATION_THRESHOLD = 0.95


@dataclass(frozen=True)
class StepReport:
    """單一清理步驟的結果。"""

    name: str
    dropped: int
    remaining: int


def find_constant_columns(features: pd.DataFrame) -> tuple[str, ...]:
    """常數欄。全為缺失的欄 nunique 為 0，也會被歸在這裡。"""
    distinct = features.nunique(dropna=True)
    return tuple(distinct[distinct <= 1].index)


def find_high_missing_columns(
    features: pd.DataFrame,
    threshold: float = MISSING_RATE_THRESHOLD,
) -> tuple[str, ...]:
    """缺失率高於門檻的欄。"""
    rates = features.isna().mean()
    return tuple(rates[rates > threshold].index)


def _connected_components(adjacency: np.ndarray) -> list[list[int]]:
    """找出鄰接矩陣裡的連通分量，回傳每個分量的欄位索引清單。

    相關性不只發生在「成對」欄位之間：A-B、B-C 都高相關，但 A-C 不一
    定超過門檻，三者仍是同一份重複資訊。用 BFS 找完整分量，不能只看
    欄位是否與「欄號比自己小」的欄位相關——那樣會漏掉 B 只跟後面的 C
    相關、卻不跟前面的 A 直接相關的情況。
    """
    n = adjacency.shape[0]
    visited = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        stack, component = [start], []
        visited[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(component)

    return components


def find_correlated_columns(
    features: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[str, ...]:
    """相關係數 > threshold 的欄位視為同一組，每組只留一欄。

    保留規則：缺失率較低者優先；缺失率相同則保留欄號較小者。理由見
    README。此時資料還有缺失值，corr 用成對可用值算；不先補值是因為
    補值的統計量本身也只能來自訓練集，順序上排在這步之後。
    """
    correlation = features.corr().abs()
    adjacency = correlation.to_numpy() > threshold
    np.fill_diagonal(adjacency, False)
    missing_rate = features.isna().mean()

    columns = list(features.columns)
    dropped: list[str] = []
    for component in _connected_components(adjacency):
        if len(component) == 1:
            continue
        survivor = min(component, key=lambda idx: (missing_rate.iloc[idx], int(columns[idx])))
        dropped.extend(columns[idx] for idx in component if idx != survivor)

    return tuple(dropped)


class SecomColumnCleaner(BaseEstimator, TransformerMixin):
    """依序移除常數欄、高缺失欄、高相關欄，剩餘缺失用中位數補值。

    所有決定在 fit 時從傳入的 X 記錄下來（fitted attributes 以 trailing
    underscore 標記）；transform 只套用既有紀錄，不重新計算任何統計量。
    放進 Pipeline 後，X 在每一折會被 TimeSeriesSplit 換成當折的訓練
    部分，fit 因此永遠不會碰到驗證或測試資料。
    """

    def __init__(
        self,
        missing_threshold: float = MISSING_RATE_THRESHOLD,
        correlation_threshold: float = CORRELATION_THRESHOLD,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SecomColumnCleaner:
        reports: list[StepReport] = []

        remaining = X
        dropped = find_constant_columns(remaining)
        remaining = remaining.drop(columns=list(dropped))
        reports.append(StepReport("1. 常數欄", len(dropped), remaining.shape[1]))

        dropped = find_high_missing_columns(remaining, self.missing_threshold)
        remaining = remaining.drop(columns=list(dropped))
        reports.append(
            StepReport(
                f"2. 缺失率 > {self.missing_threshold:.0%}", len(dropped), remaining.shape[1]
            )
        )

        dropped = find_correlated_columns(remaining, self.correlation_threshold)
        remaining = remaining.drop(columns=list(dropped))
        reports.append(
            StepReport(
                f"3. 相關係數 > {self.correlation_threshold}", len(dropped), remaining.shape[1]
            )
        )

        self.columns_to_keep_ = tuple(remaining.columns)
        self.fill_values_ = remaining.median()
        self.step_reports_ = tuple(reports)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.columns_to_keep_) - set(X.columns)
        if missing:
            raise ValueError(f"待套用的資料缺少 fit 時記錄的欄位：{sorted(missing)[:10]}")

        return X[list(self.columns_to_keep_)].fillna(self.fill_values_)


def format_cleaning_report(cleaner: SecomColumnCleaner) -> str:
    """把 fit 後的逐步清理結果排成人看得懂的表。"""
    lines = [f"起始欄數：{cleaner.n_features_in_}"]
    for report in cleaner.step_reports_:
        lines.append(
            f"  {report.name:<22} 砍掉 {report.dropped:>3} 欄 → 剩 {report.remaining:>3} 欄"
        )
    lines.append(f"最終保留：{len(cleaner.columns_to_keep_)} 欄")
    return "\n".join(lines)


if __name__ == "__main__":
    import data as D

    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)

    train_features = D.extract_features(split.train)
    test_features = D.extract_features(split.test)

    cleaner = SecomColumnCleaner().fit(train_features)
    train_clean = cleaner.transform(train_features)
    test_clean = cleaner.transform(test_features)

    print(format_cleaning_report(cleaner))
    print()
    print(f"train 清理後缺失值：{int(train_clean.isna().sum().sum())}")
    print(f"test  清理後缺失值：{int(test_clean.isna().sum().sum())}")

    leak_check_column = next(
        c for c in cleaner.columns_to_keep_ if test_features[c].isna().any()
    )
    filled = test_clean.loc[test_features[leak_check_column].isna(), leak_check_column].iloc[0]
    print()
    print(f"洩漏檢查（欄 {leak_check_column}）：")
    print(f"  套用的補值   = {filled:.4f}")
    print(f"  訓練集中位數 = {train_features[leak_check_column].median():.4f}  ← 應相同")
    print(f"  測試集中位數 = {test_features[leak_check_column].median():.4f}  ← 應不同")
