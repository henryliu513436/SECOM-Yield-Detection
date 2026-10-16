"""train.py 的訓練路徑：尾端只拿來選閾值、超參數由時間序列 CV 從網格選出。"""

import numpy as np
import pandas as pd
import pytest

from train import XGBParams, fit_xgboost_pipeline, select_xgb_params


def _synthetic(n: int = 300, fail_every: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """少量有訊號的合成資料：Fail 的第 0 欄偏高，讓模型有東西可學。"""
    rng = np.random.default_rng(seed)
    y = pd.Series((np.arange(n) % fail_every == 0).astype("int8"), name="fail")
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[str(i) for i in range(6)])
    X["0"] += y * 1.5
    return X, y


def test_fit_trains_exactly_n_estimators_without_early_stopping() -> None:
    X, y = _synthetic()
    params = XGBParams(learning_rate=0.1, n_estimators=37)

    result = fit_xgboost_pipeline(X, y, params)

    booster = result.pipeline.named_steps["model"].get_booster()
    assert booster.num_boosted_rounds() == 37
    assert result.params == params


def test_fit_holds_out_tail_for_threshold_selection() -> None:
    X, y = _synthetic(n=300)

    result = fit_xgboost_pipeline(X, y, XGBParams(0.1, 20))

    assert result.val_labels is not None
    assert len(result.val_labels) == len(result.val_scores) == 60  # 300 × 20%
    np.testing.assert_array_equal(result.val_labels, y.iloc[-60:].to_numpy())


def test_fit_raises_when_fit_portion_has_no_fail() -> None:
    X, y = _synthetic(n=200)
    y = y * 0

    with pytest.raises(ValueError, match="正樣本數為 0"):
        fit_xgboost_pipeline(X, y, XGBParams(0.1, 10))


def test_select_xgb_params_returns_grid_member_and_full_table() -> None:
    X, y = _synthetic(n=400)
    grid = (XGBParams(0.1, 10), XGBParams(0.1, 40))

    best, table = select_xgb_params(X, y, n_splits=3, grid=grid)

    assert best in grid
    assert [row.params for row in table] == list(grid)
    best_row = max(table, key=lambda row: row.mean_pr_auc_ratio)
    assert best_row.params == best
