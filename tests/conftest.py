"""讓 tests/ 底下的測試能直接 import src/ 裡的模組（data、features、api...）。

src/ 裡的模組彼此用平行匯入（例如 train.py 寫 `import data as D`），不是
一個 package，所以測試端只能靠調整 sys.path，不能改成套件式匯入。
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "metadata.json"


def pytest_configure(config) -> None:
    """`api.py` 在匯入時就讀 models/metadata.json，缺檔案會直接匯入失敗。

    metadata.json 是 `python train.py` 的訓練產物、不進 git，全新環境
    第一次跑測試時通常還沒有，這裡在測試蒐集前補跑一次訓練把它生出
    來，測試套件才能不靠手動步驟自己跑得起來。
    """
    if METADATA_PATH.is_file():
        return

    import data as D
    from train import (
        XGB_THRESHOLD_MIN_PRECISION,
        fit_xgboost_pipeline,
        save_production_metadata,
        select_threshold_at_min_precision,
    )

    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    X_train = D.extract_features(split.train)
    y_train = D.extract_labels(split.train)

    xgb_result = fit_xgboost_pipeline(X_train, y_train)
    threshold = select_threshold_at_min_precision(
        xgb_result.val_labels, xgb_result.val_scores, XGB_THRESHOLD_MIN_PRECISION
    )
    save_production_metadata(X_train, xgb_result, threshold)
