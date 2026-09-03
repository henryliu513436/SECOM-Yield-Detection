# SECOM 良率異常偵測

## 專案目標

用 UCI SECOM 半導體製程資料，建立一個不良批次偵測系統，並包成可呼叫的 API 服務。
重點是方法論的正確性與可解釋性，不是追求高分。

## 最高原則：切分優先

**任何依賴資料統計量的決定，都必須在切分之後、且只能從訓練集計算。**

這條原則優先於本文件的所有其他規定。如果任何步驟看起來與它衝突，服從這條原則，
並在回報時明確告訴我衝突在哪。

以下每一項都是「必須從訓練集計算」的統計量：
- 要砍掉哪些欄（常數、高缺失、高相關）
- 補值用的中位數
- IsolationForest 的 `contamination`
- XGBoost 的 `scale_pos_weight`
- 早停用的驗證集
- 判定閾值

## 工作方式

**一次只做一個階段，做完停下來等我確認，不要一次寫完整個專案。**

階段順序：
1. 載入與時間切分
2. 清理（fit 在訓練集，transform 套用到兩邊）
3. 模型訓練
4. 評估
5. SHAP 解釋
6. FastAPI 服務
7. （選配）LLM agent 層

每個階段結束時說明你做了什麼、以及有哪些地方我應該自己確認。
如果你發現本文件有矛盾或不合理之處，直接指出來，不要默默繞過。

## 資料

單一檔案 `data/raw/uci-secom.csv`，1567 列 × 592 欄。

| 欄位 | 說明 |
|---|---|
| `Time` | 時間戳。**只用於排序與切分，絕對不可當作特徵** |
| `0` ~ `589` | 590 個匿名感測器讀值，含大量缺失 |
| `Pass/Fail` | 標籤。**`-1` = Pass，`1` = Fail**。轉成 0/1 時 Fail 為 1 |

已知事實：Fail 共 104 筆（6.6%），全表缺失值約 41951 個。

---

## 階段 1：載入與時間切分

- 按 `Time` 排序（先 `pd.to_datetime`）
- 前 70% 訓練、後 30% 測試
- 交叉驗證用 `sklearn.model_selection.TimeSeriesSplit`，5 折
- 禁止 `train_test_split` 的隨機切分
- 禁止 `StratifiedKFold` 或任何會打亂時間順序的切分

**回報**：訓練/測試各幾筆、各自的時間範圍、各自的 Fail 數量與比例。
如果測試集的 Fail 少於 20 筆，回報這個事實並提醒我指標會很不穩。

---

## 階段 2：清理

**四個步驟全部包成一個 sklearn transformer，`fit` 只接觸訓練集。**

依序執行，每步回報砍掉幾欄、剩幾欄：

1. 移除常數欄（在訓練集上 `nunique() <= 1`）
2. 移除缺失率 > 50% 的欄（缺失率從訓練集算）
3. 移除相關係數 > 0.95 的成對欄位：保留缺失率較低的那一欄，缺失率相同時保留欄號較小的（相關矩陣從訓練集算，理由見 README）
4. 剩餘缺失用**訓練集的中位數**補值

**所有決定（砍哪些欄、用哪個中位數）在 `fit` 時從訓練集記錄下來，
`transform` 時原封不動套用到測試集。測試集不重新計算任何統計量。**

必須用 `sklearn.pipeline.Pipeline` 把這個 transformer 和模型串起來，
讓 TimeSeriesSplit 的每一折都正確地只 fit 訓練部分。

禁止在 Pipeline 之外做任何 `fillna`、`StandardScaler.fit`、或欄位篩選。

---

## 階段 3：模型

### IsolationForest（非監督）
- 訓練時不使用標籤
- `contamination` 設為 **0.07，作為領域先驗的超參數**，README 要註明
  這個值來自對不良率的先驗認知，不是從測試集標籤估計出來的
- 如果你認為有更誠實的設定方式，提出來討論

### XGBoost（監督）
- `scale_pos_weight` = 該折**訓練集**的 負樣本數 / 正樣本數，不是從全體算
- `max_depth = 3`
- `n_estimators` 用早停決定。**早停的驗證集必須從訓練集尾端再切一塊出來
  （例如訓練集的最後 20%），絕對不可以用測試集**
- 禁止深度學習
- 禁止 SMOTE 或任何過採樣

---

## 階段 4：評估

- **主要指標：PR-AUC（average precision）**
- 次要：固定精確率下的召回率、confusion matrix、ROC-AUC
- **禁止把 accuracy 當作評估指標或在結論中強調它**
- **隨機基準線不得寫死數字**：用該資料集實際的正樣本率計算——訓練集、測試集、CV 每一折都各自算、分別報告
- 所有指標報 TimeSeriesSplit 的平均 ± 標準差，不報單次結果

### 閾值選擇
- 不使用預設 0.5
- **閾值必須從訓練集（或訓練集內的驗證折）的 PR 曲線挑選，
  再原封不動套用到測試集。禁止在測試集上挑閾值。**
- README 說明選擇依據（例如「在精確率 20% 的條件下最大化召回」）

---

## 階段 5：SHAP

- `shap.TreeExplainer`，背景資料集用**訓練集**
- 單筆輸出必須包含 `base_value`、`risk_score`、以及 top-5 的
  `feature` / `shap_value` / `actual_value`

---

## 階段 6：FastAPI

- 三個端點：`GET /health`、`POST /predict`、`POST /explain`
- Pydantic schema，輸入驗證失敗回 422 並說明原因
- 模型在服務啟動時載入一次
- 每次請求記錄 log：批次識別、風險分數、耗時
- 附 Dockerfile

### 測試（只寫這三個，不要更多）
1. `test_no_time_leakage`：訓練集最晚時間 < 測試集最早時間
2. `test_imputation_uses_train_only`：補值中位數等於訓練集中位數，且不等於全體中位數
3. `test_api_rejects_wrong_shape`：欄位數錯誤時回 422

---

## 階段 7（選配）：LLM agent 層

只有在階段 1–6 全部完成後才開始。

- Ollama，模型 `gemma4:e4b`（原生支援 tool calling）
- 兩個工具，都只是呼叫本專案的 FastAPI：
  - `get_risk(lot_index)` → `POST /predict`
  - `get_contributors(lot_index)` → `POST /explain`
- 互動模式：**使用者提問驅動**，LLM 自主決定呼叫順序，最後產出文字報告
- LLM 不做任何計算或異常判斷，只做工具調度與結果翻譯
- **實測 20 次，把 tool calling 的成功率記在 README**
- 失敗率超過三成就降級為固定流程（程式碼依序呼叫兩個 API，
  只把最終 JSON 交給 LLM 寫報告），並在 README 記錄這個決定與理由

---

## 禁止事項

- 不要調參追高分數。這個資料集訊號本來就弱，PR-AUC 落在 0.15–0.30 是正常的
- **如果出現 ROC-AUC > 0.9 或 PR-AUC > 0.5，先停下來檢查資料洩漏，不要當成成功**
- 不要做前端網頁
- 不要加雲端部署
- 不要 fine-tune 任何模型
- 不要增加本文件以外的模組
- 不要註解 pandas 基本用法，只註解「為什麼這樣選」

## 程式風格

- Python 3.10+，type hints
- 每個函式一件事，不超過 50 行
- 隨機種子固定為 42
- 不要用全域變數傳遞狀態

---

## Repo 結構

```
secom-yield-detection/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── data/raw/uci-secom.csv     # 不進 git
├── notebooks/01_eda.ipynb
├── src/
│   ├── data.py        # 載入、時間切分
│   ├── features.py    # 清理 transformer
│   ├── train.py       # 訓練與評估
│   ├── explain.py     # SHAP
│   ├── api.py         # FastAPI
│   └── agent.py       # 階段 7
├── tests/
├── models/                    # 不進 git
└── reports/figures/
```