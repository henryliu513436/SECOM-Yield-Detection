# import pandas as pd

# df = pd.read_csv('/Users/liuheng/Desktop/secom-yield-detection/data/raw/uci-secom.csv')
# df.iloc[:5, :10]                    # 看前 5 筆、前 10 欄
# df.iloc[:, 1:-1].describe().T.head(20)   # 看各欄的統計摘要
# df.isna().mean().sort_values(ascending=False).head(20)  # 哪些欄缺最多

# print((df.iloc[:,1:-1].nunique() <= 1).sum())                          # 預期 (1567, 592)
# print(df.describe())
# print(df.isna())
# print(df.columns[:3].tolist())           # 預期 ['Time', '0', '1'] 之類
# print(df.columns[-1])                    # 預期 'Pass/Fail'
# print(df['Pass/Fail'].value_counts())    # 預期 -1 約 1463 筆、1 約 104 筆
# print(df.isna().sum().sum())             # 預期幾萬個缺失值


# import pandas as pd
# df = pd.read_csv('data/raw/uci-secom.csv')
# t = pd.to_datetime(df['Time'])

# back = t.index[t.diff() < pd.Timedelta(0)]
# for i in back:
#     print(f"index {i}: {t[i-1]}  →  {t[i]}")

import pandas as pd
df = pd.read_csv('data/raw/uci-secom.csv')

print("前 3 筆原始字串：")
print(df['Time'].head(3).tolist())
print("\nindex 61~65：")
print(df['Time'].iloc[61:66].tolist())
print("\nindex 198~202：")
print(df['Time'].iloc[198:202].tolist())

t2 = pd.to_datetime(df['Time'], dayfirst=True)
print("dayfirst=True 的倒退次數：", (t2.diff() < pd.Timedelta(0)).sum())

t3 = pd.to_datetime(df['Time'], dayfirst=False)
print("dayfirst=False 的倒退次數：", (t3.diff() < pd.Timedelta(0)).sum())