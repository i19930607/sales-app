"""
GitHub Actions 每日自動執行
讀取 orders_clean.csv + 抄貨紀錄_all.csv 訓練模型
產生 predictions.json
"""
import pandas as pd
import numpy as np
import json, requests, base64, warnings, os
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import time as time_module

TW_HOLIDAYS = set([
    "2023-01-01","2023-01-02","2023-01-20","2023-01-21","2023-01-22",
    "2023-01-23","2023-01-24","2023-01-25","2023-01-26","2023-01-27",
    "2023-02-27","2023-02-28","2023-04-03","2023-04-04","2023-04-05",
    "2023-05-01","2023-06-22","2023-06-23","2023-09-29","2023-10-09","2023-10-10",
    "2024-01-01","2024-02-08","2024-02-09","2024-02-10","2024-02-11",
    "2024-02-12","2024-02-13","2024-02-14","2024-02-15","2024-02-16",
    "2024-02-28","2024-04-04","2024-04-05","2024-05-01","2024-06-10",
    "2024-09-17","2024-10-10",
    "2025-01-01","2025-01-27","2025-01-28","2025-01-29","2025-01-30",
    "2025-01-31","2025-02-01","2025-02-02","2025-02-03","2025-02-04",
    "2025-02-28","2025-04-03","2025-04-04","2025-05-01","2025-05-30",
    "2025-05-31","2025-10-06","2025-10-10",
    "2026-01-01","2026-01-02","2026-02-14","2026-02-15","2026-02-16",
    "2026-02-17","2026-02-18","2026-02-19","2026-02-20","2026-02-21",
    "2026-02-27","2026-02-28","2026-04-03","2026-04-04","2026-04-05",
    "2026-04-06","2026-05-01","2026-06-19","2026-06-20","2026-09-25",
    "2026-09-26","2026-10-09","2026-10-10",
])

def is_holiday(date):
    return date.strftime('%Y-%m-%d') in TW_HOLIDAYS

def is_pre_holiday(date, days=3):
    for i in range(1, days+1):
        if is_holiday(date + timedelta(days=i)): return 1
    return 0

def days_to_next_holiday(date):
    for i in range(1, 8):
        if is_holiday(date + timedelta(days=i)): return i
    return 8

def consecutive_holiday_length(date):
    if not is_pre_holiday(date): return 0
    length = 0
    d = date + timedelta(days=1)
    while is_holiday(d):
        length += 1
        d += timedelta(days=1)
    return length

REGION_COORDS = {
    'taipei':     (25.048, 121.531),
    'new_taipei': (25.012, 121.465),
    'taoyuan':    (24.993, 121.301),
    'hsinchu':    (24.807, 120.968),
    'miaoli':     (24.560, 120.820),
}
STORE_REGIONS = {
    'S001':'taipei','S002':'new_taipei','S003':'taipei','S004':'new_taipei',
    'S005':'new_taipei','S006':'new_taipei','S007':'new_taipei','S008':'new_taipei',
    'S009':'new_taipei','S010':'new_taipei','S011':'new_taipei','S012':'new_taipei',
    'S013':'new_taipei','S014':'new_taipei','S015':'new_taipei','S016':'new_taipei',
    'S017':'new_taipei','S018':'new_taipei','S019':'new_taipei','S020':'new_taipei',
    'S021':'new_taipei','S022':'new_taipei','S023':'new_taipei','S024':'new_taipei',
    'S025':'new_taipei','S026':'new_taipei','S027':'new_taipei','S028':'new_taipei',
    'S029':'new_taipei','S030':'new_taipei','S031':'taoyuan','S032':'taoyuan',
    'S033':'taoyuan','S034':'taoyuan','S035':'taoyuan','S036':'taoyuan',
    'S037':'taoyuan','S038':'taoyuan','S039':'taoyuan','S040':'taoyuan',
    'S041':'taoyuan','S042':'taoyuan','S043':'taoyuan','S044':'taoyuan',
    'S045':'taoyuan','S046':'taoyuan','S047':'taoyuan','S048':'taoyuan',
    'S049':'hsinchu','S050':'hsinchu','S051':'hsinchu','S052':'hsinchu',
    'S053':'hsinchu','S054':'miaoli','S055':'miaoli','S056':'miaoli',
}

# 載入資料
df = pd.read_csv('orders_clean.csv')
df['order_date'] = pd.to_datetime(df['order_date'])
df = df[df['qty'] > 0].copy()
print(f"✓ 歷史訂單：{len(df):,} 筆")

# 合併抄貨紀錄
if os.path.exists('抄貨紀錄_all.csv'):
    df_rec = pd.read_csv('抄貨紀錄_all.csv')
    df_rec['order_date'] = pd.to_datetime(df_rec['order_date'])
    df_rec = df_rec[df_rec['qty'] > 0].copy()
    df_rec['_key'] = df_rec['store_id']+df_rec['sku_id'].astype(str)+df_rec['order_date'].astype(str)
    df['_key'] = df['store_id']+df['sku_id'].astype(str)+df['order_date'].astype(str)
    df_rec = df_rec[~df_rec['_key'].isin(df['_key'])]
    df = pd.concat([df, df_rec[df.columns.intersection(df_rec.columns)]], ignore_index=True)
    df = df.drop(columns=['_key'], errors='ignore')
    print(f"✓ 合併抄貨紀錄：共 {len(df):,} 筆")

# 天氣
def fetch_weather(lat, lon, start, end):
    url = (f"https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
           f"&timezone=Asia/Taipei")
    d = requests.get(url, timeout=30).json().get('daily', {})
    if not d: return pd.DataFrame()
    return pd.DataFrame({'date': pd.to_datetime(d['time']),
                         'temp_max': d['temperature_2m_max'],
                         'temp_min': d['temperature_2m_min'],
                         'rain_mm':  d['precipitation_sum']})

start_date = df['order_date'].min().strftime('%Y-%m-%d')
end_date   = df['order_date'].max().strftime('%Y-%m-%d')
wframes = []
for region, (lat, lon) in REGION_COORDS.items():
    try:
        w = fetch_weather(lat, lon, start_date, end_date)
        w['region'] = region
        wframes.append(w)
        print(f"✓ 天氣 {region}")
    except Exception as e:
        print(f"✗ 天氣 {region}: {e}")
    time_module.sleep(1)

weather_all = pd.concat(wframes, ignore_index=True)

# 特徵
df['region']    = df['store_id'].map(STORE_REGIONS)
df['date_only'] = df['order_date'].dt.normalize()
df = df.merge(weather_all[['date','region','temp_max','temp_min','rain_mm']],
              left_on=['date_only','region'], right_on=['date','region'], how='left')
df['temp_max'] = df['temp_max'].fillna(28)
df['temp_min'] = df['temp_min'].fillna(22)
df['rain_mm']  = df['rain_mm'].fillna(0)
df['month']        = df['order_date'].dt.month
df['day_of_week']  = df['order_date'].dt.dayofweek
df['is_month_end'] = (df['order_date'].dt.day >= 25).astype(int)
df['is_summer']    = df['month'].isin([5,6,7,8]).astype(int)
df['is_winter']    = df['month'].isin([11,12,1,2]).astype(int)
df['quarter']      = df['order_date'].dt.quarter
df['is_holiday']      = df['order_date'].apply(is_holiday).astype(int)
df['is_pre_holiday']  = df['order_date'].apply(is_pre_holiday).astype(int)
df['days_to_holiday'] = df['order_date'].apply(days_to_next_holiday)
df['holiday_length']  = df['order_date'].apply(consecutive_holiday_length)
df['is_drink']  = df['sku_name'].str.contains('水|飲料|茶|涼茶|可爾必思|汽水', na=False).astype(int)
df['is_beer']   = df['sku_name'].str.contains('啤酒', na=False).astype(int)
df['is_liquor'] = df['sku_name'].str.contains('威士忌|燒酒|高粱|清露|蒙面', na=False).astype(int)
df['is_food']   = df['sku_name'].str.contains('麵|一條根|鮪魚|鯖魚', na=False).astype(int)
df = df.sort_values(['store_id','sku_id','order_date'])
global_mean = df['qty'].mean()
df['hist_avg_90d']  = (df.groupby(['store_id','sku_id'])['qty']
    .transform(lambda x: x.shift(1).rolling(90, min_periods=3).mean())).fillna(global_mean)
df['hist_avg_180d'] = (df.groupby(['store_id','sku_id'])['qty']
    .transform(lambda x: x.shift(1).rolling(180, min_periods=5).mean())).fillna(global_mean)

FEATURES = ['temp_max','temp_min','rain_mm','month','day_of_week','is_month_end',
            'is_summer','is_winter','quarter','is_holiday','is_pre_holiday',
            'days_to_holiday','holiday_length','is_drink','is_beer','is_liquor',
            'is_food','hist_avg_90d','hist_avg_180d']

df_model = df.dropna(subset=FEATURES+['qty']).copy()
print(f"✓ 訓練資料：{len(df_model):,} 筆")

X = df_model[FEATURES]; y = df_model['qty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42, n_jobs=-1)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"✓ MAE = {mae:.2f}")

# 預測
def fetch_forecast(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
           f"&timezone=Asia/Taipei&forecast_days=7")
    d = requests.get(url, timeout=15).json().get('daily', {})
    return {'temp_max': np.mean(d['temperature_2m_max'][:3]),
            'temp_min': np.mean(d['temperature_2m_min'][:3]),
            'rain_mm':  np.mean(d['precipitation_sum'][:3])}

forecast_dict = {}
for region, (lat, lon) in REGION_COORDS.items():
    try: forecast_dict[region] = fetch_forecast(lat, lon)
    except: forecast_dict[region] = {'temp_max':28,'temp_min':22,'rain_mm':0}

today = datetime.now()
store_skus = (df.groupby(['store_id','sku_id'])
    .agg(sku_name=('sku_name','first'), unit=('unit','first'),
         hist_avg_90d=('qty', lambda x: x.tail(10).mean()),
         hist_avg_180d=('qty','mean')).reset_index())

predictions = {}
for _, row in store_skus.iterrows():
    sid = row['store_id']
    fc = forecast_dict.get(STORE_REGIONS.get(sid,'taoyuan'), {'temp_max':28,'temp_min':22,'rain_mm':0})
    feat = {
        'temp_max': fc['temp_max'], 'temp_min': fc['temp_min'], 'rain_mm': fc['rain_mm'],
        'month': today.month, 'day_of_week': today.weekday(),
        'is_month_end': 1 if today.day >= 25 else 0,
        'is_summer': 1 if today.month in [5,6,7,8] else 0,
        'is_winter': 1 if today.month in [11,12,1,2] else 0,
        'quarter': (today.month-1)//3+1,
        'is_holiday': 1 if is_holiday(today) else 0,
        'is_pre_holiday': is_pre_holiday(today),
        'days_to_holiday': days_to_next_holiday(today),
        'holiday_length': consecutive_holiday_length(today),
        'is_drink':  1 if any(k in str(row['sku_name']) for k in ['水','飲料','茶','涼茶']) else 0,
        'is_beer':   1 if '啤酒' in str(row['sku_name']) else 0,
        'is_liquor': 1 if any(k in str(row['sku_name']) for k in ['威士忌','燒酒','高粱','清露']) else 0,
        'is_food':   1 if any(k in str(row['sku_name']) for k in ['麵','一條根','鮪魚']) else 0,
        'hist_avg_90d':  row['hist_avg_90d'] if pd.notna(row['hist_avg_90d']) else global_mean,
        'hist_avg_180d': row['hist_avg_180d'] if pd.notna(row['hist_avg_180d']) else global_mean,
    }
    pred = max(0, round(float(model.predict(pd.DataFrame([feat]))[0]), 1))
    if sid not in predictions: predictions[sid] = []
    predictions[sid].append({'sku_id': row['sku_id'], 'sku_name': row['sku_name'],
                              'unit': row['unit'], 'xgb_pred': pred})

for sid in predictions:
    predictions[sid] = sorted(predictions[sid], key=lambda x: x['xgb_pred'], reverse=True)

pre_h = is_pre_holiday(today)
h_len = consecutive_holiday_length(today)
print(f"✓ 預測完成：{len(predictions)} 家")
if pre_h:
    print(f"⚠️  連假前{days_to_next_holiday(today)}天，連假{h_len}天")

output = {'generated_at': today.isoformat(), 'model_mae': round(mae, 2),
          'is_pre_holiday': bool(pre_h), 'holiday_length': h_len,
          'predictions': predictions}
with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)
print("✓ predictions.json 儲存完成")
