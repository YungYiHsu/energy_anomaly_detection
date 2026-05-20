import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# 設定本地資料路徑
DATA_PATH = "data"

# 1. 建立計時器，取代原版的 timer
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] 耗時: {time.time() - t0:.2f} 秒")

# 2. 核心記憶體優化函數 (避免本地電腦記憶體崩潰)
def reduce_mem_usage(df, skip_cols=None, verbose=True):
    if skip_cols is None:
        skip_cols = []
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if col in skip_cols:
            continue
        col_type = df[col].dtype
        
        if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if col_type == object:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'記憶體使用量從 {start_mem:.2f} MB 降至 {end_mem:.2f} MB (減少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    return df

# 3. 處理時間戳記
def process_timestamp(df): 
    df.timestamp = pd.to_datetime(df.timestamp)
    # 轉換成自 2016-01-01 以來的小時數
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600            

# 4. 氣象資料清理與校正
def process_weather(df, fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    if fix_timestamps:
        # 各 Site 相對 GMT 的時區偏差修正
        site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)

    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # 確保補值時時間軸連續 (Train 的時間範圍在 0 ~ 8783 小時)
            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))
            site_df.site_id = site_id
            
            for col in [c for c in site_df.columns if c != "site_id"]:
                if add_na_indicators: 
                    site_df[f"had_{col}"] = ~site_df[col].isna()
                # 使用三次樣條曲線 (Spline) 內插法補值
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='spline', order=3)
                # 全空欄位則用整體中位數做安全防線
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).reset_index()

    if add_na_indicators:
        for col in df.columns:
            if df[col].isna().any(): 
                df[f"had_{col}"] = ~df[col].isna()

    return df.fillna(-1)

# 5. 製造氣象延遲與滾動特徵 (Lag features)
def add_lag_feature(df, window=3, group_cols="site_id", lag_cols=["air_temperature"]):
    rolled = df.groupby(group_cols)[lag_cols].rolling(window=window, min_periods=0, center=True)
    #print("---0---") # debug 用，確認程式跑到這裡了
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    #print("---1---")
    lag_max = rolled.quantile(0.95).reset_index().astype(np.float16)
    #print("---2---")
    lag_min = rolled.quantile(0.05).reset_index().astype(np.float16)
    #print("---3---")
    lag_std = rolled.std().reset_index().astype(np.float16)
    #print("---4---")
    """
    RuntimeWarning: overflow encountered in cast  return arr.astype(dtype, copy=True)
    出現溢位警告，推測是index轉換成float16時數值過大導致的，後續不把index加回df，故不影響。
    確認最終data數值正常，不存在inf,-inf或nan。
    """

    for col in lag_cols:
        df[f"{col}_mean_lag{window}"] = lag_mean[col]
        df[f"{col}_max_lag{window}"] = lag_max[col]
        df[f"{col}_min_lag{window}"] = lag_min[col]
        df[f"{col}_std_lag{window}"] = lag_std[col]

# 6. 製造時間與交互作用特徵
def add_features(df):
    df["hour"] = df.ts.dt.hour
    df["weekday"] = df.ts.dt.weekday
    df["month"] = df.ts.dt.month
    df["year"] = df.ts.dt.year    
    
    df["weekday_hour"] = df.weekday.astype(str) + "-" + df.hour.astype(str)
    
    # 週期性特徵轉換 (用 Cos / Sin 捕捉時間循環)
    df["hour_x"] = np.cos(2*np.pi*df.timestamp/24)
    df["hour_y"] = np.sin(2*np.pi*df.timestamp/24)
    df["month_x"] = np.cos(2*np.pi*df.timestamp/(30.4*24))
    df["month_y"] = np.sin(2*np.pi*df.timestamp/(30.4*24))
    df["weekday_x"] = np.cos(2*np.pi*df.timestamp/(7*24))
    df["weekday_y"] = np.sin(2*np.pi*df.timestamp/(7*24))
            
    df["year_built"] = df["year_built"] - 1900

    # 交叉特徵交互作用 (改為 category 型態防止記憶體爆炸)
    bm_ = df.building_id.astype(str) + "-" + df.meter.astype(str) + "-" 
    df["building_weekday_hour"] = (bm_ + df.weekday_hour).astype('category')
    df["building_weekday"]      = (bm_ + df.weekday.astype(str)).astype('category')
    df["building_month"]        = (bm_ + df.month.astype(str)).astype('category')
    df["building_hour"]         = (bm_ + df.hour.astype(str)).astype('category')    
    df["building_meter"]        = bm_.astype('category')

    # 美國國定假日特徵
    dates_range = pd.date_range(start="2015-12-31", end="2019-01-01")
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())    
    df["is_holiday"] = (df.ts.dt.date.astype("datetime64[ns]").isin(us_holidays)).astype(np.int8)   


if __name__ == "__main__":
    
    # === Step 1: 讀取資料 ===
    with timer("Loading data"):
        train = pd.read_csv(f"{DATA_PATH}/train.csv")
        building_meta = pd.read_csv(f"{DATA_PATH}/building_metadata.csv")
        train_weather = pd.read_csv(f"{DATA_PATH}/weather_train.csv")

    # === Step 2: 時間戳記轉換 ===
    with timer("Process timestamp"):
        train["ts"] = pd.to_datetime(train.timestamp)
        process_timestamp(train)
        process_timestamp(train_weather)

    # === Step 3: 氣象特徵加工 ===
    with timer("Process weather"):
        train_weather = process_weather(train_weather)


        # 加上 7 小時與 73 小時的滑動視窗特徵
        for window_size in [7, 73]:
            add_lag_feature(train_weather, window=window_size)
        

    # === Step 4: 合併資料集 ===
    with timer("Combine data"):
        full_df = pd.merge(train, building_meta, on="building_id", how="left")
        full_df = pd.merge(full_df, train_weather, on=["site_id", "timestamp"], how="left")
        
        del train, train_weather
        gc.collect()

    # === Step 5: 貼上預測目標 (Anomaly Label) ===
    with timer("Flag bad meter readings"):
        # 讀取解壓後的異常標籤檔案
        bad_readings = pd.read_csv(f"{DATA_PATH}/bad_meter_readings.csv")
        # 假設 bad_meter_readings.csv 內容為與 train.csv 同長度的 0/1 欄位
        full_df["is_bad_meter_reading"] = bad_readings.values.flatten()

    # === Step 6: 原始解決方案的 Site 0 校正 ===
    with timer("Correct site 0 meter reading"):
        full_df.loc[(full_df.site_id == 0) & (full_df.meter == 0), "meter_reading"] *= 0.2931

    # === Step 7: 製造基礎與交叉特徵 ===
    with timer("Add base features"):
        add_features(full_df)
        gc.collect()

    # === Step 8: 依照教授要求，按「建築物數量」進行 50/50 拆分 ===
    with timer("Split train/test by 50% buildings"):
        unique_buildings = full_df["building_id"].unique()
        
        # 設定隨機種子確保實驗能被重複驗證
        np.random.seed(42)
        np.random.shuffle(unique_buildings)
        
        split_idx = len(unique_buildings) // 2
        train_buildings = unique_buildings[:split_idx]
        test_buildings = unique_buildings[split_idx:]
        
        # 根據切割好的 Building ID 將資料分成兩半
        local_train = full_df[full_df["building_id"].isin(train_buildings)].copy()
        local_test = full_df[full_df["building_id"].isin(test_buildings)].copy()
        
        del full_df
        gc.collect()

    # === Step 9: 記憶體降維優化 ===
    with timer("Reduce memory usage"):
        local_train = reduce_mem_usage(local_train, skip_cols=['ts', 'timestamp'], verbose=True)
        local_test = reduce_mem_usage(local_test, skip_cols=['ts', 'timestamp'], verbose=True)

    # === Step 10: 移除多餘欄位並儲存成果 ===
    with timer("Remove unnecessary columns & Save to Pickle"):
        # 移除時間物件欄位以免影響機器學習訓練
        if "ts" in local_train.columns:
            local_train.drop(columns=["ts"], inplace=True)
        if "ts" in local_test.columns:
            local_test.drop(columns=["ts"], inplace=True)
            
        # 建立預處理成品目錄
        os.makedirs(f"{DATA_PATH}/preprocessed", exist_ok=True)
        
        local_train.to_pickle(f"{DATA_PATH}/preprocessed/local_train_features.pkl")
        local_test.to_pickle(f"{DATA_PATH}/preprocessed/local_test_features.pkl")
        
        print(f"訓練集維度: {local_train.shape}, 測試集維度: {local_test.shape}")
        print("🎉 特徵工程完成！檔案已成功儲存至 data/preprocessed/")