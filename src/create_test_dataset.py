# src/data/preprocessing.py

import pandas as pd
from workalendar.asia import SouthKorea
from datetime import datetime

def create_holiday_features(start_date, end_date):
    """
    공휴일 정보를 포함한 데이터프레임 생성
    """
    cal = SouthKorea()
    date_range = pd.date_range(start=start_date, end=end_date)
    
    weekdays = date_range.to_series().dt.day_name()
    holidays = [cal.is_working_day(date) for date in date_range]
    holiday_labels = [1 if not is_working_day else 0 for is_working_day in holidays]
    
    df = pd.DataFrame({
        'DATE': date_range,
        'week': weekdays,
        'holiday': holiday_labels
    }).reset_index(drop=True)
    
    df['DATE'] = df['DATE'].dt.strftime('%Y-%m-%d')
    return df

def merge_weather_data(df, weather_df):
    """
    날씨 데이터 병합
    """
    weather_df = weather_df.rename(columns={'일시':'DATETIME'})
    df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'].astype(str).str.zfill(2) + ':00').dt.strftime('%Y-%m-%d %H:%M')
    return pd.merge(df, weather_df, on='DATETIME', how='left')

def merge_bus_and_gid_data(df, bus_df, gid_df):
    """
    버스 정류장 및 GID 데이터 병합
    """
    df = pd.merge(df, bus_df, on='gid')
    df = pd.merge(df, gid_df, on='gid', how='left')
    df = df.rename(columns={'count': 'bus_count'})
    return df

def prepare_final_dataset(df):
    """
    최종 데이터셋 준비
    """
    columns = ['gid', 'DATE', 'TIME', 'ALIGHT_DEMAND', 'week', 'holiday', 
              'bus_count', 'floor_area_ratio', 'old_population', 
              'total_population', 'working_population', 'total_building', 
              'DATETIME', '기온(°C)', 'building_to_land_ratio']
    return df[columns]

def main():
    """
    전체 데이터 전처리 파이프라인
    """
    # 데이터 로드
    train_df = pd.read_csv('path/to/test_data_modified.csv')
    weather_df = pd.read_csv('path/to/weather_data.csv', encoding='euc-kr')
    bus_df = pd.read_csv('path/to/gid_data.csv')
    gid_df = pd.read_csv('path/to/bus_stop_data.csv')
    
    # 공휴일 정보 생성
    holiday_df = create_holiday_features(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31)
    )
    
    # 데이터 병합
    df = pd.merge(train_df, holiday_df, on='DATE', how='inner')
    df = merge_weather_data(df, weather_df)
    df = merge_bus_and_gid_data(df, bus_df, gid_df)
    
    # 최종 데이터셋 준비
    final_df = prepare_final_dataset(df)
    
    # 저장
    final_df.to_csv('processed_data.csv', index=False)
    
if __name__ == "__main__":
    main()
