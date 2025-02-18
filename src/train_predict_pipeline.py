import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_absolute_error
import joblib

def load_and_preprocess_data(data_path):
    """데이터 로드 및 전처리"""
    data = pd.read_csv(data_path)
    # 불필요한 컬럼 제거
    columns_to_drop = ['기온 QC플래그', '강수량(mm)', # ... 생략된 컬럼들]
    return data.drop(columns=columns_to_drop)

def prepare_features():
    """학습 및 테스트용 특성 정의"""
    train_features = ['gid', 'DATE', 'TIME', 'RIDE_DEMAND', "ALIGHT_DEMAND",
                     'week', 'holiday', 'bus_count', 'floor_area_ratio',
                     'old_population', 'total_population', 'working_population',
                     'total_building']
    
    test_features = ['gid', 'DATE', 'TIME', "ALIGHT_DEMAND", 'week',
                    'holiday', 'bus_count', 'floor_area_ratio', 'old_population',
                    'total_population', 'working_population', 'total_building',
                    'DATETIME', '기온(°C)', 'building_to_land_ratio']
    
    return train_features, test_features

def train_model(train_data):
    """AutoGluon 모델 학습"""
    predictor = TabularPredictor(label='RIDE_DEMAND', eval_metric='mae')
    predictor.fit(train_data)
    return predictor

def postprocess_predictions(predictions):
    """예측값 후처리 (음수 제거 및 반올림)"""
    predictions[predictions < 0] = 0
    return np.round(predictions)

def evaluate_model(true_values, predictions):
    """모델 성능 평가"""
    return mean_absolute_error(true_values, predictions)

def save_predictions(test_data, predictions, output_path):
    """최종 예측 결과 저장"""
    test_data['RIDE_DEMAND'] = predictions
    final_columns = ['gid', 'DATE', 'TIME', 'ALIGHT_DEMAND', 'RIDE_DEMAND']
    test_data[final_columns].to_csv(output_path, index=False)

def main():
    """전체 파이프라인 실행"""
    # 1. 학습 데이터 준비
    data = load_and_preprocess_data('path/to/train_data_add_gid_weather.csv')
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    
    # 2. 특성 선택
    train_features, test_features = prepare_features()
    train_data = train_set[train_features]
    
    # 3. 모델 학습
    predictor = train_model(train_data)
    
    # 4. 모델 저장
    joblib.dump(predictor, 'autogluon_model.pickle')
    
    # 5. 테스트 데이터에 대한 예측
    test_data = pd.read_csv('path/to/test1.csv')
    predictions = predictor.predict(test_data)
    
    # 6. 예측값 후처리
    processed_predictions = postprocess_predictions(predictions)
    
    # 7. 결과 저장
    save_predictions(test_data, processed_predictions, 'test_final.csv')

if __name__ == "__main__":
    main()