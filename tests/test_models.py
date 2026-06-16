from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import pytest

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[1]


def test_serialized_model_and_features():
    model_path = BASE_DIR / "models/model.pkl"
    features_path = BASE_DIR / "models/features.pkl"

    # 1. Pastikan file model dan fitur ter-serialisasi ada
    assert model_path.exists()
    assert features_path.exists()

    # 2. Pastikan file model dan fitur bisa di-load menggunakan pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    # 3. Validasi list fitur
    assert isinstance(features, list)
    assert len(features) == 4
    assert 'lag_1' in features
    assert 'lag_2' in features

    # 4. Uji inferensi model dengan data mock
    mock_x = pd.DataFrame([{
        'lag_1': 1000000.0,
        'lag_2': 500000.0,
        'rolling_mean_3': 750000.0,
        'growth_rate': 1.0
    }])
    
    y_pred = model.predict(mock_x[features])
    
    assert len(y_pred) == 1
    assert isinstance(y_pred[0], (float, np.float32, np.float64))
