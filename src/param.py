from pathlib import Path


class Params:
    random_state = 42,
    assets_path = Path('./brst_assets')
    original = assets_path / 'brst_original' / 'data.csv'
    data = assets_path / 'brst_data'
    features = assets_path / 'brst_features'
    models = assets_path / 'brst_models'
    metrics = assets_path / 'brst_metrics'
