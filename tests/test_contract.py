from __future__ import annotations

import pickle
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_openclaw_required_model_artifacts_exist() -> None:
    required = [
        "models/latest_lightgbm_model.pkl",
        "models/latest_crossmamba_model.pkl",
        "models/latest_tst_model.pkl",
        "models/latest_model.pkl",
    ]

    for relative_path in required:
        artifact = REPO_ROOT / relative_path
        assert artifact.exists(), f"missing model artifact: {relative_path}"
        assert artifact.stat().st_size > 0, f"empty model artifact: {relative_path}"


def test_lightgbm_artifact_has_inference_contract() -> None:
    artifact = REPO_ROOT / "models/latest_lightgbm_model.pkl"

    with artifact.open("rb") as file:
        model_data = pickle.load(file)

    assert isinstance(model_data, dict)
    assert model_data.get("models"), "LightGBM artifact must contain trained models"
    assert model_data.get("feature_names"), "LightGBM artifact must contain feature names"
    assert len(model_data["feature_names"]) >= 10


def test_openclaw_import_contract() -> None:
    from config import Config
    from model import EnsembleRanker
    from signal_generator import SignalGenerator

    cfg = Config()
    generator = SignalGenerator(cfg)
    ranker = EnsembleRanker(cfg.model)

    assert generator.cfg is cfg
    assert ranker.models == []
