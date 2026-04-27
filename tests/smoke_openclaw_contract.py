#!/usr/bin/env python3
"""Smoke test for OpenClaw's CS_SYSTEM_PATH integration contract."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "lightgbm": REPO_ROOT / "models/latest_lightgbm_model.pkl",
    "crossmamba": REPO_ROOT / "models/latest_crossmamba_model.pkl",
    "tst": REPO_ROOT / "models/latest_tst_model.pkl",
}


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))

    from config import Config
    from model import EnsembleRanker
    from signal_generator import SignalGenerator

    missing = [name for name, path in REQUIRED.items() if not path.exists()]
    if missing:
        print(f"Missing required model artifacts: {', '.join(missing)}")
        return 1

    with REQUIRED["lightgbm"].open("rb") as file:
        lightgbm_data = pickle.load(file)

    feature_names = lightgbm_data.get("feature_names") or []
    if not lightgbm_data.get("models") or not feature_names:
        print("LightGBM artifact is missing trained models or feature names")
        return 1

    cfg = Config()
    generator = SignalGenerator(cfg)
    generator.model = EnsembleRanker(cfg.model)
    generator.model.models = lightgbm_data["models"]
    generator.model.feature_names = [str(feature) for feature in feature_names]
    generator.selected_features = generator.model.feature_names
    generator.initialize_risk()

    print(
        "OpenClaw contract OK: "
        f"{len(generator.model.models)} LightGBM models, "
        f"{len(generator.selected_features)} features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
