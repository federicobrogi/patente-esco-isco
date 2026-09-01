"""Caricamento e validazione minimale del file di configurazione YAML."""

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    required_top = ["patents", "esco", "skill_text", "idf_weighting",
                     "embedding", "cache", "aggregation", "diagnostics", "output"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Config incompleto, chiavi mancanti: {missing}")

    active = cfg["embedding"]["active_model"]
    if active not in cfg["embedding"]["candidates"]:
        raise ValueError(
            f"active_model='{active}' non presente in embedding.candidates"
        )
