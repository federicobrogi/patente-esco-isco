"""Costruzione della libreria skill/knowledge ESCO, sia a livello di
occupazione ESCO (stage 1) sia aggregata per codice ISCO (compatibilita'
con la pipeline precedente, se serve un livello unico).

Sostituisce, come testo di ancoraggio, le descrizioni sintetiche di ruolo
con le singole skill/knowledge essenziali (label + alt_labels +
description), piu' vicine per registro linguistico al testo dei brevetti.
"""

import numpy as np
import pandas as pd


def _build_skill_text(row: pd.Series, text_cfg: dict) -> str:
    parts = []
    if text_cfg.get("use_label", True):
        parts.append(str(row["skill_label"]))

    if text_cfg.get("use_alt_labels", True) and isinstance(row["skill_alt_labels"], str):
        alts = [a.strip() for a in row["skill_alt_labels"].split("\n") if a.strip()]
        max_alts = text_cfg.get("max_alt_labels", 0)
        if max_alts and max_alts > 0:
            alts = alts[:max_alts]
        parts.extend(alts)

    if text_cfg.get("use_description", True) and isinstance(row["skill_description"], str):
        desc = row["skill_description"]
        max_chars = text_cfg.get("max_description_chars", 0)
        if max_chars and max_chars > 0:
            desc = desc[:max_chars]
        parts.append(desc)

    return " / ".join(parts)


def build_skill_library_esco(cfg: dict) -> pd.DataFrame:
    """Libreria skill a granularita' di SINGOLA occupazione ESCO (stage 1
    della pipeline a due stadi). Una riga per (esco_occupation_code,
    skill_uri), con anche i codici/titoli ISCO 3 e 4 cifre corrispondenti
    (mapping 1:1 verificato sui dati) cosi' da poter aggregare a valle
    senza un secondo join.
    """
    esco_cfg = cfg["esco"]
    text_cfg = cfg["skill_text"]

    df = pd.read_csv(esco_cfg["path"], sep=esco_cfg.get("sep", ";"))

    if esco_cfg.get("essential_only", True):
        df = df[df["is_essential"] == True]

    df = df.copy()
    df["skill_text"] = df.apply(lambda r: _build_skill_text(r, text_cfg), axis=1)

    cols = [
        "esco_occupation_code", "esco_occupation_label",
        "isco_3digit_code", "isco_3digit_title",
        "isco_4digit_code", "isco_4digit_title",
        "skill_uri", "skill_text",
        "skill_label", "skill_alt_labels", "skill_description",
        "skill_type", "reuse_level",
    ]
    lib = df[cols].drop_duplicates(
        subset=["esco_occupation_code", "skill_uri"]
    ).reset_index(drop=True)

    return lib


def build_skill_library(cfg: dict) -> pd.DataFrame:
    """Libreria skill aggregata direttamente a livello di codice ISCO
    (opzione A, mantenuta per compatibilita' / confronto con la B)."""
    esco_cfg = cfg["esco"]
    text_cfg = cfg["skill_text"]

    df = pd.read_csv(esco_cfg["path"], sep=esco_cfg.get("sep", ";"))

    if esco_cfg.get("essential_only", True):
        df = df[df["is_essential"] == True]

    isco_level = esco_cfg["isco_level"]

    df = df.copy()
    df["skill_text"] = df.apply(lambda r: _build_skill_text(r, text_cfg), axis=1)

    lib = df[[isco_level, "skill_uri", "skill_text", "skill_type"]].drop_duplicates(
        subset=[isco_level, "skill_uri"]
    ).reset_index(drop=True)

    return lib


def compute_idf_weights(lib: pd.DataFrame, group_col: str) -> pd.Series:
    """Peso IDF-like: penalizza le skill condivise da molti gruppi (analogia
    con le stop-words), senza escluderle del tutto. group_col e' la chiave
    di aggregazione stage-appropriata: 'esco_occupation_code' per la
    libreria a livello ESCO, oppure il livello ISCO per quella aggregata."""
    doc_freq = lib.groupby("skill_uri")[group_col].nunique()
    n_groups = lib[group_col].nunique()
    idf = np.log((n_groups + 1) / (doc_freq + 1)) + 1
    return idf
