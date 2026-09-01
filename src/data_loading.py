"""Caricamento del corpus brevetti (file .parquet canonico)."""

import pandas as pd


def load_patents(cfg: dict) -> tuple[pd.DataFrame, str]:
    """Ritorna il DataFrame brevetti con una colonna di testo unificata
    (title + patent_text) pronta per l'embedding, e il nome di quella
    colonna."""
    patents_cfg = cfg["patents"]
    patents = pd.read_parquet(patents_cfg["path"])

    text_col = "_matching_text"
    patents[text_col] = (
        patents[patents_cfg["title_col"]].fillna("") + ". " +
        patents[patents_cfg["text_col"]].fillna("")
    )
    return patents, text_col
