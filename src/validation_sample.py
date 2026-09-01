"""Generazione del CSV di validazione manuale: sample casuale di brevetti
con occupazione ESCO, skill di provenienza, e top-2 codici ISCO 3/4 cifre,
per DUE segnali (producer e user) affiancati sulla stessa riga.
"""

import pandas as pd


def _signal_columns(esco_top1_df, isco3_topn_df, isco4_topn_df, sample_idx, prefix: str) -> dict:
    out = {}
    esco_cols = ["esco_occupation_code", "esco_occupation_label", "esco_score"]
    skill_cols = ["top_skill_uri", "top_skill_label", "top_skill_description",
                  "top_skill_type", "reuse_level"]
    for c in esco_cols + skill_cols:
        out[f"{prefix}_{c}"] = esco_top1_df.loc[sample_idx, c].values

    for rank in (1, 2):
        for c in (f"isco3_top{rank}_code", f"isco3_top{rank}_title", f"isco3_top{rank}_score"):
            out[f"{prefix}_{c}"] = isco3_topn_df.loc[sample_idx, c].values
        for c in (f"isco4_top{rank}_code", f"isco4_top{rank}_title", f"isco4_top{rank}_score"):
            out[f"{prefix}_{c}"] = isco4_topn_df.loc[sample_idx, c].values
    return out


def build_validation_sample(
    patents_df: pd.DataFrame,
    text_col_title: str,
    text_col_body: str,
    id_col: str,
    signal_results: dict,
    n_sample: int,
    snippet_chars: int,
    random_state: int,
) -> pd.DataFrame:
    """signal_results: dict nome_segnale -> {"esco_top1": df, "isco3_topn":
    df, "isco4_topn": df} (l'output di match_patents_two_stage_multi_signal).
    Per ogni segnale, le colonne nel CSV finale sono prefissate con il nome
    del segnale (es. producer_esco_occupation_code, user_esco_score, ...),
    cosi' i due segnali sono affiancati sulla stessa riga per il confronto
    manuale."""

    sample_idx = patents_df.sample(n=min(n_sample, len(patents_df)), random_state=random_state).index

    out = pd.DataFrame(index=sample_idx)
    out["patent_id"] = patents_df.loc[sample_idx, id_col].values
    out["patent_title"] = patents_df.loc[sample_idx, text_col_title].values
    out["patent_text_snippet"] = (
        patents_df.loc[sample_idx, text_col_body].fillna("").str.slice(0, snippet_chars).values
    )

    for signal_name, dfs in signal_results.items():
        cols = _signal_columns(
            dfs["esco_top1"], dfs["isco3_topn"], dfs["isco4_topn"], sample_idx, prefix=signal_name
        )
        for k, v in cols.items():
            out[k] = v

    return out.reset_index(drop=True)
