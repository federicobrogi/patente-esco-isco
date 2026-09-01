"""Matching brevetto -> occupazione ESCO -> codice ISCO, a due stadi,
calcolato per DUE SEGNALI in un unico passaggio sul corpus brevetti:

  - producer: chi ha creato il brevetto. Usa TUTTE le skill/knowledge
    essenziali di un'occupazione ESCO (competenze possedute).
  - user: chi usera'/subira' l'impatto del brevetto. Usa solo le skill di
    tipo 'skill/competence' (formulate come azioni, es. "calibrate
    electronic instruments"), piu' vicine al concetto di "task svolta"
    nel senso di Septiandri et al. (2024) rispetto alle 'knowledge'
    (piu' dichiarative, "cosa un lavoratore deve conoscere").

I due segnali condividono lo stesso embedding dei brevetti e delle skill
(calcolato una sola volta per chunk): cambia solo QUALI colonne (skill)
vengono considerate nell'aggregazione per occupazione ESCO. Il costo
computazionale aggiuntivo per il secondo segnale e' quindi trascurabile
(solo l'aggregazione via numpy, non un secondo embedding).

Stage 1 (granularita' fine): per ogni brevetto, similarity contro le skill
eleggibili per ciascun segnale, poi aggregazione per SINGOLA occupazione
ESCO (max pesato IDF), con provenance della skill che ha prodotto il
match migliore.

Stage 2 (aggregazione): gli score per occupazione ESCO vengono aggregati
per codice ISCO 3 e 4 cifre (mapping 1:1 ESCO->ISCO verificato sui dati),
prendendo il MASSIMO tra le occupazioni ESCO sotto lo stesso codice.

Elaborazione a CHUNK sui brevetti per corpora grandi (es. 100k brevetti).
"""

import numpy as np
import pandas as pd

from .embeddings import EmbeddingProvider, get_or_compute_embeddings


# ----------------------------------------------------------------------
# MAPPE DI SUPPORTO (costruite una sola volta, riusate per ogni chunk)
# ----------------------------------------------------------------------
def _build_esco_to_skillcols(lib_esco: pd.DataFrame, skill_order: list, skill_types=None) -> dict:
    """occupazione ESCO -> indici di colonna (nell'array skill_emb) delle
    sue skill eleggibili per il segnale corrente. skill_types=None usa
    tutte le skill (segnale producer); una lista (es. ['skill/competence'])
    filtra solo quel tipo (segnale user)."""
    df = lib_esco
    if skill_types is not None:
        df = df[df["skill_type"].isin(skill_types)]

    uri_to_col = {uri: i for i, uri in enumerate(skill_order)}
    grouped = df.groupby("esco_occupation_code")["skill_uri"].apply(list)
    return {
        code: np.array([uri_to_col[u] for u in uris], dtype=np.int64)
        for code, uris in grouped.items()
    }


def _build_esco_metadata(lib_esco: pd.DataFrame) -> pd.DataFrame:
    """Una riga per occupazione ESCO, con titolo e codici/titoli ISCO
    3-4 cifre corrispondenti (mapping 1:1)."""
    return lib_esco.drop_duplicates(subset=["esco_occupation_code"])[[
        "esco_occupation_code", "esco_occupation_label",
        "isco_3digit_code", "isco_3digit_title",
        "isco_4digit_code", "isco_4digit_title",
    ]].reset_index(drop=True)


def _build_isco_to_esco_cols(esco_meta: pd.DataFrame, isco_col: str, esco_order: list) -> dict:
    """codice ISCO (3 o 4 digit) -> indici di colonna (nell'array delle
    occupazioni ESCO del segnale corrente) delle occupazioni ESCO che vi
    appartengono. esco_order e' specifico del segnale (producer/user
    possono avere universi di occupazioni leggermente diversi)."""
    code_to_row = {code: i for i, code in enumerate(esco_order)}
    meta_sub = esco_meta[esco_meta["esco_occupation_code"].isin(esco_order)]
    grouped = meta_sub.groupby(isco_col)["esco_occupation_code"].apply(list)
    return {
        isco_code: np.array([code_to_row[c] for c in esco_codes], dtype=np.int64)
        for isco_code, esco_codes in grouped.items()
    }


class _SignalView:
    """Tutto cio' che serve per aggregare UN segnale (producer o user)
    dentro il loop a chunk, precalcolato una sola volta."""

    def __init__(self, name: str, lib_esco: pd.DataFrame, skill_order: list,
                 esco_meta: pd.DataFrame, skill_types=None):
        self.name = name
        self.esco_to_cols = _build_esco_to_skillcols(lib_esco, skill_order, skill_types=skill_types)
        self.esco_order = sorted(self.esco_to_cols.keys())

        self.isco3_to_esco_cols = _build_isco_to_esco_cols(esco_meta, "isco_3digit_code", self.esco_order)
        self.isco4_to_esco_cols = _build_isco_to_esco_cols(esco_meta, "isco_4digit_code", self.esco_order)
        self.isco3_order = sorted(self.isco3_to_esco_cols.keys())
        self.isco4_order = sorted(self.isco4_to_esco_cols.keys())

        self.esco_meta_by_code = esco_meta.set_index("esco_occupation_code")
        self.isco3_meta = esco_meta.drop_duplicates("isco_3digit_code").set_index("isco_3digit_code")["isco_3digit_title"]
        self.isco4_meta = esco_meta.drop_duplicates("isco_4digit_code").set_index("isco_4digit_code")["isco_4digit_title"]

        self.esco_top1_rows = []
        self.isco3_topn_rows = []
        self.isco4_topn_rows = []


# ----------------------------------------------------------------------
# AGGREGAZIONE PER CHUNK
# ----------------------------------------------------------------------
def _aggregate_stage1_chunk(sim_weighted: np.ndarray, esco_to_cols: dict, esco_order: list):
    n_chunk = sim_weighted.shape[0]
    n_esco = len(esco_order)
    esco_scores = np.empty((n_chunk, n_esco), dtype=np.float32)
    best_skill_col = np.empty((n_chunk, n_esco), dtype=np.int64)

    for j, code in enumerate(esco_order):
        cols = esco_to_cols[code]
        sub = sim_weighted[:, cols]
        local_argmax = sub.argmax(axis=1)
        esco_scores[:, j] = sub[np.arange(n_chunk), local_argmax]
        best_skill_col[:, j] = cols[local_argmax]

    return esco_scores, best_skill_col


def _aggregate_stage2(esco_scores_chunk: np.ndarray, isco_to_esco_cols: dict, isco_order: list) -> np.ndarray:
    n_chunk = esco_scores_chunk.shape[0]
    n_isco = len(isco_order)
    out = np.empty((n_chunk, n_isco), dtype=np.float32)
    for j, code in enumerate(isco_order):
        cols = isco_to_esco_cols[code]
        out[:, j] = esco_scores_chunk[:, cols].max(axis=1)
    return out


def _top_n_per_row(scores: np.ndarray, n: int = 2):
    order = np.argsort(-scores, axis=1)[:, :n]
    top_scores = np.take_along_axis(scores, order, axis=1)
    return order, top_scores


# ----------------------------------------------------------------------
# MAIN: MATCHING MULTI-SEGNALE, UN SOLO PASSAGGIO SUI BREVETTI
# ----------------------------------------------------------------------
def match_patents_two_stage_multi_signal(
    patents_df: pd.DataFrame,
    text_col: str,
    lib_esco: pd.DataFrame,
    provider: EmbeddingProvider,
    model_id: str,
    idf_weights: pd.Series,
    cache_cfg: dict,
    signal_definitions: dict,
    skill_prefix: str = "",
    patent_prefix: str = "",
    chunk_size: int = 5000,
    progress_every: int = 1,
    top_n: int = 2,
):
    """signal_definitions: dict nome_segnale -> lista skill_type ammessi
    (None = tutte). Esempio:
        {"producer": None, "user": ["skill/competence"]}

    Ritorna un dict: nome_segnale -> {"esco_top1": df, "isco3_topn": df,
    "isco4_topn": df}.
    """
    unique_skills = lib_esco.drop_duplicates(subset=["skill_uri"])[
        ["skill_uri", "skill_text", "skill_label", "skill_description", "skill_type", "reuse_level"]
    ].reset_index(drop=True)
    skill_order = unique_skills["skill_uri"].tolist()

    print("Embedding libreria skill ESCO (condivisa tra i segnali)...")
    skill_emb = get_or_compute_embeddings(
        provider, model_id, unique_skills["skill_text"].tolist(), cache_cfg,
        prefix=skill_prefix, cache_prefix="skills_esco",
    )

    if idf_weights is not None:
        idf_vec = idf_weights.reindex(skill_order).fillna(1.0).values.astype(np.float32)
    else:
        idf_vec = np.ones(len(skill_order), dtype=np.float32)

    esco_meta = _build_esco_metadata(lib_esco)

    views = {
        name: _SignalView(name, lib_esco, skill_order, esco_meta, skill_types=skill_types)
        for name, skill_types in signal_definitions.items()
    }
    for name, v in views.items():
        print(f"  segnale '{name}': {len(v.esco_order)} occupazioni ESCO, "
              f"{len(v.isco3_order)} ISCO-3, {len(v.isco4_order)} ISCO-4")

    n_patents = len(patents_df)
    n_chunks = (n_patents + chunk_size - 1) // chunk_size
    print(f"Matching {n_patents} brevetti in {n_chunks} chunk da {chunk_size} "
          f"(segnali: {list(views.keys())})...")

    texts_all = patents_df[text_col].tolist()

    for i, start in enumerate(range(0, n_patents, chunk_size)):
        end = min(start + chunk_size, n_patents)
        chunk_texts = texts_all[start:end]

        if i % progress_every == 0:
            print(f"  chunk {i + 1}/{n_chunks} (brevetti {start}-{end})...")

        # --- parte condivisa: un solo embedding brevetti per chunk ---
        chunk_emb = provider.encode(chunk_texts, prefix=patent_prefix)
        sim = chunk_emb @ skill_emb.T
        sim *= idf_vec  # peso condiviso: proprieta' della skill, non del segnale

        # --- parte specifica per segnale: solo aggregazione, economica ---
        for name, v in views.items():
            esco_scores, best_skill_col = _aggregate_stage1_chunk(sim, v.esco_to_cols, v.esco_order)

            best_esco_idx = esco_scores.argmax(axis=1)
            for row_i in range(esco_scores.shape[0]):
                j = best_esco_idx[row_i]
                esco_code = v.esco_order[j]
                skill_col = best_skill_col[row_i, j]
                skill_row = unique_skills.iloc[skill_col]
                meta = v.esco_meta_by_code.loc[esco_code]
                v.esco_top1_rows.append({
                    "esco_occupation_code": esco_code,
                    "esco_occupation_label": meta["esco_occupation_label"],
                    "esco_score": esco_scores[row_i, j],
                    "top_skill_uri": skill_row["skill_uri"],
                    "top_skill_label": skill_row["skill_label"],
                    "top_skill_description": skill_row["skill_description"],
                    "top_skill_type": skill_row["skill_type"],
                    "reuse_level": skill_row["reuse_level"],
                })

            isco3_scores = _aggregate_stage2(esco_scores, v.isco3_to_esco_cols, v.isco3_order)
            isco4_scores = _aggregate_stage2(esco_scores, v.isco4_to_esco_cols, v.isco4_order)

            order3, top3 = _top_n_per_row(isco3_scores, n=top_n)
            order4, top4 = _top_n_per_row(isco4_scores, n=top_n)

            for row_i in range(order3.shape[0]):
                row = {}
                for rank in range(top_n):
                    code = v.isco3_order[order3[row_i, rank]]
                    row[f"isco3_top{rank+1}_code"] = code
                    row[f"isco3_top{rank+1}_title"] = v.isco3_meta.loc[code]
                    row[f"isco3_top{rank+1}_score"] = top3[row_i, rank]
                v.isco3_topn_rows.append(row)

            for row_i in range(order4.shape[0]):
                row = {}
                for rank in range(top_n):
                    code = v.isco4_order[order4[row_i, rank]]
                    row[f"isco4_top{rank+1}_code"] = code
                    row[f"isco4_top{rank+1}_title"] = v.isco4_meta.loc[code]
                    row[f"isco4_top{rank+1}_score"] = top4[row_i, rank]
                v.isco4_topn_rows.append(row)

        del chunk_emb, sim

    results = {}
    for name, v in views.items():
        esco_top1_df = pd.DataFrame(v.esco_top1_rows)
        isco3_topn_df = pd.DataFrame(v.isco3_topn_rows)
        isco4_topn_df = pd.DataFrame(v.isco4_topn_rows)
        for d in (esco_top1_df, isco3_topn_df, isco4_topn_df):
            d.index = patents_df.index
        results[name] = {
            "esco_top1": esco_top1_df,
            "isco3_topn": isco3_topn_df,
            "isco4_topn": isco4_topn_df,
        }

    return results
