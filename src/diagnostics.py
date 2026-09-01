"""Metriche diagnostiche sul risultato del matching.

avg_max_score  : quanto e' forte, in media, il miglior match per brevetto.
avg_topk_gap   : quanto il modello discrimina tra i primi candidati
                 (differenza tra il 1o e il k-esimo score piu' alto).
Un segnale "piatto" (entrambe basse) e' il sintomo diagnosticato in questo
progetto: si veda il riepilogo di progetto per le tre cause ipotizzate
(mismatch di registro linguistico, granularita' ISCO, brevetti ibridi).
"""

import numpy as np
import pandas as pd


def diagnostics(results: pd.DataFrame, top_k: int) -> pd.DataFrame:
    arr = results.values
    sorted_desc = -np.sort(-arr, axis=1)
    max_score = sorted_desc[:, 0]
    k_eff = min(top_k, arr.shape[1])
    topk_gap = sorted_desc[:, 0] - sorted_desc[:, k_eff - 1]
    return pd.DataFrame({"max_score": max_score, "topk_gap": topk_gap}, index=results.index)
