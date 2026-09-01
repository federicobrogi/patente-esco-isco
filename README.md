# patent-isco-matching

Matching semantico brevetti <-> codici ISCO tramite skill/knowledge ESCO,
al posto delle sole descrizioni ISCO a 3/4 cifre (troppo generiche rispetto
al registro tecnico di un abstract di brevetto).

## Struttura

```
.
├── config.yaml              # tutti i parametri della pipeline
├── run.py                   # entry point
├── requirements.txt
├── data/
│   └── patents/
│       └── patents_en_for_isco.parquet   # <-- da copiare qui (non versionato)
├── classification/
│   └── esco/
│       └── esco_isco_competences_knowledge_dataset_v1_2_1.csv   # <-- idem
├── src/
│   ├── config.py             # caricamento/validazione config
│   ├── skill_library.py      # libreria skill ESCO per codice ISCO + pesi IDF
│   ├── embeddings.py         # provider HuggingFace / Azure OpenAI + caching
│   ├── data_loading.py       # caricamento brevetti (.parquet)
│   ├── matching.py           # cosine similarity + aggregazione per ISCO
│   └── diagnostics.py        # metriche avg_max_score / avg_topk_gap
├── cache_embeddings/          # embedding delle skill, cachati (non versionato)
└── output/                    # scores, diagnostica, confronto modelli (non versionato)
```

## Setup

```bash
pip install -r requirements.txt
```

Copiare i due file dati nelle rispettive cartelle (`data/patents/`,
`classification/esco/`) — non sono versionati nel repo (vedi `.gitignore`).

## Esecuzione

```bash
python run.py --config config.yaml
```

Per confrontare piu' modelli sullo stesso corpus, senza editare il config:

```bash
python run.py --config config.yaml --model e5_large
python run.py --config config.yaml --model bge_m3
python run.py --config config.yaml --model mpnet_multilingual
```

Ogni run scrive in `output/`:
- `{modello}_patent_isco_scores.csv` — matrice brevetto x codice ISCO
- `{modello}_patent_isco_diagnostics.csv` — max_score e topk_gap per brevetto
- `model_comparison_summary.csv` — una riga per run, per confrontare i modelli

## Note GPU (VM Azure T4 / A100)

- `embedding.device: "cuda"` nel config; fallback automatico su CPU se non
  disponibile.
- `embedding.fp16: true` consigliato su A100; su T4 funziona ma il guadagno
  e' minore.
- `batch_size`: 32-64 su T4 (16GB VRAM), fino a 128-256 su A100.

## Note sui modelli e5

`intfloat/multilingual-e5-base` e `-large` richiedono i prefissi
`"query: "` / `"passage: "` per performare correttamente: i brevetti sono
trattati come query, le skill ESCO come passage. Gestito automaticamente
dal codice quando `prefix_style: "e5"` e' impostato nel config per quel
modello; per gli altri modelli il prefisso non viene applicato.
