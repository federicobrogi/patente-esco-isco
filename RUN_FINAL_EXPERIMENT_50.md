# Esecuzione dell'esperimento finale sui 50 brevetti

## Scopo

Eseguire due famiglie sperimentali sul benchmark congelato di 50 brevetti:

1. effetto della dimensione del modello;
2. effetto della strategia di troncamento.

Le run producono CSV TOP10 long-format compatibili con le annotazioni del workbook precedente.

## Preparazione locale

Il benchmark è già salvato in:

```text
data/patents/patents_50_annotated_benchmark.csv
```

Può essere rigenerato, senza interrogare fonti esterne, con:

```powershell
python scripts/build_benchmark_50.py
```

## Preparazione Azure

Aggiornare il repository e l'ambiente:

```bash
cd ~/projects/patente-esco-isco
git pull origin main
~/projects/.venv/bin/pip install -r requirements.txt
```

Verificare almeno `transformers>=4.51.3`, necessario per Qwen3.

## Smoke test

Eseguire un brevetto per ogni famiglia prima della run completa:

```bash
~/projects/.venv/bin/python run_final_experiment.py \
  --config final_experiment_50_config.yaml \
  --family scale \
  --models qwen3_embedding_0_6b \
  --limit 1

~/projects/.venv/bin/python run_final_experiment.py \
  --config final_experiment_50_config.yaml \
  --family truncation \
  --models bge_m3 \
  --limit 1
```

Usare `--overwrite` soltanto per sostituire esplicitamente risultati già presenti.

## Famiglia A — dimensione

Eseguire un modello per processo:

```bash
~/projects/.venv/bin/python run_final_experiment.py --family scale --models qwen3_embedding_0_6b
~/projects/.venv/bin/python run_final_experiment.py --family scale --models qwen3_embedding_4b
~/projects/.venv/bin/python run_final_experiment.py --family scale --models qwen3_embedding_8b
~/projects/.venv/bin/python run_final_experiment.py --family scale --models gte_qwen2_7b_instruct
~/projects/.venv/bin/python run_final_experiment.py --family scale --models bge_multilingual_gemma2
```

Ogni modello viene valutato con:

- budget comune di 512 token;
- capacità nativa;
- `title_abstract`;
- `title_abstract_first_claim`.

## Famiglia B — troncamento

```bash
~/projects/.venv/bin/python run_final_experiment.py --family truncation --models bge_m3
~/projects/.venv/bin/python run_final_experiment.py --family truncation --models e5_base
~/projects/.venv/bin/python run_final_experiment.py --family truncation --models qwen3_embedding_0_6b
~/projects/.venv/bin/python run_final_experiment.py --family truncation --models embeddinggemma_300m
```

Ogni modello viene valutato a 512 token con:

- `head_only`;
- `head_tail`;
- `field_budget`;
- `chunk_max`.

## Output

```text
output/final_50_scale_and_truncation/
```

File principali:

- un CSV TOP10 per configurazione;
- `all_final_experiment_results.csv`;
- `experiment_manifest.json`.

Ogni riga conserva l'etichetta manuale, `ANNOTAZIONE NEW`, token grezzi e inviati, quota conservata, troncamento, numero di chunk, prompt, revisione del modello e TOP10.

## Controlli minimi dopo la run

- 50 righe per ogni configurazione completa;
- nessun duplicato per `run_id + patent_id`;
- valori di `ANNOTAZIONE NEW` soltanto in `{0,1,2,3}`;
- esattamente 3.039 professioni ESCO caricate;
- nessun errore nel manifest;
- nessun troncamento silenzioso oltre il budget dichiarato.

