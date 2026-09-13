# Esperimenti sul descriptor professionale ESCO

Data delle prove: 10 settembre 2026.

Questo documento conserva le due configurazioni attualmente più promettenti da integrare nella pipeline definitiva del repository. Entrambe lavorano direttamente sulle professioni ESCO, evitando la precedente selezione arbitraria di una professione a partire da una singola skill.

## Base ESCO comune

- Un descriptor per ogni professione identificata da `esco_occupation_uri`.
- Numero effettivo di professioni: **3.039 URI**. Il conteggio precedente di circa 2.990 corrispondeva ai codici locali distinti; uno stesso codice può identificare più URI/label.
- Sono incluse soltanto le relazioni con `is_essential == True`.
- Le skill sono deduplicate per coppia `esco_occupation_uri + skill_uri` e ordinate alfabeticamente.
- Skill e knowledge sono mantenute nei dati di origine, ma unite senza pesi differenti nel descriptor.
- Descriptor ESCO:

  ```text
  Job title: {esco_occupation_label}. Essential skills and knowledge: {essential skill_label...}.
  ```

- Se il descriptor supera la lunghezza massima del modello, viene diviso per skill intere in segmenti che ripetono il job title. Gli embedding dei segmenti sono mediati e normalizzati, producendo comunque un solo vettore per professione.
- Il ranking è calcolato mediante similarità coseno brevetto-professione.
- La `skill determinante` è la skill essenziale individualmente più simile al brevetto all'interno della professione TOP1. È soltanto diagnostica e **non modifica il ranking**.

## Run 1 — configurazione principale

### Descriptor del brevetto

```text
title + abstract
```

### Modelli

- BGE-M3
- multilingual E5-base
- multilingual E5-large
- LaBSE
- DistilUSE multilingual
- multilingual MiniLM
- multilingual MPNet

Il modello italiano è escluso.

### Risultati rispetto all'ESCO suggerito manualmente

| Modello | Vecchio TOP1 | Nuovo TOP1 | Nuovo TOP3 |
|---|---:|---:|---:|
| BGE-M3 | 0/30 | **10/30** | **15/30** |
| DistilUSE multilingual | 1/30 | **4/30** | **7/30** |
| E5-base | 1/30 | **0/30** | **0/30** |
| E5-large | 2/30 | **7/30** | **10/30** |
| LaBSE | 0/30 | **1/30** | **7/30** |
| MiniLM multilingual | 2/30 | **5/30** | **10/30** |
| MPNet multilingual | 3/30 | **7/30** | **13/30** |

### Evidenze principali

- BGE-M3 ottiene il miglior TOP1.
- MPNet è competitivo nel TOP3.
- E5-base peggiora e non è adatto a questa configurazione senza ulteriori verifiche.
- `US-7493295-B2` passa da *occupational therapist* ad *artificial intelligence engineer*.
- `US-10417563-B1` passa da *rail traffic controller* ad *artificial intelligence engineer*.
- Per `US-7778946-B2` la professione suggerita entra nel TOP2.
- Per `US-8346692-B2` la professione suggerita entra nel TOP3.

### Output

```text
outputs/bge_m3_full_prevalidation/occupation_descriptor_v1/
confronto_modelli_descriptor_professione_essential_top3.xlsx
```

I risultati grezzi TOP10 sono conservati in:

```text
outputs/bge_m3_full_prevalidation/all_models_occupation_descriptor_top10.csv
```

## Run 2 — ablation con first claim

### Descriptor del brevetto

```text
title + abstract + first claim
```

### Modello

Solo **BGE-M3**, sugli stessi 30 brevetti e sugli stessi 3.039 descriptor professionali ESCO.

### Confronto con la Run 1 BGE-M3

| Misura | Run 1 | Run 2 |
|---|---:|---:|
| TOP1 corretto | 10/30 | **11/30** |
| TOP3 corretto | **15/30** | 14/30 |
| TOP1 cambiati | — | 13/30 |

La Run 2 genera due nuovi TOP1 corretti, ma perde un TOP1 precedentemente corretto. Il miglioramento netto è quindi soltanto `+1/30`, accompagnato da una riduzione del TOP3 e da elevata instabilità.

Nei sette casi anomali, `US-7778946-B2` migliora da TOP2 a TOP1. `US-7493295-B2` e `US-10417563-B1` restano corretti. Gli altri casi non vengono risolti e `US-8346692-B2` peggiora nel TOP3.

## Decisione provvisoria

La configurazione da considerare come baseline candidata per la pipeline definitiva è:

```text
Patent descriptor = title + abstract
ESCO descriptor = job title + tutte le essential skill_label della professione
Model = BGE-M3
Output principale = TOP3
```

La Run 2 va conservata come ablation: mostra che aggiungere indiscriminatamente il first claim non produce un miglioramento stabile. Prima di aggiornare la pipeline definitiva, la baseline dovrà essere verificata su un campione annotato più ampio dei 30 brevetti.

## Run 3 — undici modelli e due varianti

Data: 11 settembre 2026.

Sono stati eseguiti gli undici modelli definiti in `TASK_PIPELINE_ESPERIMENTO_DESCRIPTOR_PROFESSIONALE.md` sui medesimi 30 brevetti, con entrambe le varianti `title_abstract` e `title_abstract_first_claim`. Il risultato comprende 660 righe senza duplicati.

| Modello | TOP1 T+A | TOP3 T+A | TOP1 T+A+C | TOP3 T+A+C |
|---|---:|---:|---:|---:|
| BGE-M3 | 10/30 | 15/30 | 11/30 | 14/30 |
| E5-base | 0/30 | 0/30 | 0/30 | 0/30 |
| E5-large | 7/30 | 10/30 | 3/30 | 5/30 |
| GTE multilingual | 8/30 | 17/30 | 12/30 | 19/30 |
| Qwen3 Embedding 0.6B | 5/30 | 11/30 | 7/30 | 12/30 |
| Qwen3 Embedding 4B | **25/30** | **29/30** | 24/30 | **29/30** |
| EmbeddingGemma 300M | 14/30 | 23/30 | 13/30 | 23/30 |
| LaBSE | 1/30 | 7/30 | 1/30 | 4/30 |
| DistilUSE multilingual | 4/30 | 7/30 | 3/30 | 8/30 |
| MiniLM multilingual | 5/30 | 10/30 | 5/30 | 11/30 |
| MPNet multilingual | 7/30 | 13/30 | 8/30 | 12/30 |

Legenda: `T+A` = title + abstract; `T+A+C` = title + abstract + first independent claim.

Qwen3 Embedding 4B è il risultato migliore: con title + abstract raggiunge 25/30 nel TOP1, 29/30 nel TOP3 e 30/30 nel TOP10. Il first claim non migliora questa configurazione. EmbeddingGemma è il secondo modello nel TOP1 con title + abstract. GTE beneficia maggiormente del first claim, passando da 8/30 a 12/30 nel TOP1.

Il campione è fortemente sbilanciato: 22 dei 30 riferimenti manuali sono `artificial intelligence engineer`. La baseline maggioritaria ottiene quindi 22/30 senza usare il testo. Il 25/30 di Qwen 4B supera questa baseline, ma non basta per una scelta definitiva. Occorre verificare il modello su un campione più ampio, diversificato e annotato indipendentemente.

GTE ha richiesto un ambiente isolato con `transformers==4.51.3`, perché il `5.11.0` produceva un errore nel codice remoto del modello. Questa dipendenza deve essere fissata per la riproducibilità.

Output principale:

```text
outputs/bge_m3_full_prevalidation/occupation_descriptor_v1/
smoke_11models_two_variants/
confronto_modelli_descriptor_professione_essential_top3_11modelli.xlsx
```
