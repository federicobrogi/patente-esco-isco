# Task — pipeline sperimentale patent–ESCO con descriptor professionali

## Obiettivo

Replicare in modo riproducibile l'esperimento di associazione semantica tra brevetti e professioni ESCO, confrontando tutti i modelli selezionati e due varianti del descriptor brevettuale.

La pipeline deve produrre un ranking diretto delle professioni ESCO. Non deve prima selezionare una skill e poi scegliere arbitrariamente una delle professioni collegate.

## Perimetro iniziale

- Usare gli stessi 30 brevetti già annotati e analizzati.
- Usare la classificazione ESCO locale `esco_isco_competences_knowledge_dataset_v1_2_1`.
- Usare una professione ESCO per ogni `esco_occupation_uri`.
- Numero atteso sulla versione corrente del dataset: **3.039 professioni/URI**.
- Conservare l'ESCO suggerito manualmente come riferimento di valutazione.
- Non modificare le annotazioni manuali esistenti.

## Modelli da eseguire

1. `BAAI/bge-m3`
2. `intfloat/multilingual-e5-base`
3. `intfloat/multilingual-e5-large`
4. `Alibaba-NLP/gte-multilingual-base`
5. `Qwen/Qwen3-Embedding-0.6B`
6. `sentence-transformers/LaBSE`
7. `sentence-transformers/distiluse-base-multilingual-cased-v1`
8. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
9. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
10. `google/embeddinggemma-300m`
11. `Qwen/Qwen3-Embedding-4B`

Il modello esclusivamente italiano non deve essere eseguito.

Registrare per ogni modello:

- `model_id` completo;
- revisione/commit del modello;
- versione di `sentence-transformers`, `transformers`, PyTorch e CUDA;
- lunghezza massima effettiva;
- pooling impiegato;
- dimensione dell'embedding;
- uso di `trust_remote_code`;
- normalizzazione applicata.

## Varianti del descriptor brevettuale

Eseguire **entrambi** i descriptor per tutti gli undici modelli.

### Variante A

```text
title + ". " + abstract
```

Codice della variante:

```text
title_abstract
```

### Variante B

```text
title + ". " + abstract + ". " + first_claim
```

Codice della variante:

```text
title_abstract_first_claim
```

Il `first_claim` deve essere il primo claim indipendente. Campi mancanti devono diventare stringhe vuote senza generare i valori testuali `NaN` o `None`.

## Costruzione dei descriptor ESCO

1. Filtrare le relazioni ESCO con `is_essential == True`.
2. Deduplicare per coppia `esco_occupation_uri + skill_uri`.
3. Raggruppare per:

   - `esco_occupation_uri`;
   - `esco_occupation_code`;
   - `esco_occupation_label`.

4. Riunire tutte le `skill_label` essenziali della professione.
5. Ordinare alfabeticamente le label per rendere il processo deterministico.
6. Non attribuire pesi differenti a skill e knowledge.

Formato:

```text
Job title: {esco_occupation_label}. Essential skills and knowledge: {skill_label_1}; {skill_label_2}; ... .
```

La distinzione `skill/knowledge` deve essere conservata nei dati sorgente per analisi future, ma non deve entrare nel descriptor o nel punteggio.

## Gestione dei testi lunghi

- Non applicare troncamento silenzioso ai descriptor ESCO.
- Se un descriptor supera il limite del modello, dividerlo per `skill_label` intere.
- Ogni segmento deve ripetere il job title.
- Calcolare l'embedding di ogni segmento.
- Mediare gli embedding dei segmenti appartenenti alla stessa professione.
- Normalizzare nuovamente il vettore medio.
- Ottenere sempre un solo embedding finale per `esco_occupation_uri`.
- Registrare numero di professioni segmentate, numero complessivo di segmenti e massimo numero di segmenti per professione.

Applicare il limite del modello anche al descriptor brevettuale e registrare i brevetti eventualmente troncati. Non dividere automaticamente i brevetti senza documentare un'ulteriore variante sperimentale.

## Formattazione specifica per modello

| Modello | Descriptor brevetto | Descriptor ESCO | Altre impostazioni |
|---|---|---|---|
| BGE-M3 | testo della variante senza prefisso | testo puro | normalizzazione L2; nessuna istruzione |
| E5-base | `query: {patent_descriptor}` | `passage: {esco_descriptor}` | normalizzazione L2 |
| E5-large | `query: {patent_descriptor}` | `passage: {esco_descriptor}` | normalizzazione L2 |
| GTE multilingual | testo della variante senza prefisso | testo puro | `trust_remote_code=True`; pooling previsto dal modello; normalizzazione L2 |
| Qwen3 Embedding | istruzione inglese + query brevettuale | testo puro senza istruzione | pooling previsto dal modello; normalizzazione L2 |
| LaBSE | testo della variante senza prefisso | testo puro | usare il pooling CLS configurato dal modello; normalizzazione L2 |
| DistilUSE multilingual | testo della variante senza prefisso | testo puro | pooling configurato da Sentence Transformers; normalizzazione L2 |
| MiniLM multilingual | testo della variante senza prefisso | testo puro | pooling configurato da Sentence Transformers; normalizzazione L2 |
| MPNet multilingual | testo della variante senza prefisso | testo puro | pooling configurato da Sentence Transformers; normalizzazione L2 |
| EmbeddingGemma 300M | `task: search result \| query: {patent_descriptor}` | `title: {esco_occupation_label} \| text: {esco_descriptor}` | usare `encode_query`/`encode_document`; dimensione 768; `bfloat16` o `float32`, mai `float16`; normalizzazione L2 |
| Qwen3 Embedding 4B | stessa istruzione inglese usata per Qwen 0.6B + query brevettuale | testo puro senza istruzione | pooling previsto dal modello; BF16; normalizzazione L2 |

Istruzione Qwen da applicare **soltanto al brevetto**, sia per il checkpoint 0.6B sia per il checkpoint 4B:

```text
Instruct: Given a patent description, retrieve the ESCO occupation whose essential skills and knowledge are most relevant to the patented technology.
Query: {patent_descriptor}
```

Non applicare indiscriminatamente `query:` o `passage:` ai modelli che non li richiedono.

Riferimenti ufficiali:

- BGE-M3: <https://huggingface.co/BAAI/bge-m3>
- Multilingual E5: <https://huggingface.co/intfloat/multilingual-e5-base>
- GTE multilingual: <https://huggingface.co/Alibaba-NLP/gte-multilingual-base>
- Qwen3 Embedding: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- LaBSE: <https://huggingface.co/sentence-transformers/LaBSE>
- Sentence Transformers multilingual MPNet: <https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2>
- EmbeddingGemma: <https://huggingface.co/google/embeddinggemma-300m>
- Qwen3 Embedding 4B: <https://huggingface.co/Qwen/Qwen3-Embedding-4B>

`google/embeddinggemma-300m` è un modello gated: prima dell'esecuzione occorre accettare la licenza Google su Hugging Face e verificare che il token disponibile sulla macchina Azure possa scaricare il checkpoint. Le sue attivazioni non supportano `float16`.

## Calcolo del ranking

Per ciascuna combinazione di:

```text
30 brevetti × 11 modelli × 2 varianti descriptor
```

produrre **660 righe**.

Per ogni riga:

1. calcolare la similarità coseno tra embedding del brevetto ed embedding delle 3.039 professioni;
2. ordinare le professioni per punteggio decrescente;
3. conservare TOP10 nell'output grezzo;
4. mostrare TOP1, TOP2 e TOP3 nel file finale;
5. calcolare il margine `TOP1 score − TOP2 score`.

Il ranking deve avvenire per `esco_occupation_uri`, non per il solo codice locale, perché più URI possono condividere lo stesso codice.

## Skill determinante

Soltanto dopo aver selezionato la professione TOP1:

1. considerare le skill essenziali collegate alla professione TOP1;
2. confrontare individualmente ogni `skill_label` con il descriptor del brevetto usando lo stesso modello e la stessa formattazione pertinente;
3. scegliere la skill con similarità maggiore;
4. salvarla nella colonna `new_top1_determinant_skill`.

La skill determinante è diagnostica. Non deve contribuire al punteggio professionale e non deve modificare il ranking.

## Struttura dell'output grezzo

Produrre un CSV TOP10 con almeno:

- `patent_id`;
- `patent_title`;
- `model_id`;
- `model_revision`;
- `patent_descriptor_variant`;
- `patent_input_tokens`;
- `patent_was_truncated`;
- per ciascuna posizione 1–10: codice, URI, professione e score;
- `new_top1_determinant_skill`;
- `new_top1_top2_gap`.

## Aggiornamento del file finale

Aggiornare:

```text
outputs/bge_m3_full_prevalidation/occupation_descriptor_v1/
confronto_modelli_descriptor_professione_essential_top3.xlsx
```

Requisiti:

- un solo foglio;
- prima riga composta esclusivamente dalle intestazioni;
- struttura verticale, una riga per brevetto–modello–variante;
- colonna `patent_descriptor_variant` obbligatoria anche sulle righe già presenti;
- mantenere tutte le colonne esistenti;
- aggiungere URI e risultati dei nuovi modelli, se non già presenti;
- mantenere `esco_code_suggested` ed `esco_profession_suggested`;
- mantenere `ANNOTAZIONE su TOP1` per le righe originali BGE-M3;
- lasciare vuota `ANNOTAZIONE su TOP1` nelle nuove righe;
- non generare automaticamente nuove annotazioni manuali;
- non aggiungere colonne arbitrarie del tipo `uguale/diverso`;
- rendere filtrabili almeno modello e variante del descriptor.

Le righe BGE-M3 già calcolate possono essere riutilizzate soltanto se input, modello, revisione, pooling, normalizzazione e descriptor coincidono esattamente con la configurazione documentata. In caso contrario devono essere ricalcolate.

## Valutazione

Usare l'ESCO suggerito manualmente come riferimento e calcolare, per ciascun modello e variante:

- accuratezza TOP1;
- recall TOP3;
- recall TOP10;
- numero di TOP1 cambiati tra le due varianti;
- numero di casi migliorati dal first claim;
- numero di casi peggiorati dal first claim;
- margine medio e mediano TOP1–TOP2;
- posizione media dell'ESCO suggerito quando presente nel TOP10.

Produrre inoltre una tabella specifica per i sette casi anomali:

- `US-7089218-B1`;
- `US-10417563-B1`;
- `US-8346692-B2`;
- `US-8001067-B2`;
- `US-6604094-B1`;
- `US-7778946-B2`;
- `US-7493295-B2`.

Per questi casi mostrare TOP1–TOP3, skill determinante, ESCO suggerito, posizione dell'ESCO suggerito e cambiamento prodotto dal first claim.

## Controlli di qualità

- Verificare esattamente 30 brevetti distinti.
- Verificare esattamente 11 modelli e 2 varianti.
- Verificare 660 righe complessive nel nuovo esperimento.
- Verificare assenza di duplicati su `patent_id + model_id + patent_descriptor_variant`.
- Verificare che ogni ranking contenga URI distinti.
- Segnalare codici locali duplicati associati a URI differenti senza eliminarli automaticamente.
- Verificare assenza di descriptor ESCO vuoti.
- Verificare assenza di score mancanti o non finiti.
- Verificare che le annotazioni manuali non siano state alterate.
- Salvare log, configurazione, revisioni dei modelli e riepilogo della segmentazione.
- Impostare seed deterministici dove applicabile.

## Ablation supplementare Qwen

Oltre al disegno principale, eseguire facoltativamente entrambi i checkpoint Qwen senza istruzione sul brevetto. Questa prova deve essere marcata come ablation separata e non confusa con le 660 righe principali.

Codice consigliato:

```text
qwen_no_instruction_ablation
```

Serve a misurare sul nostro campione l'effetto reale dell'istruzione raccomandata dal produttore.

## Deliverable

1. CSV grezzo TOP10 con 660 righe principali.
2. File Excel aggiornato con TOP1–TOP3.
3. CSV o foglio di riepilogo delle metriche per modello e variante.
4. Tabella diagnostica dei sette casi anomali.
5. File JSON/YAML con configurazione completa e revisioni.
6. Aggiornamento di `EXPERIMENTS_OCCUPATION_DESCRIPTOR.md` con risultati, decisioni e percorso degli output.

## Criterio decisionale

La configurazione candidata per la pipeline definitiva non deve essere scelta soltanto per il miglior TOP1 sui 30 brevetti. Considerare congiuntamente:

- TOP1 e TOP3;
- stabilità tra le due varianti;
- margine TOP1–TOP2;
- comportamento sui sette casi anomali;
- coerenza semantica degli errori;
- replicabilità della configurazione.

Il campione di 30 brevetti è una fase di sviluppo. La configurazione vincente deve essere successivamente verificata su un campione annotato più ampio e separato.
