# Promemoria per l'articolo scientifico — associazione brevetti–professioni ESCO lato producer

Ultimo aggiornamento: 23 settembre 2026.

## Obiettivo

Sviluppare e valutare una pipeline riproducibile che associ il contenuto tecnologico di un brevetto alle professioni ESCO che potrebbero produrre o sviluppare quella tecnologia. L'unità di ranking è la **professione ESCO**, non la singola skill.

Questo documento conserva le scelte metodologiche, i risultati intermedi e soprattutto gli errori semanticamente informativi emersi durante lo sviluppo. Non è ancora il testo dell'articolo: è la base da cui ricostruire metodo, risultati, discussione e limiti.

## Evoluzione della pipeline

### 1. Primo approccio: ranking delle singole skill

Il brevetto veniva confrontato con circa 60.000 righe/descriptor derivati dalle skill ESCO. Il descriptor della skill comprendeva:

```text
skill_label + alt labels + skill_description
```

Dopo il retrieval, una professione collegata alla skill vincente veniva scelta per rappresentare il risultato.

Questo approccio ha prodotto errori forti perché:

- una skill può essere collegata a molte professioni;
- il punteggio apparteneva alla skill, non alla professione selezionata;
- la professione poteva quindi essere scelta in modo arbitrario tra più occupazioni con lo stesso score;
- parole polisemiche o comuni potevano dominare l'embedding senza conferma del settore.

Esempi guida emersi nell'analisi degli errori:

- `train` inteso dal modello come settore ferroviario invece che come addestramento di una rete neurale;
- concetti di psicologia/cognizione associati a *gambling games developer*;
- sistemi di intelligenza artificiale associati a professioni sanitarie perché una skill descriveva l'uso di programmi informatici con pazienti;
- termini biologici o comportamentali capaci di trascinare il ranking verso professioni settoriali non pertinenti.

Questi non sono semplici errori casuali: mostrano un problema strutturale di **disambiguazione skill–professione** e di mancata coerenza tra oggetto tecnico del brevetto e dominio occupazionale.

### 2. Passaggio al descriptor professionale

La soluzione adottata è costruire un solo descriptor per ogni professione ESCO:

```text
Job title: {esco_occupation_label}.
Essential skills and knowledge: {essential skill_label_1}; ...; {essential skill_label_n}.
```

Regole consolidate:

- 3.039 professioni identificate tramite `esco_occupation_uri`;
- soltanto relazioni `is_essential == True`;
- deduplicazione per `esco_occupation_uri + skill_uri`;
- skill e knowledge usate insieme, senza pesi differenti;
- `skill_type` conservato nei dati, ma escluso da descriptor e punteggio;
- skill ordinate alfabeticamente per garantire riproducibilità;
- segmentazione dei descriptor troppo lunghi per skill intere, ripetendo il job title;
- media e rinormalizzazione degli embedding dei segmenti;
- un solo vettore finale per professione.

Questa modifica elimina la selezione a posteriori di una professione tra quelle legate a una skill condivisa. La *skill determinante* viene calcolata soltanto dopo il ranking, all'interno della professione TOP1, e serve esclusivamente come spiegazione diagnostica.

### 3. Descriptor brevettuali confrontati

Sono state mantenute due varianti:

```text
title_abstract = title + abstract
title_abstract_first_claim = title + abstract + first independent claim
```

Brevetto e descriptor ESCO vengono codificati separatamente e poi confrontati tramite similarità coseno. Non vengono concatenati tra loro.

I wrapper specifici dei modelli devono essere rispettati. Per esempio, E5 usa `query:` per il brevetto e `passage:` per ESCO; i modelli che distinguono query e document ricevono i due testi tramite le rispettive funzioni o istruzioni ufficiali.

## Disegno sperimentale raggiunto

- Dataset iniziale: 30 brevetti già annotati.
- Limite riconosciuto: 22 dei 30 riferimenti appartenevano ad *artificial intelligence engineer*; la baseline maggioritaria era quindi 22/30.
- Estensione: 20 nuovi brevetti selezionati per aumentare difficoltà e copertura.
- Benchmark corrente: 50 brevetti annotati.
- Descriptor ESCO: 3.039 descriptor professionali.
- Modelli confrontati: 11.
- Varianti brevettuali: 2.
- Output conservato: TOP10 grezzo; nel workbook sono visualizzati TOP1–TOP3 e confronti aggregati.

Ogni modello produce **100 valutazioni** nel riepilogo aggregato: 50 brevetti × 2 varianti del descriptor. I conteggi aggregati non rappresentano 100 brevetti distinti.

## Modelli valutati

1. BGE-M3;
2. multilingual E5-base;
3. multilingual E5-large;
4. GTE multilingual;
5. Qwen3-Embedding-0.6B;
6. Qwen3-Embedding-4B;
7. EmbeddingGemma 300M;
8. LaBSE;
9. DistilUSE multilingual;
10. MiniLM multilingual;
11. MPNet multilingual.

## Risultati sui 50 brevetti

### Prestazioni separate per descriptor

| Modello | TOP1 senza claim | TOP1 con claim | TOP3 senza claim | TOP3 con claim |
|---|---:|---:|---:|---:|
| BGE-M3 | 32% | 30% | 46% | 44% |
| E5-base | 2% | 2% | 2% | 2% |
| E5-large | 16% | 6% | 28% | 16% |
| GTE multilingual | 28% | 36% | 50% | 54% |
| Qwen3-Embedding-0.6B | 14% | 22% | 34% | 34% |
| **Qwen3-Embedding-4B** | **72%** | **70%** | **90%** | **92%** |
| EmbeddingGemma 300M | 40% | 32% | 62% | 54% |
| LaBSE | 4% | 2% | 18% | 12% |
| DistilUSE multilingual | 10% | 8% | 16% | 18% |
| MiniLM multilingual | 16% | 16% | 32% | 32% |
| MPNet multilingual | 18% | 20% | 36% | 34% |

### Evidenze principali

- Qwen3-Embedding-4B domina nettamente il confronto: 72% TOP1 e 90% TOP3 senza claim; 70% TOP1 e 92% TOP3 con claim.
- Il first claim **non produce un miglioramento generale**. Il suo effetto dipende dal modello.
- GTE è il modello che beneficia più chiaramente del claim: TOP1 da 28% a 36% e TOP3 da 50% a 54%.
- Qwen3-0.6B migliora nel TOP1 da 14% a 22%, ma non nel TOP3.
- E5-large ed EmbeddingGemma peggiorano sensibilmente quando viene aggiunto il claim.
- Per Qwen3-4B il claim riduce leggermente il TOP1 ma aumenta leggermente il TOP3: aggiunge quindi recall senza migliorare la prima posizione.
- EmbeddingGemma 300M è il secondo risultato TOP1 senza claim (40%), interessante rispetto alla sua dimensione.
- E5-base è sostanzialmente inefficace in questa configurazione e richiede un controllo specifico di implementazione, pooling, prefissi e adeguatezza al task prima di trarre conclusioni sul modello in generale.

### Confronto tra i 30 casi originari e i 20 nuovi

Valori aggregati sulle due varianti del descriptor:

| Modello | TOP1: 30 | TOP1: 20 | TOP1: 50 | TOP3: 30 | TOP3: 20 | TOP3: 50 |
|---|---:|---:|---:|---:|---:|---:|
| BGE-M3 | 35.0% | 25.0% | 31.0% | 48.3% | 40.0% | 45.0% |
| E5-base | 0.0% | 5.0% | 2.0% | 0.0% | 5.0% | 2.0% |
| E5-large | 16.7% | 2.5% | 11.0% | 25.0% | 17.5% | 22.0% |
| GTE multilingual | 33.3% | 30.0% | 32.0% | 60.0% | 40.0% | 52.0% |
| Qwen3-Embedding-0.6B | 20.0% | 15.0% | 18.0% | 38.3% | 27.5% | 34.0% |
| **Qwen3-Embedding-4B** | **81.7%** | **55.0%** | **71.0%** | **96.7%** | **82.5%** | **91.0%** |
| EmbeddingGemma 300M | 45.0% | 22.5% | 36.0% | 76.7% | 30.0% | 58.0% |
| LaBSE | 3.3% | 2.5% | 3.0% | 18.3% | 10.0% | 15.0% |
| DistilUSE multilingual | 11.7% | 5.0% | 9.0% | 25.0% | 5.0% | 17.0% |
| MiniLM multilingual | 16.7% | 15.0% | 16.0% | 35.0% | 27.5% | 32.0% |
| MPNet multilingual | 25.0% | 10.0% | 19.0% | 41.7% | 25.0% | 35.0% |

I 20 casi nuovi risultano più difficili per quasi tutti i modelli. Il calo di Qwen3-4B da 81,7% a 55,0% nel TOP1 e di EmbeddingGemma da 45,0% a 22,5% indica che il campione iniziale sovrastimava le prestazioni generali. Tuttavia Qwen3-4B conserva un vantaggio molto ampio e porta la professione annotata entro il TOP3 nell'82,5% delle valutazioni dei nuovi casi.

## Perché gli errori sono scientificamente interessanti

Gli errori individuati consentono di distinguere almeno cinque meccanismi:

1. **Polisemia lessicale**: una parola tecnicamente corretta assume un significato settoriale sbagliato, come `train`.
2. **Skill condivisa, professione arbitraria**: il sistema riconosce una capacità plausibile ma la traduce in una professione non giustificata.
3. **Somiglianza tematica senza coerenza produttiva**: il testo tratta psicologia, salute o comportamento, ma il brevetto non è prodotto dalla professione settoriale recuperata.
4. **Rumore del first claim**: dettagli procedurali e formule giuridico-tecniche possono diluire i concetti distintivi presenti in titolo e abstract.
5. **Bias del descriptor ESCO**: descriptor lunghi o composti da molte skill possono includere termini generici che aumentano similarità spurie.

I casi errati devono essere conservati come materiale qualitativo per la sezione di error analysis dell'articolo. La loro funzione non è soltanto mostrare che il modello sbaglia, ma spiegare **quale passaggio rappresentazionale produce l'errore**.

## Interpretazione provvisoria

La sostituzione dei descriptor di skill con descriptor professionali ha corretto un difetto concettuale della pipeline: ora lo score appartiene direttamente alla professione. Questo ha risolto alcuni errori macroscopici, incluso il caso in cui l'addestramento di una rete neurale era stato interpretato come controllo ferroviario.

Il vantaggio di Qwen3-Embedding-4B non può però essere attribuito automaticamente alla sola dimensione. Possono contribuire:

- qualità e obiettivo dei dati di addestramento;
- capacità di seguire istruzioni di retrieval;
- finestra di contesto più ampia;
- tokenizzazione e pooling;
- robustezza a descriptor eterogenei e lunghi.

Per isolare l'effetto della scala è necessario un confronto within-family tra Qwen3 0.6B, 4B e 8B, a contenuto e budget di token controllati.

## Limiti da dichiarare

- Il benchmark contiene soltanto 50 brevetti.
- I 30 casi originari erano fortemente sbilanciati verso *artificial intelligence engineer*.
- I 20 nuovi casi sono più difficili e non costituiscono necessariamente un campione casuale della popolazione dei brevetti.
- Le percentuali aggregate sulle due varianti contano due valutazioni dello stesso brevetto e non osservazioni indipendenti.
- Le etichette ESCO possono essere ambigue: per alcuni brevetti più professioni sono plausibili.
- Il TOP1 manuale non esaurisce necessariamente tutte le associazioni professionalmente difendibili.
- La stessa versione ESCO, gli stessi URI e le stesse annotazioni devono essere congelati prima del benchmark definitivo.
- Le differenze osservate richiedono intervalli di confidenza e test appaiati prima di essere presentate come significative.

## Prossimi esperimenti già pianificati

### Famiglia A — dimensione del modello

- Qwen3-Embedding 0.6B, 4B e 8B;
- controlli grandi cross-family: GTE-Qwen2-7B-instruct e BGE multilingual Gemma2;
- due condizioni: budget token comune e capacità nativa;
- misure TOP1, TOP3, TOP10, MRR, tempo, throughput e memoria GPU.

### Famiglia B — gestione dei testi lunghi

Sui modelli piccoli confrontare:

- head-only;
- head-tail;
- budget separato per title, abstract e first claim;
- chunking + max;
- successivamente mean e top-k mean.

Queste due famiglie devono restare separate per non confondere effetto dei parametri ed effetto del troncamento.

## Possibile struttura dell'articolo

1. Problema e contributo: mapping semantico brevetti–professioni ESCO lato producer.
2. Limiti del retrieval basato su singole skill.
3. Costruzione dei descriptor professionali ESCO.
4. Dataset, annotazione e protocollo sperimentale.
5. Modelli, prompt, tokenizzazione e ranking.
6. Risultati sui 50 brevetti.
7. Ablation del first claim.
8. Analisi qualitativa degli errori e della polisemia.
9. Effetto della dimensione del modello.
10. Effetto del troncamento e della gestione dei testi lunghi.
11. Limiti, validità esterna e sviluppi futuri.

## Artefatti da preservare

- `EXPERIMENTS_OCCUPATION_DESCRIPTOR.md`: cronologia delle run sui 30 casi;
- `TASK_ANALISI_FINALE_PATENT_ESCO_PRODUCER.md`: protocollo sperimentale futuro;
- `TASK_PIPELINE_ESPERIMENTO_DESCRIPTOR_PROFESSIONALE.md`: specifiche della pipeline multi-modello;
- `outputs/candidate_pool_100_two_variants/all_models_two_variants_top10.csv`: ranking grezzo;
- `outputs/candidate_pool_100_two_variants/experiment_manifest.json`: ambiente e manifest;
- `outputs/candidate_pool_100_two_variants/run_all_11_models.log`: log della run;
- `outputs/candidate_pool_100_two_variants/confronto_11_modelli_50_casi_annotazione_new_grafici.xlsx`: benchmark annotato e grafici finali.

## Messaggio centrale da verificare nell'articolo

Il risultato più promettente non deriva soltanto dalla scelta di un modello più potente. Deriva dalla correzione dell'unità semantica del retrieval: confrontare il brevetto con una **professione descritta dal job title e dalle sue competenze essenziali**, anziché recuperare una singola skill e convertirla successivamente in una professione. Qwen3-Embedding-4B sfrutta questa rappresentazione meglio degli altri modelli provati, ma il vantaggio deve essere confermato su un benchmark più ampio, bilanciato e statisticamente valutato.
