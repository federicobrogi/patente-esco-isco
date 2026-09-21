# Task — analisi finale patent–ESCO lato producer

## Obiettivo generale

Definire e valutare una pipeline riproducibile che, partendo dal testo di un brevetto, restituisca le professioni ESCO lato **producer** maggiormente associate alla tecnologia brevettata.

L'unità di ranking deve essere la professione ESCO (`esco_occupation_uri`), non la singola skill. Le skill e le knowledge servono a descrivere semanticamente la professione e, dopo il ranking, a spiegare il risultato.

L'analisi finale comprende due famiglie sperimentali separate:

1. effetto della dimensione del modello di embedding;
2. effetto della gestione e del troncamento dei testi lunghi nei modelli piccoli.

Le due famiglie non devono essere mescolate in un'unica conclusione causale.

## Baseline consolidata

### Descriptor del brevetto

Configurazione principale:

```text
title + ". " + abstract
```

Ablation già prevista:

```text
title + ". " + abstract + ". " + first independent claim
```

I campi mancanti devono diventare stringhe vuote, mai `NaN` o `None` testuali.

### Descriptor professionale ESCO

Costruire un solo descriptor per ogni `esco_occupation_uri`:

```text
Job title: {esco_occupation_label}.
Essential skills and knowledge: {essential skill_label_1}; ...; {essential skill_label_n}.
```

Regole:

- usare soltanto relazioni ESCO con `is_essential == True`;
- deduplicare per `esco_occupation_uri + skill_uri`;
- ordinare alfabeticamente le skill label;
- unire skill e knowledge senza pesi differenti;
- conservare `skill_type` nei dati, ma non usarlo nel descriptor o nello score;
- usare l'URI come identificatore primario, perché codici locali uguali possono riferirsi a URI diversi;
- non scegliere una professione a posteriori partendo da una skill condivisa da più professioni.

Sulla versione locale corrente sono attesi circa **3.039 descriptor professionali ESCO**.

### Ranking e spiegazione

- Calcolare embedding separati per brevetto e descriptor ESCO: i due testi non devono essere concatenati tra loro.
- Applicare la formattazione query/document ufficiale del singolo modello.
- Normalizzare L2 gli embedding e usare similarità coseno.
- Conservare TOP10 grezzo e TOP1–TOP3 nell'output di confronto.
- Calcolare il margine `TOP1 score - TOP2 score`.
- Calcolare la `skill determinante` soltanto dentro la professione TOP1, dopo il ranking.
- La skill determinante è diagnostica e non può modificare lo score della professione.
- Non usare reranker nella configurazione principale; un eventuale reranker deve essere una sperimentazione separata.

## Dataset e disegno di valutazione

### Fase di sviluppo

Usare i 30 brevetti già analizzati e annotati per verificare codice, memoria, tempi e comparabilità con le run precedenti.

Il campione non può determinare da solo la pipeline definitiva: 22 riferimenti su 30 appartengono alla professione `artificial intelligence engineer`, quindi la baseline maggioritaria è 22/30.

### Estensione a 50 brevetti annotati

Integrare i 30 casi esistenti con 20 nuovi brevetti. I nuovi casi devono aumentare contemporaneamente difficoltà e copertura, senza essere scelti perché sbagliati da un solo modello.

Selezione consigliata:

- **12 casi difficili**, individuati senza usare l'etichetta corretta;
- **8 casi di copertura**, estratti in modo stratificato da domini CPC e lunghezze poco rappresentati nei primi 30.

Costruire prima un insieme candidato più ampio e assegnare a ogni brevetto indicatori osservabili prima dell'annotazione:

- forte disaccordo tra modelli o tra TOP1;
- margine TOP1–TOP2 basso;
- instabilità del ranking tra `title_abstract` e `title_abstract_first_claim`;
- superamento del limite di token di almeno un modello piccolo;
- molti chunk o bassa quota di testo conservato;
- pluralità di codici CPC o CPC appartenenti a domini differenti;
- assenza di termini occupazionali espliciti nel titolo e nell'abstract;
- bassa similarità massima rispetto a tutti i descriptor ESCO;
- presenza di professioni candidate appartenenti a gruppi ESCO/ISCO differenti.

La regola di selezione e le soglie devono essere fissate prima dell'annotazione. Non usare l'ESCO manualmente atteso, né la correzione a posteriori degli errori, per scegliere i 20 casi.

Per evitare un challenge set non rappresentativo, riportare sempre i risultati in tre forme:

1. sui 30 casi originari;
2. sui 20 casi aggiuntivi, distinguendo `difficult` e `coverage`;
3. sul totale dei 50 casi.

Annotare i 20 nuovi casi senza mostrare agli annotatori i risultati dei modelli. Prevedere due annotazioni indipendenti, una successiva adjudication e i valori `ambiguous`/`multiple acceptable occupations` quando una sola professione non è difendibile.

### Fase finale

Usare il dataset annotato di 50 brevetti come benchmark comparativo dell'esperimento. Se lo stesso benchmark viene consultato ripetutamente per scegliere modelli e iperparametri, considerarlo un development set e prevedere successivamente un test set esterno non osservato.

Il dataset di 50 casi deve:

- contenere più professioni ESCO lato producer;
- evitare che una sola professione domini la metrica;
- includere brevetti sotto e sopra i limiti di token;
- conservare almeno un'annotazione professionale di riferimento per brevetto;
- indicare eventuali casi ambigui o con più professioni plausibili;
- essere congelato prima del confronto definitivo dei modelli;
- conservare la provenienza di ogni riga: `original`, `difficult` oppure `coverage`.

## Famiglia A — effetto della dimensione del modello

### Domanda di ricerca

A parità di input, descriptor e procedura di ranking, l'aumento del numero di parametri migliora l'associazione brevetto–professione ESCO?

### Confronto principale within-family

Usare:

1. `Qwen/Qwen3-Embedding-0.6B`;
2. `Qwen/Qwen3-Embedding-4B`;
3. `Qwen/Qwen3-Embedding-8B`.

Questo è il confronto principale per stimare l'effetto della scala, perché mantiene il più possibile costanti famiglia, obiettivo di training e modalità d'uso.

### Controlli cross-family grandi

Usare almeno:

1. `Alibaba-NLP/gte-Qwen2-7B-instruct`;
2. `BAAI/bge-multilingual-gemma2` (9B).

Controllo facoltativo, soltanto se brevetti e descriptor sono interamente in inglese:

3. `intfloat/e5-mistral-7b-instruct`.

I confronti cross-family verificano se il risultato dipende dalla sola scala oppure dalla famiglia del modello. Non devono essere interpretati come una stima pura dell'effetto dei parametri, perché cambiano architettura, dati di training, pooling e istruzioni.

### Condizioni controllate

Eseguire due condizioni distinte:

1. **budget comune**: stesso numero massimo di token per tutti i modelli;
2. **capacità nativa**: limite massimo ufficialmente supportato da ciascun modello.

Nella condizione a budget comune devono restare invariati:

- brevetti;
- descriptor brevettuale;
- descriptor ESCO;
- contenuto semantico di query e document;
- metodo di troncamento;
- normalizzazione e metrica;
- TOP-k restituito.

Usare comunque per ogni modello il wrapper ufficiale richiesto:

- Qwen3: istruzione sulla query brevettuale, documento ESCO senza istruzione;
- GTE-Qwen2: formato ufficiale `Instruct`/`Query` sulla query, documento senza istruzione;
- BGE Gemma2: formato query/document indicato dalla model card;
- E5-Mistral: istruzione sulla query e documento semplice.

Il testo aggiunto dal prompt deve essere registrato e incluso nel conteggio dei token effettivi.

### Ipotesi da verificare

- H1: Qwen3 8B supera Qwen3 4B e 0.6B nel TOP1/TOP3.
- H2: il miglioramento cresce soprattutto sui brevetti semanticamente ambigui.
- H3: una parte del vantaggio apparente deriva dalla maggiore finestra di contesto e non dai parametri.
- H4: Qwen3 8B mantiene il vantaggio anche nella condizione con budget di token comune.

## Famiglia B — strategie di troncamento per modelli piccoli

### Domanda di ricerca

Quanto incidono troncamento e selezione del contenuto sulla qualità dei modelli piccoli, mantenendo fisso il modello?

### Modelli iniziali

Usare almeno:

1. `BAAI/bge-m3`;
2. `intfloat/multilingual-e5-base`;
3. `Qwen/Qwen3-Embedding-0.6B`.

Facoltativamente aggiungere `google/embeddinggemma-300m` perché ha ottenuto risultati competitivi nella run precedente.

### Strategie da confrontare

1. **head-only**: conservare i primi token fino al limite;
2. **head-tail**: conservare una quota iniziale e una finale, documentando la proporzione;
3. **budget per campi**: assegnare budget separati a title, abstract e first claim, preservando sempre il titolo;
4. **chunking + max**: confrontare ogni chunk del brevetto con la professione e usare la similarità massima;
5. **chunking + mean**: mediare gli embedding dei chunk, poi normalizzare nuovamente;
6. **chunking + top-k mean**: mediare gli score dei chunk più rilevanti, con `k` fissato prima dell'esperimento;
7. **selezione di frasi**: selezionare frasi informative con un metodo deterministico e indipendente dall'etichetta ESCO corretta.

Le prime quattro strategie costituiscono il nucleo minimo dell'esperimento. Le altre devono essere aggiunte solo dopo il controllo della pipeline.

### Regole anti-bias

- Non usare l'ESCO annotato per selezionare frasi o chunk.
- Non variare contemporaneamente prompt, pooling e troncamento.
- Usare lo stesso limite di token per tutte le strategie confrontate nello stesso blocco.
- Registrare separatamente token grezzi, token inviati, quota conservata e numero di chunk.
- Distinguere chiaramente troncamento del brevetto e segmentazione del descriptor ESCO.
- Non applicare troncamenti silenziosi della libreria.

### Descriptor ESCO lunghi

La gestione ESCO resta quella consolidata:

- segmentare sulle `skill_label` intere;
- ripetere il job title in ogni segmento;
- non spezzare una skill a metà salvo che la singola label superi da sola il limite;
- calcolare un embedding per segmento;
- mediare e rinormalizzare gli embedding per ottenere un solo vettore professionale.

Questa strategia deve rimanere fissa durante l'esperimento sul troncamento dei brevetti. Una futura ablation sul lato ESCO dovrà essere dichiarata separatamente.

### Ipotesi da verificare

- H5: head-only perde informazioni distintive presenti nel first claim o nella parte finale.
- H6: il budget per campi supera il troncamento puramente posizionale.
- H7: chunking + max recupera recall ma può aumentare falsi positivi dovuti a singoli termini rumorosi.
- H8: top-k mean o mean sono più stabili del massimo sui casi difficili.
- H9: una buona gestione del testo riduce almeno in parte il divario fra modelli piccoli e grandi.

## Varianti del descriptor brevettuale

Per entrambe le famiglie conservare la variabile:

```text
patent_descriptor_variant
```

Valori minimi:

```text
title_abstract
title_abstract_first_claim
```

Nell'esperimento di troncamento aggiungere:

```text
truncation_strategy
token_budget
chunk_size
chunk_overlap
chunk_aggregation
```

Non confondere l'effetto dell'aggiunta del first claim con l'effetto del metodo usato per comprimerlo o segmentarlo.

## Metriche

Calcolare almeno:

- accuracy TOP1;
- recall TOP3;
- recall TOP10;
- Mean Reciprocal Rank;
- posizione media e mediana dell'ESCO suggerito;
- margine medio e mediano TOP1–TOP2;
- numero e percentuale di brevetti troncati;
- percentuale media di token conservati;
- stabilità del TOP1 tra descriptor e strategie;
- tempo di embedding;
- memoria GPU massima;
- throughput in brevetti/secondo.

Riportare le metriche anche separatamente per:

- brevetti non troncati;
- brevetti troncati;
- 20 casi aggiuntivi e relativi sottogruppi `difficult`/`coverage`;
- classi ESCO sufficientemente rappresentate;
- casi in cui l'etichetta corretta compare almeno nel TOP10.

Per il campione finale calcolare intervalli di confidenza mediante bootstrap per TOP1, TOP3 e MRR. Usare test appaiati sui medesimi brevetti per confrontare configurazioni, correggendo i confronti multipli quando necessario.

## Output per ogni riga

Il CSV grezzo deve contenere almeno:

- `experiment_family`;
- `run_id`;
- `patent_id`;
- `patent_title`;
- `model_id`;
- `model_revision`;
- `parameter_count`;
- `patent_descriptor_variant`;
- `truncation_strategy`;
- `token_budget`;
- `patent_raw_tokens`;
- `patent_input_tokens`;
- `patent_was_truncated`;
- `retained_token_ratio`;
- `chunk_count`;
- `chunk_aggregation`;
- `query_prompt_template`;
- `document_prompt_template`;
- `esco_descriptor_version`;
- per TOP1–TOP10: `esco_occupation_uri`, codice, label e score;
- `new_top1_top2_gap`;
- `new_top1_determinant_skill`;
- `esco_code_suggested`;
- `esco_profession_suggested`;
- rank dell'ESCO suggerito;
- durata e memoria GPU della run.

Le annotazioni manuali esistenti devono essere preservate e mai rigenerate automaticamente.

## Artefatti di riepilogo

Produrre:

1. CSV grezzo TOP10 per ciascuna famiglia;
2. file unico long-format per il confronto complessivo;
3. tabella metriche per modello, dimensione, descriptor e strategia;
4. matrice appaiata brevetto × configurazione;
5. tabella dedicata ai 20 nuovi casi e ai due criteri di campionamento;
6. grafici di performance rispetto a numero di parametri e token conservati;
7. grafico accuratezza/recall rispetto a tempo e memoria GPU;
8. JSON o YAML con configurazione completa e revisioni dei checkpoint;
9. log di tokenizzazione, segmentazione, troncamento ed errori;
10. aggiornamento di `EXPERIMENTS_OCCUPATION_DESCRIPTOR.md` con risultati e decisioni.

## Modifiche richieste alla pipeline

Estendere `src/occupation_descriptor_pipeline.py` e la relativa configurazione senza alterare il comportamento delle run storiche.

La pipeline deve:

- registrare un catalogo dichiarativo dei modelli;
- supportare Qwen3 8B e i controlli grandi;
- scegliere automaticamente wrapper, pooling e dtype corretti;
- rendere configurabili le strategie di troncamento;
- impedire il troncamento implicito non registrato;
- salvare token effettivi dopo l'applicazione dei prompt;
- riutilizzare cache soltanto quando modello, revisione, prompt, descriptor e strategia coincidono;
- eseguire un modello per processo e liberare la VRAM fra le run;
- consentire ripresa dopo interruzione;
- produrre manifest e checksum degli input;
- mantenere compatibile la run esistente dei 30 brevetti e consentire il benchmark esteso di 50.

## Controlli di qualità

- Nessun duplicato sulla chiave completa della configurazione sperimentale.
- Esattamente un ranking per brevetto e configurazione.
- TOP10 composto da URI professionali distinti.
- Nessun descriptor professionale vuoto.
- Nessuno score mancante o non finito.
- Conteggi di token ottenuti dal tokenizer del modello effettivamente usato.
- Prompt inclusi nel conteggio.
- Nessuna alterazione delle annotazioni manuali.
- Seed deterministici dove applicabili.
- Revisioni di codice, modello e dataset registrate.
- Smoke test locale o su A100 sui 30 casi prima della run completa.

## Sequenza operativa

1. Congelare input, annotazioni e versione ESCO.
2. Implementare catalogo dei nuovi modelli e registrazione metadati.
3. Implementare strategie di troncamento con test unitari sui token.
4. Verificare la parità con una run storica già nota.
5. Eseguire smoke test sui 30 brevetti.
6. Costruire il pool candidato e selezionare i 20 brevetti con la regola prefissata.
7. Annotare e sottoporre ad adjudication i 20 nuovi casi senza mostrare i ranking dei modelli.
8. Congelare il dataset di 50 casi e il protocollo sperimentale.
9. Selezionare un sottoinsieme ragionevole di configurazioni senza ottimizzare ripetutamente sui 50.
10. Eseguire il benchmark di 50 casi su Azure A100.
11. Calcolare metriche, intervalli di confidenza e costi computazionali.
12. Scegliere la configurazione producer finale.
13. Aggiornare codice, configurazione, documentazione e manifest degli output.

## Flusso Git/Azure

1. Implementare e verificare le modifiche nel repository locale `patents-esco-isco`.
2. Eseguire test e smoke test prima del commit.
3. Committare su GitHub con configurazioni riproducibili, senza dati o modelli pesanti.
4. Clonare o aggiornare il repository sulla A100 Azure.
5. Collegare lo storage persistente per input, cache, log e output.
6. Salvare nell'output SHA del commit Git, revisioni dei modelli e ambiente software.
7. Riportare nel repository soltanto codice, configurazioni, metriche aggregate e documentazione consentita.

## Criterio decisionale finale

La pipeline producer non deve essere scelta usando soltanto il miglior TOP1.

La decisione deve considerare congiuntamente:

- TOP1, TOP3, TOP10 e MRR;
- intervalli di confidenza e significatività dei confronti appaiati;
- robustezza sui brevetti lunghi;
- prestazioni sui casi difficili e sui casi di copertura;
- stabilità del ranking;
- coerenza semantica degli errori;
- copertura delle professioni producer;
- memoria, velocità e costo operativo;
- riproducibilità e condizioni di licenza.

Il risultato conclusivo deve distinguere esplicitamente:

1. quanto miglioramento è attribuibile alla **dimensione del modello**;
2. quanto è attribuibile alla **gestione del testo lungo**;
3. quale combinazione offre il miglior compromesso per l'esecuzione sull'intero corpus di brevetti.
