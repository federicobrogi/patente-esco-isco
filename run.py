"""Entry point della pipeline: matching brevetti -> occupazione ESCO ->
codice ISCO (3/4 cifre), calcolato per due segnali (producer e user) in
un unico passaggio sul corpus brevetti, piu' generazione del CSV di
validazione manuale su un sample random di brevetti.

Uso:
    python run.py --config config.yaml
    python run.py --config config.yaml --model e5_large
        (override del modello attivo)
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.skill_library import build_skill_library_esco, compute_idf_weights
from src.embeddings import EmbeddingProvider, get_active_model_cfg
from src.data_loading import load_patents
from src.matching import match_patents_two_stage_multi_signal
from src.validation_sample import build_validation_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Percorso al file config.yaml")
    parser.add_argument("--model", default=None,
                         help="Override del modello attivo (chiave in embedding.candidates)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Costruzione libreria skill ESCO (granularita' occupazione)...")
    lib_esco = build_skill_library_esco(cfg)
    print(f"  {lib_esco['skill_uri'].nunique()} skill uniche, "
          f"{lib_esco['esco_occupation_code'].nunique()} occupazioni ESCO, "
          f"{lib_esco['isco_3digit_code'].nunique()} codici ISCO-3, "
          f"{lib_esco['isco_4digit_code'].nunique()} codici ISCO-4")

    idf_weights = None
    if cfg["idf_weighting"].get("enabled", True):
        idf_weights = compute_idf_weights(lib_esco, group_col="esco_occupation_code")

    model_key, model_cfg = get_active_model_cfg(cfg, args.model)
    print(f"Modello attivo: {model_key} ({model_cfg['provider']})")

    provider = EmbeddingProvider(
        model_cfg,
        batch_size=cfg["embedding"].get("batch_size", 64),
        normalize=cfg["embedding"].get("normalize_embeddings", True),
        device=cfg["embedding"].get("device", "cuda"),
        fp16=cfg["embedding"].get("fp16", False),
    )

    print("Caricamento brevetti...")
    patents, text_col = load_patents(cfg)

    skill_prefix, patent_prefix = "", ""
    if model_cfg.get("prefix_style") == "e5":
        patent_prefix = cfg["embedding"].get("patent_prefix_e5", "query: ")
        skill_prefix = cfg["embedding"].get("skill_prefix_e5", "passage: ")
        print(f"Modello e5: uso prefissi '{patent_prefix}' (brevetti) / '{skill_prefix}' (skill)")

    # definizione dei segnali: producer usa tutte le skill essenziali,
    # user solo quelle di tipo skill/competence (azioni, proxy delle task)
    signals_cfg = cfg.get("signals", {
        "producer": {"skill_types": None},
        "user": {"skill_types": ["skill/competence"]},
    })
    signal_definitions = {name: sc.get("skill_types") for name, sc in signals_cfg.items()}

    matching_cfg = cfg.get("matching", {})
    signal_results = match_patents_two_stage_multi_signal(
        patents, text_col, lib_esco, provider, provider.model_id,
        idf_weights,
        cache_cfg=cfg["cache"],
        signal_definitions=signal_definitions,
        skill_prefix=skill_prefix, patent_prefix=patent_prefix,
        chunk_size=matching_cfg.get("chunk_size", 5000),
        progress_every=matching_cfg.get("progress_every", 1),
        top_n=matching_cfg.get("top_n_isco", 2),
    )
    # NOTA: i DataFrame in signal_results mantengono l'indice posizionale
    # di `patents` (impostato dentro match_patents_two_stage_multi_signal),
    # NON vengono re-indicizzati su patent_id, cosi' restano allineati con
    # `patents` per build_validation_sample piu' sotto. patent_id viene
    # aggiunto solo come colonna nei CSV di output finali.

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    id_col = cfg["patents"]["id_col"]

    def _with_patent_id(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.insert(0, "patent_id", patents[id_col].values)
        return out

    summary_rows = []

    for signal_name, dfs in signal_results.items():
        esco_top1_df = dfs["esco_top1"]
        isco3_topn_df = dfs["isco3_topn"]
        isco4_topn_df = dfs["isco4_topn"]

        _with_patent_id(esco_top1_df).to_csv(
            out_dir / f"{model_key}_{signal_name}_esco_top1.csv", index=False)
        _with_patent_id(isco3_topn_df).to_csv(
            out_dir / f"{model_key}_{signal_name}_isco3_topn.csv", index=False)
        _with_patent_id(isco4_topn_df).to_csv(
            out_dir / f"{model_key}_{signal_name}_isco4_topn.csv", index=False)

        # diagnostica per segnale (proxy max_score/topk_gap su ISCO-4)
        diag = pd.DataFrame({
            "max_score": isco4_topn_df["isco4_top1_score"].values,
            "topk_gap": (isco4_topn_df["isco4_top1_score"] - isco4_topn_df["isco4_top2_score"]).values,
        })
        _with_patent_id(diag).to_csv(
            out_dir / f"{model_key}_{signal_name}_{cfg['output']['diagnostics_filename']}", index=False)

        print(f"[{signal_name}] avg_max_score={diag['max_score'].mean():.4f}  "
              f"avg_topk_gap={diag['topk_gap'].mean():.4f}")

        summary_rows.append({
            "model": model_key,
            "signal": signal_name,
            "provider": model_cfg["provider"],
            "avg_isco4_max_score": diag["max_score"].mean(),
            "avg_isco4_topk_gap": diag["topk_gap"].mean(),
            "n_patents": len(patents),
        })

    print(f"Output salvati in: {out_dir}")

    # --- CSV di validazione manuale (sample random, producer+user affiancati) ---
    val_cfg = cfg.get("validation_sample", {})
    if val_cfg.get("enabled", True):
        print("Costruzione CSV di validazione manuale (producer + user)...")
        sample_df = build_validation_sample(
            patents_df=patents,
            text_col_title=cfg["patents"]["title_col"],
            text_col_body=cfg["patents"]["text_col"],
            id_col=cfg["patents"]["id_col"],
            signal_results=signal_results,
            n_sample=val_cfg.get("n_sample", 1000),
            snippet_chars=val_cfg.get("snippet_chars", 500),
            random_state=val_cfg.get("random_state", 42),
        )
        sample_path = out_dir / f"{model_key}_{val_cfg.get('filename', 'validation_sample.csv')}"
        sample_df.to_csv(sample_path, index=False)
        print(f"CSV di validazione salvato in: {sample_path} ({len(sample_df)} righe)")

    # --- riepilogo modelli/segnali ---
    summary_path = out_dir / cfg["output"]["comparison_filename"]
    summary_df = pd.DataFrame(summary_rows)
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        summary_df = pd.concat([prev, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Riepilogo modelli/segnali aggiornato in: {summary_path}")


if __name__ == "__main__":
    main()
