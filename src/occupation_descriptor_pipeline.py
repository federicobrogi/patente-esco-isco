import argparse
import csv
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer


MODELS = {
    "bge_m3": {"id": "BAAI/bge-m3", "style": "plain", "trust": False, "dtype": "float16", "batch": 64},
    "e5_base": {"id": "intfloat/multilingual-e5-base", "style": "e5", "trust": False, "dtype": "float16", "batch": 128},
    "e5_large": {"id": "intfloat/multilingual-e5-large", "style": "e5", "trust": False, "dtype": "float16", "batch": 96},
    "gte_multilingual": {"id": "Alibaba-NLP/gte-multilingual-base", "style": "plain", "trust": True, "dtype": "float32", "batch": 16},
    "qwen3_embedding_0_6b": {"id": "Qwen/Qwen3-Embedding-0.6B", "style": "qwen", "trust": True, "dtype": "bfloat16", "batch": 32},
    "labse": {"id": "sentence-transformers/LaBSE", "style": "plain", "trust": False, "dtype": "float16", "batch": 128},
    "distiluse_multilingual": {"id": "sentence-transformers/distiluse-base-multilingual-cased-v1", "style": "plain", "trust": False, "dtype": "float16", "batch": 128},
    "minilm_multilingual": {"id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "style": "plain", "trust": False, "dtype": "float16", "batch": 128},
    "mpnet_multilingual": {"id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "style": "plain", "trust": False, "dtype": "float16", "batch": 96},
    "embeddinggemma_300m": {"id": "google/embeddinggemma-300m", "style": "gemma", "trust": False, "dtype": "bfloat16", "batch": 64},
    "qwen3_embedding_4b": {"id": "Qwen/Qwen3-Embedding-4B", "style": "qwen", "trust": True, "dtype": "bfloat16", "batch": 12},
}

QWEN_INSTRUCTION = (
    "Instruct: Given a patent description, retrieve the ESCO occupation whose "
    "essential skills and knowledge are most relevant to the patented technology.\n"
    "Query: "
)


def normalize(values):
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def patent_input(text, style):
    if style == "e5":
        return "query: " + text
    if style == "qwen":
        return QWEN_INSTRUCTION + text
    if style == "gemma":
        return "task: search result | query: " + text
    return text


def esco_input(text, title, style):
    if style == "e5":
        return "passage: " + text
    if style == "gemma":
        return f"title: {title} | text: {text}"
    return text


def skill_input(text, style):
    if style == "e5":
        return "passage: " + text
    if style == "gemma":
        return "title: none | text: " + text
    return text


def chunk_occupation(model, title, labels, style, max_tokens):
    lead = f"Job title: {title}. Essential skills and knowledge: "
    chunks, current = [], []
    for label in labels:
        candidate = lead + "; ".join(current + [label]) + "."
        formatted = esco_input(candidate, title, style)
        token_count = len(model.tokenizer.encode(formatted, add_special_tokens=True))
        if current and token_count > max_tokens:
            chunks.append(lead + "; ".join(current) + ".")
            current = [label]
        else:
            current.append(label)
    if current:
        chunks.append(lead + "; ".join(current) + ".")
    return chunks


def encode(model, texts, batch_size, show_progress=True):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def model_revision(model):
    candidates = [
        getattr(model, "model_card_data", None),
        getattr(model, "_model_card_vars", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("model_revision", "revision", "commit_hash"):
                if candidate.get(key):
                    return str(candidate[key])
    first = model._first_module()
    config = getattr(getattr(first, "auto_model", None), "config", None)
    return str(getattr(config, "_commit_hash", "") or "")


def top_indexes(similarities, n=10):
    n = min(n, similarities.shape[1])
    candidates = np.argpartition(-similarities, n - 1, axis=1)[:, :n]
    scores = np.take_along_axis(similarities, candidates, axis=1)
    return np.take_along_axis(candidates, np.argsort(-scores, axis=1), axis=1)


def run_model(model_key, spec, grouped, patents, out_dir, patent_batch_size=512,
              resume=True, overwrite=False):
    print(f"\n=== {model_key}: {spec['id']} ===", flush=True)
    model = SentenceTransformer(
        spec["id"], device="cuda", trust_remote_code=spec["trust"]
    )
    if spec["dtype"] == "bfloat16":
        model.bfloat16()
    elif spec["dtype"] == "float16":
        model.half()

    max_tokens = int(model.max_seq_length)
    all_chunks, chunk_owner = [], []
    for occ_i, occ in enumerate(grouped):
        chunks = chunk_occupation(model, occ["label"], occ["skills"], spec["style"], max_tokens)
        all_chunks.extend(esco_input(chunk, occ["label"], spec["style"]) for chunk in chunks)
        chunk_owner.extend([occ_i] * len(chunks))

    chunk_emb = encode(model, all_chunks, spec["batch"])
    occ_emb = np.zeros((len(grouped), chunk_emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(grouped), dtype=np.int32)
    for index, owner in enumerate(chunk_owner):
        occ_emb[owner] += chunk_emb[index]
        counts[owner] += 1
    occ_emb /= counts[:, None]
    occ_emb = normalize(occ_emb)

    variants = {
        "title_abstract": (
            patents["title"].fillna("").astype(str)
            + ". "
            + patents["abstract"].fillna("").astype(str)
        ),
        "title_abstract_first_claim": (
            patents["title"].fillna("").astype(str)
            + ". "
            + patents["abstract"].fillna("").astype(str)
            + ". "
            + patents["first_claim"].fillna("").astype(str)
        ),
    }

    truncation_rows = []
    revision = model_revision(model)
    all_skill_labels = sorted({s for occ in grouped for s in occ["skills"]}, key=str.casefold)
    skill_embeddings = encode(
        model, [skill_input(s, spec["style"]) for s in all_skill_labels], spec["batch"]
    )
    skill_index = {label: i for i, label in enumerate(all_skill_labels)}

    for variant_name, raw_series in variants.items():
        final_path = out_dir / f"{model_key}_{variant_name}_top10.csv"
        partial_path = final_path.with_suffix(".partial.csv")
        if overwrite:
            partial_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
        if final_path.exists() and resume:
            print(f"SKIP completo: {final_path.name}", flush=True)
            continue
        completed = 0
        if resume and partial_path.exists():
            completed = max(sum(1 for _ in open(partial_path, encoding="utf-8")) - 1, 0)
            print(f"RESUME {variant_name} dalla riga {completed}", flush=True)

        for start in range(completed, len(patents), patent_batch_size):
            stop = min(start + patent_batch_size, len(patents))
            batch = patents.iloc[start:stop].reset_index(drop=True)
            texts = raw_series.iloc[start:stop].tolist()
            formatted = [patent_input(text, spec["style"]) for text in texts]
            token_counts = [len(model.tokenizer.encode(t, add_special_tokens=True)) for t in formatted]
            patent_emb = encode(model, formatted, min(spec["batch"], 32), show_progress=False)
            similarities = patent_emb @ occ_emb.T
            order = top_indexes(similarities, 10)
            rows = []
            for i, patent in batch.iterrows():
                row = {
                "patent_id": patent["id"],
                "patent_title": patent["title"],
                "modello": model_key,
                "model_id": spec["id"],
                "model_revision": revision,
                "patent_descriptor_variant": variant_name,
                "patent_input_tokens": token_counts[i],
                "patent_was_truncated": token_counts[i] > max_tokens,
            }
                for rank in range(order.shape[1]):
                    oi = int(order[i, rank])
                    occ = grouped[oi]
                    row[f"new_top{rank + 1}_esco_code"] = occ["code"]
                    row[f"new_top{rank + 1}_esco_uri"] = occ["uri"]
                    row[f"new_top{rank + 1}_esco_profession"] = occ["label"]
                    row[f"new_top{rank + 1}_score"] = float(similarities[i, oi])
                top_occ = grouped[int(order[i, 0])]
                indexes = [skill_index[label] for label in top_occ["skills"]]
                skill_scores = patent_emb[i] @ skill_embeddings[indexes].T
                row["new_top1_determinant_skill"] = top_occ["skills"][int(np.argmax(skill_scores))]
                row["new_top1_top2_gap"] = row["new_top1_score"] - row["new_top2_score"]
                rows.append(row)
                truncation_rows.append({
                "patent_id": patent["id"], "modello": model_key,
                "patent_descriptor_variant": variant_name,
                "tokens": token_counts[i], "max_tokens": max_tokens,
                "truncated": token_counts[i] > max_tokens,
                })
            pd.DataFrame(rows).to_csv(
                partial_path, mode="a", header=not partial_path.exists(), index=False
            )
            print(f"{model_key}/{variant_name}: {stop}/{len(patents)}", flush=True)
            del patent_emb, similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        partial_path.replace(final_path)

    combined_path = out_dir / f"{model_key}_two_variants_top10.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as target:
        writer = None
        for variant_name in variants:
            source_path = out_dir / f"{model_key}_{variant_name}_top10.csv"
            if not source_path.exists():
                continue
            with source_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    pd.DataFrame(truncation_rows).to_csv(out_dir / f"{model_key}_truncation.csv", index=False)
    summary = {
        "model": model_key,
        "model_id": spec["id"],
        "model_revision": revision,
        "dtype": spec["dtype"],
        "max_seq_length": max_tokens,
        "embedding_dimension": int(occ_emb.shape[1]),
        "occupations": len(grouped),
        "chunks": len(all_chunks),
        "occupations_chunked": int(sum(count > 1 for count in counts)),
        "max_chunks": int(counts.max()),
        "patent_inputs_truncated": int(sum(item["truncated"] for item in truncation_rows)),
    }
    del model, chunk_emb, occ_emb, skill_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esco", required=True)
    parser.add_argument("--patents", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--abstract-col", default="abstract")
    parser.add_argument("--first-claim-col", default="first_claim")
    parser.add_argument("--patent-batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    esco = pd.read_csv(args.esco, sep="\t", encoding="cp1252")
    esco = esco[esco["is_essential"] == True].drop_duplicates(
        ["esco_occupation_uri", "skill_uri"]
    )
    grouped = []
    for (uri, code, label), part in esco.groupby(
        ["esco_occupation_uri", "esco_occupation_code", "esco_occupation_label"], sort=True
    ):
        labels = sorted(set(part["skill_label"].dropna().astype(str)), key=str.casefold)
        grouped.append({"uri": uri, "code": str(code), "label": label, "skills": labels})

    patent_path = Path(args.patents)
    patents = pd.read_parquet(patent_path) if patent_path.suffix.lower() == ".parquet" else pd.read_csv(patent_path)
    rename = {args.id_col: "id", args.title_col: "title", args.abstract_col: "abstract", args.first_claim_col: "first_claim"}
    required_source = set(rename)
    if not required_source.issubset(patents.columns):
        raise ValueError(f"Missing patent columns: {sorted(required_source - set(patents.columns))}")
    patents = patents.rename(columns=rename)
    required = {"id", "title", "abstract", "first_claim"}
    if not required.issubset(patents.columns):
        raise ValueError(f"Missing patent columns: {sorted(required - set(patents.columns))}")
    patents = patents.drop_duplicates("id").reset_index(drop=True)
    if args.limit:
        patents = patents.head(args.limit)

    summaries, errors = [], []
    for model_key in args.models:
        try:
            summary = run_model(model_key, MODELS[model_key], grouped, patents, out_dir,
                                args.patent_batch_size, not args.no_resume, args.overwrite)
            summaries.append(summary)
        except Exception as exc:
            errors.append({"model": model_key, "error": repr(exc)})
            print(f"ERROR {model_key}: {exc!r}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metadata = {
        "environment": {
            "python": platform.python_version(),
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "summaries": summaries,
        "errors": errors,
    }
    metadata_name = "_".join(args.models) + "_run_metadata.json"
    (out_dir / metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
