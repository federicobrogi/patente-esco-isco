"""Controlled scale and truncation experiments for the 50-patent ESCO benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
import yaml
from sentence_transformers import SentenceTransformer


RETRIEVAL_INSTRUCTION = (
    "Given a patent description, retrieve the ESCO occupation whose essential "
    "skills and knowledge are most relevant to developing the patented technology."
)

MODELS = {
    "bge_m3": dict(id="BAAI/bge-m3", family="bge", parameters=568_000_000,
                   style="plain", trust=False, dtype="float16", batch=32),
    "e5_base": dict(id="intfloat/multilingual-e5-base", family="e5", parameters=278_000_000,
                    style="e5", trust=False, dtype="float16", batch=64),
    "qwen3_embedding_0_6b": dict(id="Qwen/Qwen3-Embedding-0.6B", family="qwen3",
                    parameters=600_000_000, style="qwen", trust=True, dtype="bfloat16", batch=24),
    "qwen3_embedding_4b": dict(id="Qwen/Qwen3-Embedding-4B", family="qwen3",
                    parameters=4_000_000_000, style="qwen", trust=True, dtype="bfloat16", batch=8),
    "qwen3_embedding_8b": dict(id="Qwen/Qwen3-Embedding-8B", family="qwen3",
                    parameters=8_000_000_000, style="qwen", trust=True, dtype="bfloat16", batch=4),
    "gte_qwen2_7b_instruct": dict(id="Alibaba-NLP/gte-Qwen2-7B-instruct", family="gte_qwen2",
                    parameters=8_000_000_000, style="gte_instruct", trust=True,
                    dtype="bfloat16", batch=4),
    "bge_multilingual_gemma2": dict(id="BAAI/bge-multilingual-gemma2", family="bge_gemma2",
                    parameters=9_000_000_000, style="bge_instruct", trust=False,
                    dtype="bfloat16", batch=3),
    "embeddinggemma_300m": dict(id="google/embeddinggemma-300m", family="embeddinggemma",
                    parameters=300_000_000, style="gemma", trust=False,
                    dtype="bfloat16", batch=32),
}


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def query_text(text: str, style: str) -> str:
    if style == "e5":
        return "query: " + text
    if style == "qwen":
        return f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery: {text}"
    if style == "gte_instruct":
        return f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery: {text}"
    if style == "bge_instruct":
        return f"<instruct>{RETRIEVAL_INSTRUCTION}\n<query>{text}"
    if style == "gemma":
        return "task: search result | query: " + text
    return text


def document_text(text: str, title: str, style: str) -> str:
    if style == "e5":
        return "passage: " + text
    if style == "gemma":
        return f"title: {title} | text: {text}"
    return text


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def fit_formatted(tokenizer, raw: str, style: str, budget: int) -> tuple[str, int, bool]:
    """Apply query formatting and guarantee that the formatted input fits budget."""
    raw_ids = token_ids(tokenizer, raw)
    lo, hi = 0, min(len(raw_ids), budget)
    best = query_text("", style)
    best_n = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = query_text(decode(tokenizer, raw_ids[:mid]), style)
        size = len(tokenizer.encode(candidate, add_special_tokens=True))
        if size <= budget:
            best, best_n, lo = candidate, mid, mid + 1
        else:
            hi = mid - 1
    final_tokens = len(tokenizer.encode(best, add_special_tokens=True))
    return best, final_tokens, best_n < len(raw_ids)


def take_field(tokenizer, text: str, budget: int) -> str:
    return decode(tokenizer, token_ids(tokenizer, text)[:max(0, budget)])


def raw_variant(row, variant: str) -> str:
    pieces = [str(row.patent_title or ""), str(row.patent_abstract or "")]
    if variant == "title_abstract_first_claim":
        pieces.append(str(row.first_claim or ""))
    return ". ".join(piece for piece in pieces if piece)


def make_patent_inputs(tokenizer, row, variant: str, style: str, strategy: str,
                       budget: int, settings: dict) -> tuple[list[str], dict]:
    raw = raw_variant(row, variant)
    raw_count = len(tokenizer.encode(query_text(raw, style), add_special_tokens=True))
    if strategy == "head_only":
        formatted, used, truncated = fit_formatted(tokenizer, raw, style, budget)
        return [formatted], dict(raw_tokens=raw_count, input_tokens=used,
            retained_token_ratio=min(1.0, used / max(raw_count, 1)),
            patent_was_truncated=truncated, chunk_count=1)

    raw_ids = token_ids(tokenizer, raw)
    prompt_cost = len(tokenizer.encode(query_text("", style), add_special_tokens=True))
    payload = max(1, budget - prompt_cost)
    if strategy == "head_tail":
        head_fraction = float(settings["head_tail"]["head_fraction"])
        head_n = int(payload * head_fraction)
        tail_n = payload - head_n
        selected = raw_ids[:head_n]
        if tail_n:
            selected += raw_ids[-tail_n:]
        formatted, used, truncated = fit_formatted(tokenizer, decode(tokenizer, selected), style, budget)
        return [formatted], dict(raw_tokens=raw_count, input_tokens=used,
            retained_token_ratio=min(1.0, used / max(raw_count, 1)),
            patent_was_truncated=truncated or len(raw_ids) > payload, chunk_count=1)

    if strategy == "field_budget":
        fractions = settings["field_budget"]
        fields = [
            (str(row.patent_title or ""), float(fractions["title_fraction"])),
            (str(row.patent_abstract or ""), float(fractions["abstract_fraction"])),
        ]
        if variant == "title_abstract_first_claim":
            fields.append((str(row.first_claim or ""), float(fractions["first_claim_fraction"])))
        else:
            total = sum(frac for _, frac in fields)
            fields = [(text, frac / total) for text, frac in fields]
        selected = ". ".join(take_field(tokenizer, text, int(payload * frac)) for text, frac in fields)
        formatted, used, truncated = fit_formatted(tokenizer, selected, style, budget)
        return [formatted], dict(raw_tokens=raw_count, input_tokens=used,
            retained_token_ratio=min(1.0, used / max(raw_count, 1)),
            patent_was_truncated=truncated or len(raw_ids) > payload, chunk_count=1)

    if strategy.startswith("chunk_"):
        overlap = int(settings["chunking"]["overlap_tokens"])
        step = max(1, payload - overlap)
        chunks = [raw_ids[start:start + payload] for start in range(0, len(raw_ids), step)] or [[]]
        formatted_chunks = [fit_formatted(tokenizer, decode(tokenizer, chunk), style, budget)[0]
                            for chunk in chunks]
        sent = sum(len(tokenizer.encode(x, add_special_tokens=True)) for x in formatted_chunks)
        return formatted_chunks, dict(raw_tokens=raw_count, input_tokens=sent,
            retained_token_ratio=1.0, patent_was_truncated=len(chunks) > 1,
            chunk_count=len(chunks))
    raise ValueError(f"Unknown truncation strategy: {strategy}")


def occupation_chunks(model, title: str, labels: list[str], style: str, budget: int) -> list[str]:
    lead = f"Job title: {title}. Essential skills and knowledge: "
    chunks, current = [], []
    for label in labels:
        candidate = lead + "; ".join(current + [label]) + "."
        formatted = document_text(candidate, title, style)
        if current and len(model.tokenizer.encode(formatted, add_special_tokens=True)) > budget:
            chunks.append(document_text(lead + "; ".join(current) + ".", title, style))
            current = [label]
        else:
            current.append(label)
    chunks.append(document_text(lead + "; ".join(current) + ".", title, style))
    return chunks


def encode(model, texts: list[str], batch: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)


def top_order(scores: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(scores))
    candidates = np.argpartition(-scores, n - 1)[:n]
    return candidates[np.argsort(-scores[candidates])]


def load_esco(path: Path) -> list[dict]:
    frame = pd.read_csv(path, sep="\t", encoding="cp1252")
    frame = frame[frame["is_essential"] == True].drop_duplicates(
        ["esco_occupation_uri", "skill_uri"]
    )
    grouped = []
    for (uri, code, label), part in frame.groupby(
        ["esco_occupation_uri", "esco_occupation_code", "esco_occupation_label"], sort=True
    ):
        grouped.append(dict(uri=uri, code=str(code), label=label,
            skills=sorted(set(part["skill_label"].dropna().astype(str)), key=str.casefold)))
    return grouped


def build_occupation_embeddings(model, occupations: list[dict], spec: dict,
                                budget: int) -> tuple[np.ndarray, dict]:
    all_chunks, owners = [], []
    for owner, occupation in enumerate(occupations):
        chunks = occupation_chunks(model, occupation["label"], occupation["skills"],
                                   spec["style"], budget)
        all_chunks.extend(chunks)
        owners.extend([owner] * len(chunks))
    embedded = encode(model, all_chunks, spec["batch"])
    result = np.zeros((len(occupations), embedded.shape[1]), dtype=np.float32)
    counts = np.zeros(len(occupations), dtype=np.int32)
    for vector, owner in zip(embedded, owners):
        result[owner] += vector
        counts[owner] += 1
    result = l2(result / counts[:, None])
    return result, dict(occupation_chunks=len(all_chunks),
                        occupations_chunked=int((counts > 1).sum()), max_chunks=int(counts.max()))


def experiment_matrix(config: dict, selected_family: str) -> list[dict]:
    common = config["common"]
    combinations = []
    if selected_family in ("all", "scale"):
        section = config["scale_experiment"]
        for model in section["models"]:
            for condition in section["token_conditions"]:
                for variant in common["descriptor_variants"]:
                    combinations.append(dict(experiment_family="scale", model=model,
                        patent_descriptor_variant=variant, token_condition=condition,
                        truncation_strategy=section["truncation_strategy"]))
    if selected_family in ("all", "truncation"):
        section = config["truncation_experiment"]
        for model in section["models"]:
            for strategy in section["strategies"]:
                for variant in common["descriptor_variants"]:
                    combinations.append(dict(experiment_family="truncation", model=model,
                        patent_descriptor_variant=variant, token_condition="common_512",
                        truncation_strategy=strategy))
    return combinations


def run_model(model_key: str, combinations: list[dict], config: dict, patents: pd.DataFrame,
              occupations: list[dict], output_dir: Path, overwrite: bool) -> list[dict]:
    spec = MODELS[model_key]
    model = SentenceTransformer(spec["id"], device=config["runtime"]["device"],
                                trust_remote_code=spec["trust"])
    if spec["dtype"] == "bfloat16": model.bfloat16()
    if spec["dtype"] == "float16": model.half()
    revision = str(getattr(model._first_module().auto_model.config, "_commit_hash", "") or "")
    native_budget = int(model.max_seq_length)
    summaries = []
    occupation_cache = {}
    for combo in combinations:
        common_budget = int(config["common"]["common_token_budget"])
        budget = native_budget if combo["token_condition"] == "native" else min(common_budget, native_budget)
        run_id = "__".join([combo["experiment_family"], model_key,
            combo["token_condition"], combo["truncation_strategy"], combo["patent_descriptor_variant"]])
        destination = output_dir / f"{run_id}__top10.csv"
        if destination.exists() and not overwrite:
            print(f"SKIP {destination.name}", flush=True)
            continue
        cache_key = budget
        if cache_key not in occupation_cache:
            occupation_cache[cache_key] = build_occupation_embeddings(model, occupations, spec, budget)
        occupation_embeddings, occupation_meta = occupation_cache[cache_key]
        rows, started = [], time.perf_counter()
        for patent in patents.itertuples(index=False):
            inputs, token_meta = make_patent_inputs(model.tokenizer, patent,
                combo["patent_descriptor_variant"], spec["style"],
                combo["truncation_strategy"], budget, config["truncation_experiment"])
            vectors = encode(model, inputs, min(spec["batch"], len(inputs)))
            chunk_scores = vectors @ occupation_embeddings.T
            if combo["truncation_strategy"] == "chunk_max":
                scores = chunk_scores.max(axis=0)
            elif combo["truncation_strategy"] == "chunk_mean":
                scores = chunk_scores.mean(axis=0)
            else:
                scores = chunk_scores[0]
            order = top_order(scores, int(config["common"]["top_n"]))
            row = dict(run_id=run_id, experiment_family=combo["experiment_family"],
                patent_id=patent.patent_id, patent_title=patent.patent_title,
                patent_abstract=patent.patent_abstract, first_claim=patent.first_claim,
                modello=model_key, model_id=spec["id"], model_revision=revision,
                model_parameters=spec["parameters"],
                patent_descriptor_variant=combo["patent_descriptor_variant"],
                token_condition=combo["token_condition"], token_budget=budget,
                truncation_strategy=combo["truncation_strategy"],
                chunk_aggregation="max" if combo["truncation_strategy"] == "chunk_max" else "none",
                query_prompt_template=query_text("{patent_text}", spec["style"]),
                document_prompt_template=document_text("{esco_descriptor}", "{job_title}", spec["style"]),
                esco_code_suggested=patent.esco_code_suggested,
                esco_profession_suggested=patent.esco_profession_suggested,
                sample_origin=patent.sample_origin,
                annotation_sample_stratum=patent.annotation_sample_stratum, **token_meta)
            for rank, occupation_index in enumerate(order, 1):
                occupation = occupations[int(occupation_index)]
                row[f"new_top{rank}_esco_code"] = occupation["code"]
                row[f"new_top{rank}_esco_uri"] = occupation["uri"]
                row[f"new_top{rank}_esco_profession"] = occupation["label"]
                row[f"new_top{rank}_score"] = float(scores[occupation_index])
            row["new_top1_top2_gap"] = row["new_top1_score"] - row["new_top2_score"]
            row["ANNOTAZIONE NEW"] = next((rank for rank in (1, 2, 3)
                if str(row[f"new_top{rank}_esco_code"]) == str(row["esco_code_suggested"])), 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(destination, index=False)
        summaries.append(dict(run_id=run_id, rows=len(rows), seconds=time.perf_counter()-started,
                              token_budget=budget, native_budget=native_budget, **occupation_meta))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="final_experiment_50_config.yaml")
    parser.add_argument("--family", choices=["all", "scale", "truncation"], default="all")
    parser.add_argument("--models", nargs="*", choices=MODELS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    np.random.seed(int(config["experiment"]["seed"]))
    torch.manual_seed(int(config["experiment"]["seed"]))
    patents = pd.read_csv(root / config["patents"]["path"]).fillna("")
    if args.limit: patents = patents.head(args.limit)
    if patents["patent_id"].duplicated().any(): raise ValueError("Duplicate patent IDs")
    occupations = load_esco(root / config["esco"]["path"])
    if len(occupations) != 3039: raise ValueError(f"Expected 3039 occupations, found {len(occupations)}")
    output_dir = root / config["runtime"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations = experiment_matrix(config, args.family)
    if args.models: combinations = [item for item in combinations if item["model"] in args.models]
    manifest = dict(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(), torch=torch.__version__,
            transformers=transformers.__version__, sentence_transformers=sentence_transformers.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        patent_count=len(patents), occupation_count=len(occupations), summaries=[], errors=[])
    for model_key in dict.fromkeys(item["model"] for item in combinations):
        try:
            selected = [item for item in combinations if item["model"] == model_key]
            manifest["summaries"].extend(run_model(model_key, selected, config, patents,
                                                    occupations, output_dir, args.overwrite))
        except Exception as exc:
            manifest["errors"].append(dict(model=model_key, error=repr(exc)))
            print(f"ERROR {model_key}: {exc!r}", flush=True)
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files = sorted(output_dir.glob("*__top10.csv"))
    if files:
        pd.concat((pd.read_csv(path) for path in files), ignore_index=True).to_csv(
            output_dir / "all_final_experiment_results.csv", index=False)
    if manifest["errors"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
