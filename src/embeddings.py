"""Provider di embedding (HuggingFace / Azure OpenAI) e caching su disco.

Il resto della pipeline chiama sempre provider.encode(texts, prefix=...) e
non deve sapere quale provider e' effettivamente in uso.
"""

import hashlib
import os
from pathlib import Path

import numpy as np


class EmbeddingProvider:

    def __init__(self, model_cfg: dict, batch_size: int, normalize: bool,
                 device: str = "cuda", fp16: bool = False):
        self.provider = model_cfg["provider"]
        self.batch_size = batch_size
        self.normalize = normalize

        if self.provider == "huggingface":
            import torch
            from sentence_transformers import SentenceTransformer

            if device == "cuda" and not torch.cuda.is_available():
                print("  ATTENZIONE: cuda richiesta ma non disponibile, fallback su cpu")
                device = "cpu"

            self.model = SentenceTransformer(model_cfg["model_name"], device=device)
            if fp16 and device == "cuda":
                self.model.half()  # mixed precision: dimezza memoria, utile su T4/A100
            self.model_id = model_cfg["model_name"]
            print(f"  modello caricato su device={device}, fp16={fp16 and device == 'cuda'}")

        elif self.provider == "azure_openai":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
            self.deployment_name = model_cfg["deployment_name"]
            self.model_id = model_cfg["deployment_name"]

        else:
            raise ValueError(f"Provider non supportato: {self.provider}")

    def encode(self, texts: list, prefix: str = "") -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]

        if self.provider == "huggingface":
            emb = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
            return emb

        elif self.provider == "azure_openai":
            embs = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                resp = self.client.embeddings.create(
                    input=batch, model=self.deployment_name
                )
                embs.extend([d.embedding for d in resp.data])
            emb = np.array(embs, dtype=np.float32)
            if self.normalize:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / np.clip(norms, 1e-9, None)
            return emb


def get_active_model_cfg(cfg: dict, model_override):
    emb_cfg = cfg["embedding"]
    model_key = model_override or emb_cfg["active_model"]
    if model_key not in emb_cfg["candidates"]:
        raise ValueError(
            f"Modello '{model_key}' non presente in embedding.candidates del config."
        )
    return model_key, emb_cfg["candidates"][model_key]


def _cache_key(model_id: str, texts: list, prefix: str = "") -> str:
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(prefix.encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


def get_or_compute_embeddings(provider: EmbeddingProvider, model_id: str,
                               texts: list, cache_cfg: dict,
                               prefix: str = "", cache_prefix: str = "emb") -> np.ndarray:
    """Wrapper generico con caching, usato sia per le skill sia (opzionalmente)
    per i brevetti."""
    if not cache_cfg.get("enabled", True):
        return provider.encode(texts, prefix=prefix)

    cache_dir = Path(cache_cfg["dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_id, texts, prefix=prefix)
    cache_path = cache_dir / f"{cache_prefix}_{key}.npy"

    if cache_path.exists():
        print(f"  -> embedding letti da cache: {cache_path}")
        return np.load(cache_path)

    emb = provider.encode(texts, prefix=prefix)
    np.save(cache_path, emb)
    print(f"  -> embedding salvati in cache: {cache_path}")
    return emb
