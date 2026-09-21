"""Build a reproducible 100-patent candidate pool and enrich first claims.

Selection uses only patent title and abstract. It excludes the 30 reviewed
patents and combines TF-IDF topic coverage, cluster-boundary cases and long
descriptors. Claim enrichment is optional and uses the public Google Patents
HTML page identified by patent_id.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")
DEPENDENT_RE = re.compile(
    r"\b(?:of|according to|as (?:set forth|recited) in|defined in)\s+"
    r"(?:any one of\s+)?claims?\s+\d+",
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are",
    "using", "used", "use", "one", "more", "may", "can", "such", "each",
    "based", "wherein", "method", "system", "apparatus", "device", "devices",
    "comprising", "includes", "including", "provided", "providing", "first",
    "second", "plurality", "associated", "configured", "information", "data",
}


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOPWORDS]


def tfidf_matrix(texts: list[str], max_features: int = 6000) -> np.ndarray:
    documents = [tokens(text) for text in texts]
    df = Counter()
    for document in documents:
        df.update(set(document))
    vocabulary = [
        term for term, _ in sorted(df.items(), key=lambda item: (-item[1], item[0]))
        if 2 <= df[term] <= len(documents) * 0.80
    ][:max_features]
    index = {term: i for i, term in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float32)
    for row, document in enumerate(documents):
        counts = Counter(term for term in document if term in index)
        for term, count in counts.items():
            matrix[row, index[term]] = (1.0 + math.log(count)) * math.log(
                (1.0 + len(documents)) / (1.0 + df[term])
            )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.where(norms == 0, 1.0, norms)
    return matrix


def cosine_kmeans(matrix: np.ndarray, k: int, seed: int = 20260921) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
    labels = np.zeros(len(matrix), dtype=np.int32)
    for _ in range(50):
        similarities = matrix @ centers.T
        new_labels = similarities.argmax(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = matrix[labels == cluster]
            if not len(members):
                centers[cluster] = matrix[rng.integers(0, len(matrix))]
                continue
            center = members.mean(axis=0)
            norm = np.linalg.norm(center)
            centers[cluster] = center / norm if norm else center
    return labels, centers


def select_candidates(frame: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    texts = (frame["patent_title"].fillna("") + ". " + frame["patent_abstract"].fillna("")).tolist()
    matrix = tfidf_matrix(texts)
    labels, centers = cosine_kmeans(matrix, k=20)
    similarities = matrix @ centers.T
    top = np.sort(similarities, axis=1)[:, -2:]
    margins = top[:, 1] - top[:, 0]
    own_similarity = similarities[np.arange(len(frame)), labels]
    word_counts = np.array([len(tokens(text)) for text in texts])

    work = frame.reset_index(drop=True).copy()
    work["selection_cluster"] = labels
    work["cluster_similarity"] = own_similarity
    work["cluster_margin"] = margins
    work["descriptor_word_count"] = word_counts
    chosen: dict[int, str] = {}

    # Five complementary cases per cluster: two central, two boundary, one long.
    for cluster in range(20):
        indexes = np.where(labels == cluster)[0]
        central = sorted(indexes, key=lambda i: (-own_similarity[i], str(work.at[i, "patent_id"])))[:2]
        boundary = sorted(indexes, key=lambda i: (margins[i], str(work.at[i, "patent_id"])))[:2]
        longest = sorted(indexes, key=lambda i: (-word_counts[i], str(work.at[i, "patent_id"])))[:1]
        for idx in central:
            chosen.setdefault(int(idx), "coverage_central")
        for idx in boundary:
            chosen.setdefault(int(idx), "difficulty_boundary")
        for idx in longest:
            chosen.setdefault(int(idx), "difficulty_long")

    # Fill collisions deterministically while balancing underrepresented clusters.
    if len(chosen) < n:
        remaining = [i for i in range(len(work)) if i not in chosen]
        remaining.sort(key=lambda i: (margins[i], -word_counts[i], str(work.at[i, "patent_id"])))
        for idx in remaining[: n - len(chosen)]:
            chosen[idx] = "difficulty_fill"

    selected = work.loc[sorted(chosen)][:n].copy()
    selected["selection_role"] = [chosen[int(i)] for i in selected.index]
    selected = selected.sort_values(
        ["selection_cluster", "selection_role", "patent_id"], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "candidate_rank", np.arange(1, len(selected) + 1))
    return selected


def strip_tags(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", " ", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(fragment)).strip()


class ClaimParser(HTMLParser):
    """Collect complete Google Patents claim divs, including nested markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.in_claim = False
        self.number_depth: int | None = None
        self.text_depth: int | None = None
        self.number_parts: list[str] = []
        self.text_parts: list[str] = []
        self.claims: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "div":
            return
        classes = set(dict(attrs).get("class", "").split())
        if not self.in_claim and "claim" in classes:
            self.in_claim = True
            self.depth = 1
            self.number_parts = []
            self.text_parts = []
        elif self.in_claim:
            self.depth += 1
        if self.in_claim and "claim-num" in classes:
            self.number_depth = self.depth
        if self.in_claim and "claim-text" in classes and self.text_depth is None:
            self.text_depth = self.depth

    def handle_data(self, data: str) -> None:
        if self.in_claim and self.number_depth is not None:
            self.number_parts.append(data)
        if self.in_claim and self.text_depth is not None:
            self.text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "div" or not self.in_claim:
            return
        if self.number_depth == self.depth:
            self.number_depth = None
        if self.text_depth == self.depth:
            self.text_depth = None
        self.depth -= 1
        if self.depth == 0:
            number_text = " ".join(self.number_parts)
            text = re.sub(r"\s+", " ", " ".join(self.text_parts)).strip()
            number_match = re.search(r"\d+", number_text)
            if text:
                number = int(number_match.group()) if number_match else len(self.claims) + 1
                self.claims.append((number, text))
            self.in_claim = False


def extract_claims(page: str) -> list[tuple[int, str]]:
    parser = ClaimParser()
    parser.feed(page)
    return parser.claims


def google_patent_id(patent_id: str) -> str:
    reissue = re.fullmatch(r"US-RE(\d+)-E", patent_id, flags=re.IGNORECASE)
    if reissue:
        return f"USRE{reissue.group(1)}E1"
    return patent_id.replace("-", "")


def fetch_first_independent_claim(patent_id: str, timeout: int = 30) -> dict[str, object]:
    compact_id = google_patent_id(patent_id)
    url = f"https://patents.google.com/patent/{compact_id}/en"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 patent-esco-research/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            page = response.read().decode("utf-8", errors="replace")
        claims = extract_claims(page)
        if not claims:
            return {
                "first_claim": "", "first_claim_number": "",
                "first_claim_is_independent": "", "claim_source_url": url,
                "claim_extraction_status": "no_claims_found",
            }
        independent = [(number, text) for number, text in claims if not DEPENDENT_RE.search(text[:350])]
        number, text = independent[0] if independent else claims[0]
        status = "ok_rule_based" if independent else "fallback_first_claim_needs_review"
        return {
            "first_claim": text, "first_claim_number": number,
            "first_claim_is_independent": bool(independent), "claim_source_url": url,
            "claim_extraction_status": status,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "first_claim": "", "first_claim_number": "",
            "first_claim_is_independent": "", "claim_source_url": url,
            "claim_extraction_status": f"fetch_error:{type(exc).__name__}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-xlsx", required=True)
    parser.add_argument("--reviewed-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--enrich-claims", action="store_true")
    parser.add_argument("--delay", type=float, default=0.35)
    args = parser.parse_args()

    source = pd.read_excel(args.source_xlsx, header=3)
    reviewed = pd.read_csv(args.reviewed_csv)
    reviewed_ids = set(reviewed["id"].astype(str))
    source = source[~source["patent_id"].astype(str).isin(reviewed_ids)].copy()
    source = source.drop_duplicates("patent_id")
    selected = select_candidates(source, n=100)

    keep = [
        "candidate_rank", "case_id", "patent_id", "patent_title", "patent_abstract",
        "selection_cluster", "selection_role", "cluster_similarity", "cluster_margin",
        "descriptor_word_count", "ai_likely_valid_side", "review_stratum",
    ]
    selected = selected[keep]
    if args.enrich_claims:
        claim_rows = []
        for patent_id in selected["patent_id"].astype(str):
            claim_rows.append(fetch_first_independent_claim(patent_id))
            time.sleep(args.delay)
        selected = pd.concat([selected, pd.DataFrame(claim_rows)], axis=1)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"saved={output}")
    print(f"rows={len(selected)} unique_patents={selected['patent_id'].nunique()}")
    print(selected["selection_role"].value_counts().to_string())
    if args.enrich_claims:
        print(selected["claim_extraction_status"].value_counts().to_string())


if __name__ == "__main__":
    random.seed(20260921)
    np.random.seed(20260921)
    main()
