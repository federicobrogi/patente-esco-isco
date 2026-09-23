"""Freeze the 50-patent benchmark from the annotated comparison workbook."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "outputs" / "candidate_pool_100_two_variants" / (
    "confronto_11_modelli_50_casi_annotazione_new_grafici.xlsx"
)
DESTINATION = ROOT / "data" / "patents" / "patents_50_annotated_benchmark.csv"

KEEP = [
    "patent_id", "patent_title", "patent_abstract", "first_claim",
    "cpc_codes", "cpc_description", "esco_code_suggested",
    "esco_profession_suggested", "esco_motivation", "sample_origin",
    "annotation_sample_stratum",
]


def main() -> None:
    source = pd.read_excel(SOURCE, sheet_name="Foglio1")
    missing = sorted(set(KEEP) - set(source.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    benchmark = source[KEEP].drop_duplicates("patent_id", keep="first").copy()
    if len(benchmark) != 50:
        raise ValueError(f"Expected 50 distinct patents, found {len(benchmark)}")
    if benchmark[["patent_id", "patent_title", "patent_abstract"]].isna().any().any():
        raise ValueError("The frozen benchmark contains missing core patent fields")
    benchmark.to_csv(DESTINATION, index=False, encoding="utf-8")
    print(f"Wrote {len(benchmark)} patents to {DESTINATION}")


if __name__ == "__main__":
    main()
