"""Write a blind sample of judged claims for a person to label, and a separate key that holds the judge's labels.

The sample file carries the review text, so it stays out of git, as claim_audit_sample.csv does. The key file carries no review text.
The sample is stratified by judge label, so it over-represents the rare labels and its agreement does not describe all 399 claims.

    python scripts/make_human_audit.py                       # writes the blind sample and the key
    python scripts/make_human_audit.py --score FILLED.csv    # agreement with the judge, from the filled sample and the key
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from steam_review_rag import answer_eval as ae  # noqa: E402
from steam_review_rag.artifacts import ensure_artifacts  # noqa: E402
from steam_review_rag.evaluation import review_hash  # noqa: E402

RUN = ROOT / "outputs" / "evaluation" / "generated-v1"
AUDIT = ROOT / "evaluation" / "audit"
SAMPLE = AUDIT / "human_audit_sample.csv"
KEY = AUDIT / "human_audit_key.csv"
SAMPLE_SIZE = 60
SAMPLE_FIELDS = ["audit_id", "query_id", "claim_id", "question", "claim_text", "claim_citations", "evidence", "reviewer_label", "reviewer", "rationale"]
KEY_FIELDS = ["audit_id", "query_id", "claim_id", "judge_label", "judge_supporting"]


def evidence_text(item: dict, by_id) -> str:
    rows = []
    for entry in item["top_5"]["reranked"]:
        row = by_id.loc[entry["review_id"]]
        if review_hash(str(row["review"])) != entry["review_sha256"]:
            raise ValueError(f"Evidence hash mismatch in {item['query_id']}")
        rows.append(row)
    return "\n\n".join(ae.evidence_block(i, row) for i, row in enumerate(rows, start=1))


def build(size: int = SAMPLE_SIZE) -> int:
    ensure_artifacts(ROOT)
    corpus = pd.read_csv(ROOT / "outputs" / "rag" / "rag_corpus.csv", low_memory=False)
    by_id = corpus.assign(raw_row_id=corpus.raw_row_id.astype(str)).set_index("raw_row_id")
    items = {q["query_id"]: q for q in json.loads((RUN / "per_query.json").read_text())}
    report = json.loads((RUN / "answer_eval.json").read_text())
    pool = []
    for record in report["records"]:
        if record["verdict"] is None:
            continue
        text = evidence_text(items[record["query_id"]], by_id)
        pool += [{**claim, "query_id": record["query_id"], "question": record["question"], "evidence": text} for claim in record["claims"]]
    sample = ae.audit_sample(pool, size)
    with SAMPLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for number, claim in enumerate(sample, start=1):
            writer.writerow({"audit_id": number, "query_id": claim["query_id"], "claim_id": claim["claim_id"], "question": claim["question"],
                             "claim_text": claim["text"], "claim_citations": claim["citations"], "evidence": claim["evidence"],
                             "reviewer_label": "", "reviewer": "", "rationale": ""})
    with KEY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEY_FIELDS)
        writer.writeheader()
        for number, claim in enumerate(sample, start=1):
            writer.writerow({"audit_id": number, "query_id": claim["query_id"], "claim_id": claim["claim_id"], "judge_label": claim["label"],
                             "judge_supporting": claim["supporting"]})
    return len(sample)


def score(filled: Path) -> dict:
    with filled.open(newline="", encoding="utf-8") as handle:
        rows = {r["audit_id"]: r for r in csv.DictReader(handle)}
    with KEY.open(newline="", encoding="utf-8") as handle:
        key = {r["audit_id"]: r for r in csv.DictReader(handle)}
    labelled = [i for i, r in rows.items() if r["reviewer_label"].strip()]
    bad = {rows[i]["reviewer_label"].strip() for i in labelled} - set(ae.LABELS)
    if bad:
        raise ValueError(f"Unknown reviewer labels: {sorted(bad)}")
    result = ae.label_agreement([key[i]["judge_label"] for i in labelled], [rows[i]["reviewer_label"].strip() for i in labelled])
    result["rows_in_sample"] = len(rows)
    result["rows_labelled"] = len(labelled)
    by_label = {}
    for i in labelled:
        by_label.setdefault(key[i]["judge_label"], []).append(rows[i]["reviewer_label"].strip() == key[i]["judge_label"])
    result["agreement_by_judge_label"] = {k: {"n": len(v), "agreed": int(sum(v))} for k, v in sorted(by_label.items())}
    (AUDIT / "human_audit_agreement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", type=Path, help="A filled human_audit_sample.csv")
    args = parser.parse_args()
    print(json.dumps(score(args.score), indent=2) if args.score else f"wrote {build()} claims")
