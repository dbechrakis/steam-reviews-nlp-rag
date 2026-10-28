"""Judge the generated answers of a benchmark run, claim by claim.

Input is a run directory written by ``scripts/evaluate_retrieval.py --generate``. A second
model reads each answer next to the exact evidence the generator saw and labels every claim.
The labels are LLM judgments. ``--score-audit`` measures how far a human reviewer agrees.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steam_review_rag import answer_eval as ae  # noqa: E402
from steam_review_rag.evaluation import review_hash  # noqa: E402

DEFAULT_JUDGE = "openai/gpt-oss-20b"
AUDIT_FIELDS = ["audit_id", "query_id", "claim_id", "question", "claim_text", "claim_citations",
                "judge_label", "judge_supporting", "evidence", "reviewer_label", "reviewer", "rationale"]


def groq_completer(model: str, api_key: str, pause: float):
    from groq import Groq

    client = Groq(api_key=api_key, timeout=60, max_retries=2)

    def complete(system: str, user: str) -> str:
        time.sleep(pause)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""

    return complete


def evidence_for(item: dict, by_id) -> list[dict]:
    rows = []
    for entry in item["top_5"]["reranked"]:
        row = by_id.loc[entry["review_id"]]
        if review_hash(str(row["review"])) != entry["review_sha256"]:
            raise ValueError(f"Evidence hash mismatch in {item['query_id']}")
        rows.append(row)
    return rows


def judge_item(item, rows, complete, rng):
    blocks = [ae.evidence_block(i, row) for i, row in enumerate(rows, start=1)]
    claims = ae.split_claims(item["answer"])
    outcome = {"claims": claims, "verdict": None, "second": None, "error": None, "blocks": blocks}
    if not claims:
        outcome["error"] = "no_claims"
        return outcome
    try:
        outcome["verdict"] = ae.judge_answer(item["question"], blocks, claims, complete)
        if outcome["verdict"] is not None:
            shuffled = claims[:]
            rng.shuffle(shuffled)
            outcome["second"] = ae.judge_answer(item["question"], blocks, shuffled, complete)
    except Exception as error:
        outcome["error"] = type(error).__name__
    if outcome["verdict"] is None and outcome["error"] is None:
        outcome["error"] = "unparseable_verdict"
    return outcome


def percent(value) -> str:
    return "unmeasured" if value is None else f"{100 * value:.1f}%"


def interval(row, key) -> str:
    low, high = row[f"{key}_ci_low"], row[f"{key}_ci_high"]
    return "" if low is None else f" ({100 * low:.0f} to {100 * high:.0f})"


def write_summary(path: Path, rows: list[dict], info: dict) -> None:
    lines = [
        f"# Answer evaluation, LLM-judged ({info['date']})",
        "",
        f"Generator `{info['generator']}`. Judge `{info['judge']}`. {info['answers']} generated answers "
        f"from the 80-question benchmark ({info['generation_failures']} questions had no answer because generation failed), "
        "judged sentence by sentence against the five reviews the generator saw.",
        "",
        "These are LLM judgments. No person has labelled these claims unless the audit section below says so.",
        "",
        "| Category | Answers judged | Claims | Supported (95% CI) | Cited claims correct (95% CI) | "
        "Abstains (95% CI) | Answers with an unsupported or contradicted claim |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['answers_judged']} of {row['answers']} | {row['claims']} | "
            f"{percent(row['supported_rate'])}{interval(row, 'supported')} | "
            f"{percent(row['citation_correct_rate'])}{interval(row, 'citation')} | "
            f"{percent(row['abstain_rate'])}{interval(row, 'abstain')} | "
            f"{row['answers_with_unsupported_claim']} |"
        )
    lines += [
        "",
        "Supported counts only the label supported. Partially supported claims are reported in the JSON. "
        "A cited claim is correct when it is supported and at least one cited review is among the reviews the "
        "judge named as support. Intervals resample answers, not claims. For the unsupported category the "
        "abstain rate is the headline and higher is better. For the other categories a high abstain rate means "
        "the answer refused although evidence was available.",
        "",
        f"Judge stability: {info['stability']}",
        "",
        info["audit"],
        "",
    ]
    path.write_text("\n".join(lines))


def run(run_dir: Path, judge_model: str, audit_size: int, pause: float) -> dict:
    import pandas as pd
    from steam_review_rag.artifacts import ensure_artifacts

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Judging requires an explicitly configured GROQ_API_KEY")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest.get("generation_requested"):
        raise ValueError("The run has no generated answers. Use evaluate_retrieval.py --generate")
    per_query = json.loads((run_dir / "per_query.json").read_text())
    ensure_artifacts(ROOT)
    corpus = pd.read_csv(ROOT / "outputs" / "rag" / "rag_corpus.csv", low_memory=False)
    by_id = corpus.assign(raw_row_id=corpus.raw_row_id.astype(str)).set_index("raw_row_id")
    complete = groq_completer(judge_model, api_key, pause)
    rng = random.Random(42)

    records, audit_pool, first_labels, second_labels, first_abstain, second_abstain = [], [], [], [], [], []
    generation_failures = 0
    for item in per_query:
        if item["generation_status"] != "generated":
            generation_failures += 1
            continue
        outcome = judge_item(item, evidence_for(item, by_id), complete, rng)
        verdict = outcome["verdict"]
        judged = ae.judged_claims(outcome["claims"], verdict) if verdict else outcome["claims"]
        records.append({
            "query_id": item["query_id"], "split": item["split"], "category": item["category"],
            "question": item["question"], "answer": item["answer"], "claims": judged,
            "verdict": verdict, "second_pass_ok": outcome["second"] is not None, "judge_error": outcome["error"],
        })
        if verdict:
            for claim in judged:
                audit_pool.append({**claim, "query_id": item["query_id"], "question": item["question"],
                                   "evidence": "\n\n".join(outcome["blocks"])})
        if verdict and outcome["second"]:
            for claim_id, result in verdict["claims"].items():
                first_labels.append(result["label"])
                second_labels.append(outcome["second"]["claims"][claim_id]["label"])
            first_abstain.append(str(verdict["abstains"]))
            second_abstain.append(str(outcome["second"]["abstains"]))
        print(item["query_id"], "judged" if verdict else f"failed:{outcome['error']}", flush=True)

    summary = ae.summarize(records)
    claim_agreement = ae.label_agreement(first_labels, second_labels)
    abstain_agreement = ae.label_agreement(first_abstain, second_abstain)
    sample = ae.audit_sample(audit_pool, audit_size)
    with (run_dir / "claim_audit_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for number, claim in enumerate(sample, start=1):
            writer.writerow({
                "audit_id": number, "query_id": claim["query_id"], "claim_id": claim["claim_id"],
                "question": claim["question"], "claim_text": claim["text"],
                "claim_citations": claim["citations"], "judge_label": claim["label"],
                "judge_supporting": claim["supporting"], "evidence": claim["evidence"],
                "reviewer_label": "", "reviewer": "", "rationale": "",
            })
    stability = (
        f"{claim_agreement['n']} claims judged twice with the claims in a different order. "
        f"Label agreement {percent(claim_agreement['agreement'])}"
        + ("" if claim_agreement["kappa"] is None else f", kappa {claim_agreement['kappa']:.2f}")
        + f". Abstain agreement {percent(abstain_agreement['agreement'])} over {abstain_agreement['n']} answers."
    )
    info = {
        "date": time.strftime("%Y-%m-%d", time.gmtime()), "generator": manifest["generator"],
        "judge": judge_model, "answers": len(records), "generation_failures": generation_failures,
        "stability": stability,
        "audit": f"Human audit: not yet done. `claim_audit_sample.csv` holds {len(sample)} claims "
                 "stratified by judge label for a reviewer to fill in.",
    }
    report = {
        "info": info, "summary": summary, "claim_agreement": claim_agreement,
        "abstain_agreement": abstain_agreement,
        "source_run_manifest_sha256": hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
        "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                        ("scripts/evaluate_answers.py", "src/steam_review_rag/answer_eval.py")},
        "records": records,
    }
    (run_dir / "answer_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary(run_dir / "answer_eval_summary.md", summary, info)
    return {"info": info, "summary": summary}


def score_audit(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    filled = [r for r in rows if r["reviewer_label"].strip()]
    bad = {r["reviewer_label"] for r in filled} - set(ae.LABELS)
    if bad:
        raise ValueError(f"Unknown reviewer labels: {sorted(bad)}")
    result = ae.label_agreement([r["judge_label"] for r in filled], [r["reviewer_label"] for r in filled])
    result["rows_in_sample"] = len(rows)
    result["rows_labelled"] = len(filled)
    path.with_name("audit_agreement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Directory with per_query.json and manifest.json")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--audit-size", type=int, default=40)
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds between judge calls")
    parser.add_argument("--score-audit", type=Path, help="Compare a filled claim_audit_sample.csv with the judge")
    args = parser.parse_args()
    if args.score_audit:
        print(json.dumps(score_audit(args.score_audit), indent=2))
    elif args.run:
        print(json.dumps(run(args.run, args.judge_model, args.audit_size, args.pause)["summary"], indent=2))
    else:
        parser.error("pass --run or --score-audit")
