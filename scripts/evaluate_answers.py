"""Judge the generated answers of a benchmark run, claim by claim.

Input is a run directory written by ``scripts/evaluate_retrieval.py --generate``. A second
model reads each answer next to the exact evidence the generator saw and labels every claim.
The labels are LLM judgments. ``--score-audit`` measures how far a human reviewer agrees.

Free-tier providers answer HTTP 429 often. Calls wait and retry. A question that still fails
is recorded as missing and ``--resume`` fills it in on a later run without redoing finished work.
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
from steam_review_rag.evaluation import citation_metrics, review_hash  # noqa: E402

DEFAULT_JUDGE = "openai/gpt-oss-20b"
GENERATOR = "openai/gpt-oss-120b"
MAX_WAIT_SECONDS = 600
MAX_ATTEMPTS = 8
AUDIT_FIELDS = ["audit_id", "query_id", "claim_id", "question", "claim_text", "claim_citations",
                "judge_label", "judge_supporting", "evidence", "reviewer_label", "reviewer", "rationale"]


class RateLimitExhausted(RuntimeError):
    """The provider asked for a wait longer than we are willing to spend, or kept refusing."""


def retry_after_seconds(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def call_with_backoff(call, sleep=time.sleep, max_wait: float = MAX_WAIT_SECONDS,
                      attempts: int = MAX_ATTEMPTS):
    """Retry only HTTP 429. A wait above max_wait means a daily limit, so give up at once."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:
            if getattr(error, "status_code", None) != 429:
                raise
            wait = retry_after_seconds(error)
            if wait is None:
                wait = min(30 * 2 ** attempt, max_wait)
            if wait > max_wait or attempt == attempts - 1:
                raise RateLimitExhausted(f"rate limited, asked to wait {wait:.0f}s") from error
            sleep(wait + 1)


def groq_completer(model: str, api_key: str, pause: float):
    from groq import Groq

    client = Groq(api_key=api_key, timeout=60, max_retries=0)

    def complete(system: str, user: str) -> str:
        time.sleep(pause)
        response = call_with_backoff(lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=2000,
        ))
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


def pending_work(per_query: list[dict], prior: dict[str, dict], regenerate: bool) -> tuple[list[str], list[str]]:
    """Question ids that need a new answer, and ids whose answer still lacks a verdict."""
    to_generate = [q["query_id"] for q in per_query if regenerate and q["generation_status"] != "generated"]
    generated = {q["query_id"] for q in per_query if q["generation_status"] == "generated"} | set(to_generate)
    to_judge = [q["query_id"] for q in per_query
                if q["query_id"] in generated and (prior.get(q["query_id"]) or {}).get("verdict") is None]
    return to_generate, to_judge


def judge_item(item, rows, complete, rng, second_pass=True):
    blocks = [ae.evidence_block(i, row) for i, row in enumerate(rows, start=1)]
    claims = ae.split_claims(item["answer"])
    outcome = {"claims": claims, "verdict": None, "second": None, "error": None, "blocks": blocks}
    if not claims:
        outcome["error"] = "no_claims"
        return outcome
    try:
        outcome["verdict"] = ae.judge_answer(item["question"], blocks, claims, complete)
        if outcome["verdict"] is not None and second_pass:
            shuffled = claims[:]
            rng.shuffle(shuffled)
            outcome["second"] = ae.judge_answer(item["question"], blocks, shuffled, complete)
    except Exception as error:
        outcome["error"] = type(error).__name__
    if outcome["verdict"] is None and outcome["error"] is None:
        outcome["error"] = "unparseable_verdict"
    return outcome


def stability(records: list[dict]) -> tuple[dict, dict]:
    first, second, first_abstain, second_abstain = [], [], [], []
    for record in records:
        again = record.get("second")
        if record["verdict"] is None or again is None:
            continue
        for claim_id, result in record["verdict"]["claims"].items():
            first.append(result["label"])
            second.append(again["claims"][claim_id]["label"])
        first_abstain.append(str(record["verdict"]["abstains"]))
        second_abstain.append(str(again["abstains"]))
    return ae.label_agreement(first, second), ae.label_agreement(first_abstain, second_abstain)


def percent(value) -> str:
    return "unmeasured" if value is None else f"{100 * value:.1f}%"


def interval(row, key) -> str:
    low, high = row[f"{key}_ci_low"], row[f"{key}_ci_high"]
    return "" if low is None else f" ({100 * low:.0f} to {100 * high:.0f})"


def write_summary(path: Path, rows: list[dict], info: dict) -> None:
    overall = rows[0]
    lines = [
        f"# Answer evaluation, LLM-judged ({info['date']})",
        "",
        f"Generator `{info['generator']}`. Judge `{info['judge']}`. {overall['answers_judged']} of "
        f"{info['questions']} benchmark questions have a judged answer, sentence by sentence against the "
        "five reviews the generator saw.",
        "",
        "These are LLM judgments. No person has labelled these claims unless the audit section below says so.",
        "",
    ]
    if overall["answers_judged"] < info["questions"]:
        lines += [
            f"**Incomplete.** {info['questions'] - overall['answers_judged']} questions have no judged answer "
            f"({info['unanswered']} without a generated answer, {info['unjudged']} without a verdict). "
            "The missing questions are not a random sample, so these rates do not describe the whole benchmark. "
            "Run the workflow again with resume to fill them in.",
            "",
        ]
    lines += [
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


def run(run_dir: Path, judge_model: str, audit_size: int, pause: float, resume: bool,
        regenerate: bool, stability_every: int, budget_minutes: float) -> dict:
    import pandas as pd
    from steam_review_rag.artifacts import ensure_artifacts

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Judging requires an explicitly configured GROQ_API_KEY")
    deadline = time.monotonic() + 60 * budget_minutes
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest.get("generation_requested"):
        raise ValueError("The run has no generated answers. Use evaluate_retrieval.py --generate")
    per_query = json.loads((run_dir / "per_query.json").read_text())
    report_path = run_dir / "answer_eval.json"
    prior = {}
    if resume and report_path.exists():
        prior = {r["query_id"]: r for r in json.loads(report_path.read_text())["records"]}
    ensure_artifacts(ROOT)
    corpus = pd.read_csv(ROOT / "outputs" / "rag" / "rag_corpus.csv", low_memory=False)
    by_id = corpus.assign(raw_row_id=corpus.raw_row_id.astype(str)).set_index("raw_row_id")
    to_generate, to_judge = pending_work(per_query, prior, regenerate)
    items = {q["query_id"]: q for q in per_query}

    regenerated = list(manifest.get("regenerated_questions", []))
    if to_generate:
        from steam_review_rag.retrieval import SteamReviewRAG
        backend = SteamReviewRAG.load(ROOT)
        for query_id in to_generate:
            if time.monotonic() > deadline:
                break
            item = items[query_id]
            evidence = pd.DataFrame([dict(row) for row in evidence_for(item, by_id)])
            try:
                item["answer"] = call_with_backoff(
                    lambda: backend.generate_answer(item["question"], evidence, api_key, GENERATOR))
            except Exception as error:
                print(query_id, "generation failed:", type(error).__name__, flush=True)
                continue
            item["generation_status"], item["generation_diagnostic"] = "generated", None
            item["citation_numbering"] = citation_metrics(item["answer"], len(evidence))
            regenerated.append(query_id)
            print(query_id, "regenerated", flush=True)
        (run_dir / "per_query.json").write_text(json.dumps(per_query, indent=2) + "\n")
        manifest["regenerated_questions"] = sorted(set(regenerated))
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        to_generate, to_judge = pending_work(per_query, prior, False)

    complete = groq_completer(judge_model, api_key, pause)
    rng = random.Random(42)
    records = []
    for index, item in enumerate(per_query):
        query_id = item["query_id"]
        if item["generation_status"] != "generated":
            continue
        record = prior.get(query_id)
        if record is None or (record["verdict"] is None and query_id in to_judge):
            if time.monotonic() > deadline:
                record = record or {"query_id": query_id, "split": item["split"], "category": item["category"],
                                    "question": item["question"], "answer": item["answer"], "claims": [],
                                    "verdict": None, "second": None, "judge_error": "not_attempted"}
            else:
                outcome = judge_item(item, evidence_for(item, by_id), complete, rng,
                                     second_pass=index % stability_every == 0)
                verdict = outcome["verdict"]
                second = None
                if outcome["second"] is not None:
                    second = {"claims": outcome["second"]["claims"], "abstains": outcome["second"]["abstains"]}
                record = {
                    "query_id": query_id, "split": item["split"], "category": item["category"],
                    "question": item["question"], "answer": item["answer"],
                    "claims": ae.judged_claims(outcome["claims"], verdict) if verdict else outcome["claims"],
                    "verdict": verdict, "second": second, "judge_error": outcome["error"],
                }
                print(query_id, "judged" if verdict else f"failed:{outcome['error']}", flush=True)
        records.append(record)

    audit_pool = []
    for record in records:
        if record["verdict"] is None:
            continue
        rows = evidence_for(items[record["query_id"]], by_id)
        text = "\n\n".join(ae.evidence_block(i, row) for i, row in enumerate(rows, start=1))
        audit_pool += [{**claim, "query_id": record["query_id"], "question": record["question"], "evidence": text}
                       for claim in record["claims"]]
    summary = ae.summarize(records)
    claim_agreement, abstain_agreement = stability(records)
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
    kappa = claim_agreement["kappa"]
    stability_text = (
        f"{claim_agreement['n']} claims in {abstain_agreement['n']} answers judged twice with the claims in a "
        f"different order. Label agreement {percent(claim_agreement['agreement'])}"
        + ("" if kappa is None else f", kappa {kappa:.2f}")
        + f". Abstain agreement {percent(abstain_agreement['agreement'])}."
    )
    judged = sum(r["verdict"] is not None for r in records)
    info = {
        "date": time.strftime("%Y-%m-%d", time.gmtime()), "generator": manifest["generator"],
        "judge": judge_model, "questions": len(per_query), "answers": len(records),
        "unanswered": sum(q["generation_status"] != "generated" for q in per_query),
        "unjudged": len(records) - judged, "regenerated_questions": sorted(set(regenerated)),
        "stability": stability_text,
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
    report_path.write_text(json.dumps(report, indent=2) + "\n")
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
    parser.add_argument("--resume", action="store_true", help="Keep finished verdicts and fill in the rest")
    parser.add_argument("--regenerate-failed", action="store_true", help="Generate answers that failed earlier")
    parser.add_argument("--stability-every", type=int, default=3,
                        help="Judge every Nth answer a second time to measure stability")
    parser.add_argument("--budget-minutes", type=float, default=150)
    parser.add_argument("--score-audit", type=Path, help="Compare a filled claim_audit_sample.csv with the judge")
    args = parser.parse_args()
    if args.score_audit:
        print(json.dumps(score_audit(args.score_audit), indent=2))
    elif args.run:
        print(json.dumps(run(args.run, args.judge_model, args.audit_size, args.pause, args.resume,
                             args.regenerate_failed, args.stability_every, args.budget_minutes)["summary"], indent=2))
    else:
        parser.error("pass --run or --score-audit")
