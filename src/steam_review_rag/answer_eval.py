"""Claim-level judging of generated answers, with no network or model imports.

The judge is a language model, so every number produced here is an LLM-judged estimate
until a person has audited a sample. The audit helpers at the bottom measure how far a
reviewer and the judge agree.
"""

from collections import Counter
import json
import math
import random
import re
from typing import Callable, Iterable, Mapping

LABELS = ("supported", "partially_supported", "unsupported", "contradicted")
SUPPORT_LABELS = ("supported", "partially_supported")
REVIEW_CHARACTER_LIMIT = 900
CITATION = re.compile(r"[\[\u3010](\d+)[\]\u3011]")
LEADING_CITATIONS = re.compile(r"^((?:[\[\u3010]\d+[\]\u3011][\s,;]*)+)(.*)$", re.DOTALL)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(*])")
LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

JUDGE_SYSTEM_PROMPT = (
    "You check claims against numbered Steam reviews. Use only the numbered reviews. "
    "Ignore what you know about the games. Reply with one JSON object and nothing else."
)


class JudgeParseError(ValueError):
    """The judge reply was not a complete, valid verdict."""


def evidence_block(number: int, row: Mapping[str, object]) -> str:
    """Render one review exactly as the generator saw it, including truncation."""
    review = " ".join(str(row["review"]).split())[:REVIEW_CHARACTER_LIMIT]
    return (
        f"[{number}] Game: {row['game_name']}\n"
        f"Player verdict: {row['recommendation']}\n"
        f"Review evidence: {review}"
    )


def is_heading(text: str) -> bool:
    """A short uncited fragment with no closing punctuation is a title, not a statement."""
    words = CITATION.sub("", text).strip()
    return (not CITATION.search(text) and len(words.split()) <= 3
            and not re.search(r"[.!?\"\u201d)]$", words))


def split_claims(answer: str) -> list[dict]:
    """Split an answer into sentence-level claims and attach their citation numbers."""
    segments: list[str] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or TABLE_SEPARATOR.match(line):
            continue
        line = LIST_MARKER.sub("", line)
        line = re.sub(r"[*_#`]+", "", line)
        if line.startswith("|"):
            line = ", ".join(cell.strip() for cell in line.strip("|").split("|") if cell.strip())
        segments.extend(part.strip() for part in SENTENCE_BOUNDARY.split(line) if part.strip())
    merged: list[str] = []
    for segment in segments:
        leading = LEADING_CITATIONS.match(segment)
        if leading and merged:
            merged[-1] += " " + leading.group(1).strip()
            segment = leading.group(2).strip()
        if segment:
            merged.append(segment)
    claims = []
    for text in merged:
        if not re.search(r"[A-Za-z]", CITATION.sub("", text)) or is_heading(text):
            continue
        citations = sorted({int(number) for number in CITATION.findall(text)})
        claims.append({"claim_id": f"C{len(claims) + 1}", "text": text, "citations": citations})
    return claims


def build_judge_prompt(question: str, evidence_blocks: Iterable[str], claims: list[dict]) -> str:
    claim_lines = "\n".join(f"{claim['claim_id']}: {claim['text']}" for claim in claims)
    return (
        "Numbered reviews:\n\n" + "\n\n".join(evidence_blocks) + f"\n\nQuestion asked: {question}\n\n"
        f"Claims from the answer:\n{claim_lines}\n\n"
        "For every claim give one label.\n"
        "supported: the reviews state it.\n"
        "partially_supported: the reviews state part of it or only imply it.\n"
        "unsupported: the reviews do not say it.\n"
        "contradicted: the reviews say the opposite.\n"
        "A claim that the reviews lack some information is supported only if none of the "
        "reviews contain that information.\n"
        "Citation numbers inside a claim are the answer's own and may be wrong. Judge the claim, "
        "then list in supporting the review numbers that actually support it, or an empty list.\n"
        "Also set abstains. It is true only if the answer says the reviews do not contain what the "
        "question asks for and does not supply that information from outside the reviews.\n"
        'Format: {"claims": [{"id": "C1", "label": "supported", "supporting": [1]}], '
        '"abstains": false}'
    )


def parse_judge_response(text: str, claim_ids: list[str], evidence_count: int) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise JudgeParseError("no JSON object in reply")
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise JudgeParseError("reply is not valid JSON") from error
    items = payload.get("claims")
    if not isinstance(items, list) or not isinstance(payload.get("abstains"), bool):
        raise JudgeParseError("reply lacks claims or abstains")
    verdicts = {}
    for item in items:
        if not isinstance(item, dict) or item.get("label") not in LABELS:
            raise JudgeParseError("invalid label")
        supporting = item.get("supporting", [])
        if not isinstance(supporting, list) or any(
            type(number) is not int or not 1 <= number <= evidence_count for number in supporting
        ):
            raise JudgeParseError("invalid supporting review number")
        if item.get("id") in verdicts:
            raise JudgeParseError("duplicate claim id")
        verdicts[item.get("id")] = {"label": item["label"], "supporting": sorted(set(supporting))}
    if set(verdicts) != set(claim_ids):
        raise JudgeParseError("claim ids do not match")
    return {"claims": verdicts, "abstains": payload["abstains"]}


def judge_answer(
    question: str,
    evidence_blocks: list[str],
    claims: list[dict],
    complete: Callable[[str, str], str],
    attempts: int = 2,
) -> dict | None:
    """Return the validated verdict, or None when the judge never produced one."""
    prompt = build_judge_prompt(question, evidence_blocks, claims)
    for _ in range(attempts):
        try:
            return parse_judge_response(
                complete(JUDGE_SYSTEM_PROMPT, prompt), [c["claim_id"] for c in claims], len(evidence_blocks)
            )
        except JudgeParseError:
            continue
    return None


def judged_claims(claims: list[dict], verdict: dict) -> list[dict]:
    """Merge claims with verdicts and decide whether each citation holds up."""
    rows = []
    for claim in claims:
        result = verdict["claims"][claim["claim_id"]]
        cited = bool(claim["citations"])
        citation_ok = (
            cited
            and result["label"] in SUPPORT_LABELS
            and bool(set(claim["citations"]) & set(result["supporting"]))
        )
        rows.append({**claim, "label": result["label"], "supporting": result["supporting"],
                     "cited": cited, "citation_ok": citation_ok})
    return rows


def refresh_record(record: dict) -> dict:
    """Re-apply the current claim rules to a stored record without asking the judge again."""
    if record["verdict"] is None:
        return record
    kept = [claim for claim in record["claims"] if not is_heading(claim["text"])]
    verdict_claims = {claim["claim_id"]: record["verdict"]["claims"][claim["claim_id"]] for claim in kept}
    claims = []
    for claim in kept:
        citations = sorted({int(number) for number in CITATION.findall(claim["text"])})
        result = verdict_claims[claim["claim_id"]]
        cited = bool(citations)
        citation_ok = cited and result["label"] in SUPPORT_LABELS and bool(set(citations) & set(result["supporting"]))
        claims.append({**claim, "citations": citations, "cited": cited, "citation_ok": citation_ok})
    return {**record, "claims": claims, "verdict": {**record["verdict"], "claims": verdict_claims}}


def bootstrap_ratio(pairs: list[tuple[int, int]], resamples: int = 2000, seed: int = 42,
                    level: float = 0.95) -> tuple[float | None, float | None, float | None]:
    """Ratio of summed numerators to denominators with an answer-level bootstrap interval.

    Claims inside one answer are not independent, so the unit of resampling is the answer.
    """
    pairs = [pair for pair in pairs if pair[1] > 0]
    if not pairs:
        return None, None, None
    total = sum(d for _, d in pairs)
    point = sum(n for n, _ in pairs) / total
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        denominator = sum(d for _, d in sample)
        draws.append(sum(n for n, _ in sample) / denominator)
    draws.sort()
    low = draws[int((1 - level) / 2 * (resamples - 1))]
    high = draws[int((1 + level) / 2 * (resamples - 1))]
    return point, low, high


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    if n == 0:
        return None, None, None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(records: list[dict]) -> list[dict]:
    """One row per category plus an overall row. Records hold one answer each."""
    groups = [("all", records)] + [
        (category, [r for r in records if r["category"] == category])
        for category in sorted({r["category"] for r in records})
    ]
    rows = []
    for name, group in groups:
        judged = [r for r in group if r["verdict"] is not None]
        claims = [c for r in judged for c in r["claims"]]
        labels = Counter(c["label"] for c in claims)
        row = {
            "category": name, "answers": len(group), "answers_judged": len(judged),
            "judge_failures": len(group) - len(judged), "claims": len(claims),
        }
        for label in LABELS:
            row[f"{label}_share"] = labels[label] / len(claims) if claims else None
        supported = [(sum(c["label"] == "supported" for c in r["claims"]), len(r["claims"])) for r in judged]
        row["supported_rate"], row["supported_ci_low"], row["supported_ci_high"] = bootstrap_ratio(supported)
        cited = [(sum(c["citation_ok"] for c in r["claims"] if c["cited"]),
                  sum(c["cited"] for c in r["claims"])) for r in judged]
        row["cited_claim_share"] = sum(c["cited"] for c in claims) / len(claims) if claims else None
        row["citation_correct_rate"], row["citation_ci_low"], row["citation_ci_high"] = bootstrap_ratio(cited)
        abstained = sum(r["verdict"]["abstains"] for r in judged)
        row["abstain_rate"], row["abstain_ci_low"], row["abstain_ci_high"] = wilson(abstained, len(judged))
        row["answers_with_unsupported_claim"] = sum(
            any(c["label"] in ("unsupported", "contradicted") for c in r["claims"]) for r in judged
        )
        rows.append(row)
    return rows


def label_agreement(first: list[str], second: list[str]) -> dict:
    """Raw agreement and Cohen's kappa between two label sequences of equal length."""
    if len(first) != len(second):
        raise ValueError("label sequences differ in length")
    n = len(first)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    observed = sum(a == b for a, b in zip(first, second)) / n
    left, right = Counter(first), Counter(second)
    expected = sum(left[label] * right[label] for label in set(left) | set(right)) / (n * n)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {"n": n, "agreement": observed, "kappa": kappa}


def audit_sample(claim_rows: list[dict], size: int, seed: int = 42) -> list[dict]:
    """Stratified by judge label so rare labels are not drowned by the supported majority."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {label: [] for label in LABELS}
    for row in claim_rows:
        by_label[row["label"]].append(row)
    for rows in by_label.values():
        rng.shuffle(rows)
    chosen: list[dict] = []
    while len(chosen) < min(size, len(claim_rows)):
        progressed = False
        for label in LABELS:
            if by_label[label] and len(chosen) < size:
                chosen.append(by_label[label].pop())
                progressed = True
        if not progressed:
            break
    return chosen
