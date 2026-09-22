"""Pure prompt construction for evidence-grounded answers."""

from collections.abc import Iterable, Mapping
from typing import Any


def build_grounded_prompt(
    question: str,
    evidence_rows: Iterable[Mapping[str, Any]],
    review_character_limit: int = 900,
) -> str:
    """Build a numbered evidence prompt without adding external information."""
    blocks = []
    for number, row in enumerate(evidence_rows, start=1):
        review = " ".join(str(row["review"]).split())[:review_character_limit]
        blocks.append(
            f"[{number}] Game: {row['game_name']}\n"
            f"Player verdict: {row['recommendation']}\n"
            f"Review evidence: {review}"
        )

    context = "\n\n".join(blocks)
    return (
        f"Player-review evidence:\n\n{context}\n\nQuestion: {question.strip()}\n\n"
        "Answer only from the evidence above."
    )
