# Answer evaluation, LLM-judged (2026-10-30)

Generator `openai/gpt-oss-120b`. Judge `openai/gpt-oss-20b`. 77 of 80 benchmark questions have a judged answer, sentence by sentence against the five reviews the generator saw.

These are LLM judgments. No person has labelled these claims unless the audit section below says so.

**Incomplete.** 3 questions have no judged answer (0 without a generated answer, 3 without a verdict). The missing questions are not a random sample, so these rates do not describe the whole benchmark. Run the workflow again with resume to fill them in.

| Category | Answers judged | Claims | Supported (95% CI) | Cited claims correct (95% CI) | Abstains (95% CI) | Answers with an unsupported or contradicted claim |
|---|---:|---:|---|---|---|---:|
| all | 77 of 80 | 381 | 88.5% (85 to 92) | 100.0% (100 to 100) | 1.3% (0 to 7) | 2 |
| comparison | 10 of 10 | 51 | 86.3% (73 to 96) | 100.0% (100 to 100) | 0.0% (0 to 28) | 1 |
| game_topic | 38 of 40 | 198 | 87.9% (82 to 93) | 100.0% (100 to 100) | 0.0% (0 to 9) | 1 |
| named_game | 19 of 20 | 116 | 88.8% (83 to 94) | 100.0% (100 to 100) | 0.0% (0 to 17) | 0 |
| unsupported | 10 of 10 | 16 | 100.0% (100 to 100) | unmeasured | 10.0% (2 to 40) | 0 |

Supported counts only the label supported. Partially supported claims are reported in the JSON. A cited claim is correct when it is supported and at least one cited review is among the reviews the judge named as support. Intervals resample answers, not claims. For the unsupported category the abstain rate is the headline and higher is better. For the other categories a high abstain rate means the answer refused although evidence was available.

Judge stability: 65 claims in 12 answers judged twice with the claims in a different order. Label agreement 89.2%, kappa 0.31. Abstain agreement 100.0%.

Human audit: not yet done. `claim_audit_sample.csv` holds 40 claims stratified by judge label for a reviewer to fill in.
