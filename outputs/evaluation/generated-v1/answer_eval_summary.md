# Answer evaluation, LLM-judged (2026-10-29)

Generator `openai/gpt-oss-120b`. Judge `openai/gpt-oss-20b`. 42 of 80 benchmark questions have a judged answer, sentence by sentence against the five reviews the generator saw.

These are LLM judgments. No person has labelled these claims unless the audit section below says so.

**Incomplete.** 38 questions have no judged answer (0 without a generated answer, 38 without a verdict). The missing questions are not a random sample, so these rates do not describe the whole benchmark. Run the workflow again with resume to fill them in.

| Category | Answers judged | Claims | Supported (95% CI) | Cited claims correct (95% CI) | Abstains (95% CI) | Answers with an unsupported or contradicted claim |
|---|---:|---:|---|---|---|---:|
| all | 42 of 80 | 204 | 85.8% (79 to 91) | 100.0% (100 to 100) | 0.0% (0 to 8) | 2 |
| comparison | 1 of 10 | 2 | 0.0% (0 to 0) | unmeasured | 0.0% (0 to 79) | 1 |
| game_topic | 24 of 40 | 123 | 84.6% (76 to 92) | 100.0% (100 to 100) | 0.0% (0 to 14) | 1 |
| named_game | 12 of 20 | 73 | 89.0% (82 to 96) | 100.0% (100 to 100) | 0.0% (0 to 24) | 0 |
| unsupported | 5 of 10 | 6 | 100.0% (100 to 100) | unmeasured | 0.0% (0 to 43) | 0 |

Supported counts only the label supported. Partially supported claims are reported in the JSON. A cited claim is correct when it is supported and at least one cited review is among the reviews the judge named as support. Intervals resample answers, not claims. For the unsupported category the abstain rate is the headline and higher is better. For the other categories a high abstain rate means the answer refused although evidence was available.

Judge stability: 0 claims in 0 answers judged twice with the claims in a different order. Label agreement unmeasured. Abstain agreement unmeasured.

Human audit: not yet done. `claim_audit_sample.csv` holds 40 claims stratified by judge label for a reviewer to fill in.
