# Answer evaluation, LLM-judged (2026-10-30)

Generator `openai/gpt-oss-120b`. Judge `openai/gpt-oss-20b`. 80 of 80 benchmark questions have a judged answer, sentence by sentence against the five reviews the generator saw.

These are LLM judgments. No person has labelled these claims unless the audit section below says so.

| Category | Answers judged | Claims | Supported (95% CI) | Cited claims correct (95% CI) | Judge abstain flag (95% CI) | Answers with an unsupported or contradicted claim |
|---|---:|---:|---|---|---|---:|
| all | 80 of 80 | 399 | 88.5% (85 to 92) | 100.0% (100 to 100) | 1.2% (0 to 7) | 2 |
| comparison | 10 of 10 | 51 | 86.3% (73 to 96) | 100.0% (100 to 100) | 0.0% (0 to 28) | 1 |
| game_topic | 40 of 40 | 209 | 88.0% (82 to 93) | 100.0% (100 to 100) | 0.0% (0 to 9) | 1 |
| named_game | 20 of 20 | 123 | 88.6% (84 to 93) | 100.0% (100 to 100) | 0.0% (0 to 16) | 0 |
| unsupported | 10 of 10 | 16 | 100.0% (100 to 100) | unmeasured | 10.0% (2 to 40) | 0 |

Supported counts only the label supported. Partially supported claims are reported in the JSON. A cited claim is correct when it is supported and at least one cited review is among the reviews the judge named as support. Intervals resample answers, not claims. The abstain column is the judge's own flag and it has been wrong when checked by hand, so it is not the generator's abstention rate. See evaluation/README.md.

Judge stability: 72 claims in 13 answers judged twice with the claims in a different order. Label agreement 87.5%, kappa 0.24. Abstain agreement 100.0%.

Human audit: not yet done. `claim_audit_sample.csv` holds 40 claims stratified by judge label for a reviewer to fill in.
