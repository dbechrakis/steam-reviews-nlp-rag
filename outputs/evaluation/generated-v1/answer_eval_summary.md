# Answer evaluation, LLM-judged (2026-10-28)

Generator `openai/gpt-oss-120b`. Judge `openai/gpt-oss-20b`. 65 generated answers from the 80-question benchmark (15 questions had no answer because generation failed), judged sentence by sentence against the five reviews the generator saw.

These are LLM judgments. No person has labelled these claims unless the audit section below says so.

| Category | Answers judged | Claims | Supported (95% CI) | Cited claims correct (95% CI) | Abstains (95% CI) | Answers with an unsupported or contradicted claim |
|---|---:|---:|---|---|---|---:|
| all | 36 of 65 | 195 | 86.2% (80 to 92) | 100.0% (100 to 100) | 0.0% (0 to 10) | 1 |
| comparison | 0 of 10 | 0 | unmeasured | unmeasured | unmeasured | 0 |
| game_topic | 23 of 32 | 120 | 84.2% (75 to 91) | 100.0% (100 to 100) | 0.0% (0 to 14) | 1 |
| named_game | 12 of 14 | 73 | 89.0% (82 to 96) | 100.0% (100 to 100) | 0.0% (0 to 24) | 0 |
| unsupported | 1 of 9 | 2 | 100.0% (100 to 100) | unmeasured | 0.0% (0 to 79) | 0 |

Supported counts only the label supported. Partially supported claims are reported in the JSON. A cited claim is correct when it is supported and at least one cited review is among the reviews the judge named as support. Intervals resample answers, not claims. For the unsupported category the abstain rate is the headline and higher is better. For the other categories a high abstain rate means the answer refused although evidence was available.

Judge stability: 178 claims judged twice with the claims in a different order. Label agreement 88.8%, kappa 0.42. Abstain agreement 100.0% over 32 answers.

Human audit: not yet done. `claim_audit_sample.csv` holds 40 claims stratified by judge label for a reviewer to fill in.
