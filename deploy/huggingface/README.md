---
title: Steam Review Retrieval API
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Steam review search API: FAISS retrieval + reranking
---

# Steam Review Retrieval API

Interactive documentation for the retrieval service from
[dbechrakis/steam-reviews-nlp-rag](https://github.com/dbechrakis/steam-reviews-nlp-rag).
This Space is rebuilt from that repository whenever the service changes.

- `/docs`: try `POST /search` with a question such as "a relaxing farming game"
- `/games`: the 241 games and their review counts
- `/health`: corpus size, device, cache statistics

The first request after a period of inactivity is slow while the models load. AI-written
answers are switched off in this public demo, so `/answer` returns the retrieved reviews only.
Portfolio demonstration; it has no authentication.
