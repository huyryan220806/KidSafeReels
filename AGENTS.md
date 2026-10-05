# Instructions for both Codex sessions

- Read README.md, docs/PLAN.md and docs/API_CONTRACT.md first.
- Use Vietnamese for project documentation and user-facing UI; use English identifiers in code.
- This is a two-person, seven-day student project. Prefer small, working increments. CPU inference and optional Colab training are the agreed hardware baseline.
- App owner: apps/, deployment, UI integration tests. ML lead: ml/, data documentation, models documentation, evaluation, reports. Coordinate changes to kidsafe/, dependency files and the API contract in the PR description.
- Work on `feat/app-*` or `feat/ml-*` branches. Keep main runnable. Never overwrite another member's work or force-push shared branches.
- Do not change API field names independently. Update schema, contract, examples and affected consumers together.
- Do not fabricate training runs, accuracy, confidence, timestamps or evidence. Mock results must be explicitly marked. Missing scores are null, not zero. Do not silently fall back from model to mock.
- Split datasets by original video/source group before frame extraction. Keep test data out of model and threshold selection.
- Do not commit user uploads, private datasets, credentials, checkpoints or notebook outputs containing them.
- Run relevant tests with `python -m pytest -q`. Keep docs current with actual behavior and limitations.
- Finish each handoff with changed files, how to run, validation performed and remaining tasks. Do not message another chat automatically.
