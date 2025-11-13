# GenAI-Ops (separate repo)

This repo contains a GenAI Ops pipeline with DVC stages, tests and GitHub Actions for CI and Continuous Training (CT).

Quick start:
1. Create repo locally and paste files.
2. `git init && git add . && git commit -m "initial"`
3. Add GitHub remote and push.
4. Add GitHub repo secrets: `DVC_REMOTE_URL`, `RUN_FINETUNE`, `HF_API_TOKEN`, `OPENAI_API_KEY` as needed.
5. (Optional) Register self-hosted runners labeled `gpu` for CT finetune.
6. Run CI by opening a PR; CT runs daily or manually via workflow_dispatch.

See src/ for scripts and tests/ for pytest tests.
