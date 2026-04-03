# NanoHorizon × Synth Managed Research (SMR)

This folder documents how to use **Synth Managed Research** to make **agent-assisted progress** on NanoHorizon (SFT / RLVR / prompt-opt) without replacing the local training scripts. SMR gives you a **managed project**, **Git-backed workspace**, **runs**, **artifacts**, and **usage** in the Synth product.

**Official Synth docs** (install, MCP config, full tool list):

- [Managed Research quickstart](https://docs.usesynth.ai/managed-research/quickstart) (UI + MCP tabs)
- [Managed Research MCP](https://docs.usesynth.ai/managed-research/mcp) (auth, `synth-ai-mcp-managed-research`, Claude/Codex wiring)

---

## 1. What you are setting up

| Step | Goal |
|------|------|
| Synth account + API key | Authenticate MCP and API calls |
| Credits + entitlements | Runs can start without spurious billing blocks; **Codex** runs use Synth-hosted or policy-allowed lanes (see Synth dashboard / plan) |
| MCP | Drive SMR from Claude Code, Codex, or other MCP clients |
| Project spec + files | Tell agents **what** to do (e.g. improve `offline_sft.py`, hit leaderboard constraints) and attach **references** (configs, notes, prior metrics) |

SMR does **not** replace `./scripts/run_offline_training.sh` or the RLVR launcher for **scored leaderboard submissions**. SMR helps you **explore, refactor, and document** toward that submission; the **official way to submit your solution** to NanoHorizon is still a **pull request** that includes your validated record tree (see below).

---

## Submitting your solution (pull request)

When you are ready to enter your result for the competition:

1. Produce a **validated** record under `records/<track>/...` using the repo’s scripts and `validate_record` (see [records/README.md](../../records/README.md)).
2. **Open a pull request** against the NanoHorizon repo (or the fork/remote your track specifies) that contains **your solution**: the record artifacts, any code or config changes you intend to ship, and a short description of what changed.
3. SMR artifacts and Synth run links are **optional context** for you and reviewers; the **PR + validated `records/` layout** is what counts as the submission unless the organizers say otherwise.

Do not rely on SMR or the Synth product alone as “submission” — always **open the PR** with the actual repo changes and records.

---

## 2. Sign up and get an API key

1. Create a **Synth** account at [usesynth.ai](https://usesynth.ai) (or your team’s onboarding link).
2. In the **dashboard**, create or copy an **API key** (`sk_...`). This is what MCP and automation use (`SYNTH_API_KEY`).
3. Optional: set `SYNTH_BACKEND_URL` if you use a **non-default** backend (e.g. local dev). Production default is documented in Synth client docs.

Treat the key like a password — use env vars or your MCP client’s secret store, not committed files.

---

## 3. Credits and Codex / SMR entitlements

Before relying on SMR for real runs:

1. **Credits / plan** — Ensure your org has **spend headroom** (or an active plan) so run creation is not blocked by billing preflight. Check the **Usage / billing** area in the Synth app.
2. **SMR access** — Managed Research should be **enabled** for your org (product flags or plan). If `trigger_run` or onboarding fails with payment or entitlement errors, fix account state in the dashboard first.
3. **Codex usage** — Agent execution often uses **Synth-hosted Codex** or a **connected** lane depending on project policy. Your project **key policy** and org settings determine whether you need extra setup (e.g. ChatGPT connect for `user_connected`). For the simplest path, use defaults that keep you on **Synth-hosted** runs if your org supports it.

Exact labels in the UI may change; the invariant is: **API key works**, **billing/entitlement allows SMR**, **runs reach `running`** rather than failing at create time.

---

## 4. Install MCP and “log in”

Managed Research’s supported automation surface is **MCP** (see Synth docs).

```bash
uv tool install synth-ai
```

Run the SMR MCP server (used by Claude Code / Codex / etc.):

```bash
export SYNTH_API_KEY="sk_..."   # required
# export SYNTH_BACKEND_URL="https://api.usesynth.ai"  # optional override
synth-ai-mcp-managed-research
```

Wire that command into your MCP client with `SYNTH_API_KEY` in `env` (example JSON blocks live in [Managed Research MCP](https://docs.usesynth.ai/managed-research/mcp)).

**Verify:** call `smr_list_projects` (or equivalent) from the client — you should get JSON, not `401`.

---

## 5. Create a project and describe your submission work

Use MCP tools in this order (names match [quickstart](https://docs.usesynth.ai/managed-research/quickstart)):

1. **`smr_create_project`** — name, timezone, basics.
2. **`smr_add_project_repo`** — point at **your fork** of `nanohorizon` (or the canonical repo) so workers can clone/push per org policy.
3. **Starting data** — `smr_get_starting_data_upload_urls` then **`smr_upload_starting_data`** for:
   - a short **intent** file (optional),
   - **reference metrics** or `metrics.json` from a baseline run,
   - track **constraints** (e.g. `offline_20min_1xa100_40gb` budget text),
   - anything else you want in-context for every run.
4. **Project spec (the “prompt”)** — In the **UI onboarding** this is the written spec; via API/MCP use the same fields your client exposes for **research description / goals** (patch project or onboarding steps as documented for your client version). Be explicit, for example:
   - *“Improve mean Craftax reward on track X without breaking validate_record. Edit only `src/nanohorizon/baselines/offline_sft.py` unless justified. Target record layout under `records/...`.”*
5. **Onboarding dry run** — complete **`smr_*` onboarding / dry_run** steps until status is **complete** (exact tool names in Synth docs).
6. **`smr_trigger_run`** — pass **`work_mode`** as required (`open_ended_discovery` vs `directed_effort` per SDK).

---

## 6. During and after a run

- Poll **`smr_get_run`** / **`smr_list_runs`** or use the **Synth web app** for run state, tasks, artifacts, and usage.
- Download reports and diffs from **artifacts** / deliverables when the run finishes.
- Iterate: merge agent work into your fork, re-trigger runs, compare outputs.
- When you have a **scored, validated record**, **open a PR** with your solution (see **Submitting your solution** above) — SMR does not replace that step.

---

## 7. Operator / template path (advanced)

For **repeatable** SMR seeds (explicit `project.research.tasks`), Synth’s **evals / reportbench** tooling uses env like `SMR_REPORT_BENCH_TASK` and a runner — see **`evals/smr/reportbench/NANOHORIZON_SMR_LAUNCH_GUIDE.md`** in the **synth-dev** repo (not shipped inside this nanohorizon clone). That path is aimed at **operators** proving infra, not every competitor.

---

## 8. Checklist (copy-paste)

- [ ] Synth account + **`SYNTH_API_KEY`**
- [ ] Credits / plan allow SMR runs
- [ ] `uv tool install synth-ai` + MCP server configured with the key
- [ ] `smr_list_projects` succeeds
- [ ] Project created, **repo** linked to **your** nanohorizon fork
- [ ] **Starting data** uploaded (references + constraints)
- [ ] **Spec** states track, edit surface, and submission expectations
- [ ] Onboarding **complete**, then **`smr_trigger_run`**
- [ ] When ready to compete: **validated `records/<track>/...`** + **open a PR** with your solution (not only SMR artifacts)

Questions: [Synth Discord](https://discord.gg/cjfAMcCZef) (also linked from the main NanoHorizon README).
