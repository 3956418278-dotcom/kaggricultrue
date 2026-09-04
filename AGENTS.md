# Kaggriculture agent entry point

Follow [`.agents/core.md`](.agents/core.md) for behavior that is always active.

Repository skills under `.agents/skills/` provide procedures when their descriptions match the work:

- use `repository-orientation` when current state or ownership is unclear;
- use `structural-implementation` for substantive code, configuration, interface, or ownership changes;
- use `debugging` to establish the cause of failures or inconsistent results;
- use `evidence-validation` to decide whether checks support an important claim;
- use `independent-review` when a consequential result needs a fresh challenge;
- use `kaggriculture-agent-evidence` for strategy hypotheses, controlled arenas, replay analysis, opponent comparisons, or competitive-performance claims.

Skills are selected by the task; they are not a mandatory sequence.

## Project authority

- [`PROJECT.md`](PROJECT.md) owns the stable competition definition, submission contract, environment semantics, agent decomposition, and architectural boundaries.
- [`STATE.md`](STATE.md) owns the accepted baseline, implemented capabilities, accepted competitive evidence, current decisions, limiting uncertainties, and next meaningful work.
- [`EVALUATION.md`](EVALUATION.md) owns environment identity and the protocol by which local or Kaggle results may support competitive claims.
- [`REFERENCES.md`](REFERENCES.md) indexes external sources and their provenance. It supports the three authorities above and never overrides them.

Read the relevant authority before changing its contract. Update these documents only when the knowledge they own changes; do not use them as activity logs.

## Repository boundaries

- Keep submission behavior in one maintained agent path: `main.py` is the eventual Kaggle entrypoint, while reusable implementation belongs under `src/kaggriculture_agent/`. Do not duplicate strategy in runners or notebooks.
- Keep local arena, packaging, and replay-analysis mechanisms outside the submitted policy, with thin commands under `scripts/`, tests under `tests/`, and experiment instances in configuration rather than copied code.
- Treat `PROJECT.md`, `EVALUATION.md`, and `REFERENCES.md` as project-body documentation. Treat `STATE.md` as the durable current-state record. Keep raw replays, match outputs, downloaded opponents, caches, credentials, and built submissions in ignored data/cache areas such as `runs/`, `replays/`, `.cache/`, and `dist/`; they are not maintained authority.
- Preserve external opponent provenance and exact hashes. Do not commit third-party agents or competition data unless their license and project role have been explicitly reviewed.

Preserve unrelated working-tree state and keep secrets outside the repository.
