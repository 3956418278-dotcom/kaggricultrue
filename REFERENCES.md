# Kaggriculture Reference Index

## Purpose and authority

This is the maintained index of external material used to understand or test the project. It records why each source matters, its exact identity where one exists, and whether anything is retained locally. It is a provenance aid, not a fourth project authority: verified definitions belong in `PROJECT.md`, current conclusions in `STATE.md`, and competitive evidence rules in `EVALUATION.md`.

Official executable source outranks prose when they disagree about game behavior. Public agents, notebooks, and discussions are leads or implementation examples, never rule authority and never evidence that this project's future agent is competitive.

## Official sources

| Source | Exact identity inspected | Useful for | Local retention |
| --- | --- | --- | --- |
| [Kaggriculture competition overview](https://www.kaggle.com/competitions/kaggriculture/overview), especially [How to Play](https://www.kaggle.com/competitions/kaggriculture/overview/how-to-play), [Evaluation](https://www.kaggle.com/competitions/kaggriculture/overview/evaluation), [Getting Started](https://www.kaggle.com/competitions/kaggriculture/overview/getting-started-test-locally-submit), and [FAQ](https://www.kaggle.com/competitions/kaggriculture/overview/frequently-asked-questions) | Page API snapshot retrieved 2026-09-04; selected page hashes are recorded in `references/official/competition-snapshot-2026-09-04.json` | Human-facing game explanation, submission layout, rating/scoring behavior, and remote validation workflow | Compact fact-and-hash snapshot only; raw API response remains disposable |
| [`Kaggle/kaggle-environments`](https://github.com/Kaggle/kaggle-environments/tree/28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c/kaggle_environments/envs/kaggriculture) | PyPI `kaggle-environments==1.32.7`, release source commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`; schema `0.1.0` | Executable rule, observation/action schema, built-in smoke agents, renderer, and local runner | Installed reproducibly; key file hashes retained in `references/official/kaggriculture-environment-1.32.7.json`; no duplicate source copy |
| [Current upstream Kaggriculture source](https://github.com/Kaggle/kaggle-environments/tree/bbda347572cf5134e56f0eb49e8058e2560f9844/kaggle_environments/envs/kaggriculture) | Commit `bbda347572cf5134e56f0eb49e8058e2560f9844`, checked 2026-09-04 | Detecting changes after the packaged release | No copy. The four contract files matched the pinned release byte-for-byte at inspection time |
| [Official getting-started notebook](https://www.kaggle.com/code/bovard/kaggriculture-getting-started) | Kaggle notebook version 4 of 4, observed 2026-09-04 | Minimal local invocation, replay rendering, and submission examples | Link only; the environment package and repository tests are the reproducible owners |

## Selected public references

| Source | Exact identity inspected | Useful for | Caveats and retention |
| --- | --- | --- | --- |
| [`COK-ZhangZiliang/Kaggriculture`](https://github.com/COK-ZhangZiliang/Kaggriculture/tree/7ef67eac458cd9ecd13786063e2e581fbe7403ec) | Commit `7ef67eac458cd9ecd13786063e2e581fbe7403ec` (2026-08-29), Apache-2.0 | Examples of a league runner, replay analysis, submission packaging, evidence manifests, and environment tests | Reference only; no code copied and no full clone retained |
| [`Seyamalam/Kaggriculture`](https://github.com/Seyamalam/Kaggriculture/tree/8b8c421eb10634c756583ce10c75189f50c83a72) | Commit `8b8c421eb10634c756583ce10c75189f50c83a72` (2026-08-05), MIT | Environment-drift notes and examples of tournament, promotion, replay, and contract-test tooling | Pins older environment `1.32.4`; procedural reference only, not current mechanics; no local copy retained |
| [`Beiciccc/Kaggriculture`](https://github.com/Beiciccc/Kaggriculture/tree/932385d2c7afee89aeee4f12ab384468d19c0201) | Commit `932385d2c7afee89aeee4f12ab384468d19c0201` (2026-08-09); no license found at inspection | Example of a concise experiment/evidence ledger | Ideas may be inspected, but material must not be copied without permission or a license; link only |
| [Kaggriculture Evaluation: Agent Performance Analysis](https://www.kaggle.com/code/msama01/kaggriculture-evaluation-agent-performance-analys) | Kaggle notebook version 2, observed 2026-09-04, Apache-2.0 | Example Bradley-Terry and agent-comparison analysis | Method lead only; validate assumptions against `EVALUATION.md`; link only |
| [“X-ray your agent” discussion](https://www.kaggle.com/competitions/kaggriculture/discussion/738563) | Discussion 738563, observed 2026-09-04 | Replay-diagnostic ideas and visual inspection workflow | Informal community material; illustrative, not rule or competitive evidence; link only |

## Retention policy

- Keep compact metadata, exact hashes, and tests when they make a future audit reproducible. The compact competition snapshot preserves selected facts and response hashes, not the discarded raw page prose.
- Prefer the pinned installed distribution and immutable upstream links over copied official source trees.
- Keep temporary page responses, wheels, archives, and full external clones outside the maintained repository.
- Before retaining third-party code or an opponent, review its license, record its origin and exact content hash, and assign it an explicit evaluation role.
- Recheck links and upstream identities when changing the environment pin; never silently transfer evidence across environment versions.
