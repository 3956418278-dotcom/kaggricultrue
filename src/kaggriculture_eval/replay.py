"""Load official Kaggriculture replays and render the official game scene."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Mapping

from kaggle_environments import make


ENVIRONMENT_NAME = "kaggriculture"
ENVIRONMENT_VERSION = "0.1.0"


class ReplayError(ValueError):
    """Raised when a file is not a usable Kaggriculture replay."""


@dataclass(frozen=True)
class ReplaySummary:
    """Small, inspectable identity summary for a loaded replay."""

    environment: str
    environment_version: str | None
    module_version: str | None
    local_module_version: str
    steps: int
    agents: int
    statuses: tuple[str, ...]
    rewards: tuple[float | int | None, ...]
    warnings: tuple[str, ...]


def _normalized_replay(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayError("replay root must be a JSON object")

    # A saved ``window.kaggle`` payload wraps the ordinary Environment.toJSON
    # replay under ``environment``. Accepting it is useful when debugging a
    # previously rendered page and does not introduce another replay format.
    wrapped = value.get("environment")
    if "steps" not in value and isinstance(wrapped, Mapping):
        value = wrapped

    replay = dict(value)
    environment_name = replay.get("name")
    if environment_name != ENVIRONMENT_NAME:
        raise ReplayError(
            f"expected environment {ENVIRONMENT_NAME!r}, got {environment_name!r}"
        )

    steps = replay.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ReplayError("replay must contain a non-empty 'steps' list")
    for index, step in enumerate(steps):
        if not isinstance(step, list) or not step:
            raise ReplayError(f"replay step {index} must be a non-empty agent-state list")
        for agent_index, state in enumerate(step):
            if not isinstance(state, Mapping):
                raise ReplayError(
                    f"replay step {index}, agent {agent_index} state must be an object"
                )

    for key in ("configuration", "info"):
        if key in replay and not isinstance(replay[key], Mapping):
            raise ReplayError(f"replay field {key!r} must be an object")

    return replay


def load_replay(path: str | Path) -> dict[str, Any]:
    """Read an official ``Environment.toJSON()`` replay from disk."""

    replay_path = Path(path)
    try:
        with replay_path.open(encoding="utf-8") as replay_file:
            value = json.load(replay_file)
    except OSError as error:
        raise ReplayError(f"could not read replay {replay_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ReplayError(
            f"replay {replay_path} is not valid JSON: line {error.lineno}, column {error.colno}"
        ) from error
    return _normalized_replay(value)


def replay_summary(replay: Mapping[str, Any]) -> ReplaySummary:
    """Describe replay identity and flag renderer-contract mismatches."""

    normalized = _normalized_replay(replay)
    steps = normalized["steps"]
    final_step = steps[-1]
    local_module_version = version("kaggle-environments")
    environment_version = normalized.get("version")
    module_version = normalized.get("module_version")
    warnings: list[str] = []

    if environment_version is None:
        warnings.append("replay does not record an environment schema version")
    elif environment_version != ENVIRONMENT_VERSION:
        warnings.append(
            f"replay schema {environment_version} differs from local {ENVIRONMENT_VERSION}"
        )
    if module_version is None:
        warnings.append("replay does not record a kaggle-environments package version")
    elif module_version != local_module_version:
        warnings.append(
            f"replay package {module_version} differs from local {local_module_version}"
        )

    statuses = tuple(str(state.get("status", "")) for state in final_step)
    rewards = tuple(state.get("reward") for state in final_step)
    return ReplaySummary(
        environment=normalized["name"],
        environment_version=environment_version,
        module_version=module_version,
        local_module_version=local_module_version,
        steps=len(steps),
        agents=len(final_step),
        statuses=statuses,
        rewards=rewards,
        warnings=tuple(warnings),
    )


def render_replay_html(
    replay: Mapping[str, Any],
    *,
    initial_step: int = 0,
    autoplay: bool = False,
) -> str:
    """Render replay states with the official installed Kaggriculture UI."""

    normalized = _normalized_replay(replay)
    step_count = len(normalized["steps"])
    if not 0 <= initial_step < step_count:
        raise ReplayError(
            f"initial step must be between 0 and {step_count - 1}, got {initial_step}"
        )

    try:
        environment = make(
            ENVIRONMENT_NAME,
            configuration=dict(normalized.get("configuration", {})),
            info=dict(normalized.get("info", {})),
            steps=normalized["steps"],
            debug=False,
        )
        html = environment.render(
            mode="html",
            controls=True,
            playing=autoplay,
            step=initial_step,
        )
    except Exception as error:
        raise ReplayError(f"official environment could not load the replay: {error}") from error
    if not isinstance(html, str) or not html.strip():
        raise ReplayError("official Kaggriculture HTML renderer returned no scene")
    return html


def write_replay(replay: Mapping[str, Any], path: str | Path) -> Path:
    """Persist an official replay, creating only its explicitly named parent."""

    normalized = _normalized_replay(replay)
    replay_path = Path(path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    with replay_path.open("w", encoding="utf-8") as replay_file:
        json.dump(normalized, replay_file, separators=(",", ":"))
    return replay_path
