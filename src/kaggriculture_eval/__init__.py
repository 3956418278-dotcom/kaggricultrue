"""Local evaluation and replay infrastructure for Kaggriculture."""

def __getattr__(name):
    # Kaggle tooling/metadata discovery does not need the simulation runtime.
    if name in __all__:
        from . import replay
        return getattr(replay, name)
    raise AttributeError(name)

__all__ = [
    "ReplayError",
    "ReplaySummary",
    "load_replay",
    "render_replay_html",
    "replay_summary",
    "write_replay",
]
