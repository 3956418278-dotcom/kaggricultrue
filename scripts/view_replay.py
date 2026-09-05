#!/usr/bin/env python3
"""Open a Kaggriculture replay in the official animated game visualizer.

Examples:
    .venv/bin/python scripts/view_replay.py replays/game.json
    .venv/bin/python scripts/view_replay.py --current --seed 17 --candidate-seat 0
    .venv/bin/python scripts/view_replay.py game.json --output replays/game.html

The replay JSON is only the input transport. The command presents the actual
Kaggriculture scene with its timeline and controls; it does not display JSON.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

from src.kaggriculture_eval.replay import (  # noqa: E402
    ReplayError,
    load_replay,
    render_replay_html,
    replay_summary,
    write_replay,
)


def _run_current_candidate(seed: int, candidate_seat: int) -> dict[str, object]:
    # Keep ordinary replay viewing independent of the submitted policy. Only
    # the explicit --current path needs to load the candidate entrypoint.
    from main import agent

    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    agents = [agent, "starter"] if candidate_seat == 0 else ["starter", agent]
    environment.run(agents)
    if len(environment.steps) != 720 or any(state.status != "DONE" for state in environment.state):
        statuses = [state.status for state in environment.state]
        raise ReplayError(f"current-candidate game did not complete cleanly: {statuses}")
    return environment.toJSON()


def _print_summary(replay: dict[str, object]) -> None:
    summary = replay_summary(replay)
    print(
        f"Replay: {summary.environment} {summary.environment_version or 'unknown'}; "
        f"kaggle-environments {summary.module_version or 'unknown'}; "
        f"{summary.steps} steps, {summary.agents} agents"
    )
    print(f"Final statuses: {list(summary.statuses)}")
    print(f"Final rewards: {list(summary.rewards)}")
    for warning in summary.warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def _serve(html: str, host: str, port: int, open_browser: bool) -> None:
    document = html.encode("utf-8")

    class ReplayHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.split("?", 1)[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(document)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(document)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers and command-line probes may stop reading once they
                # have enough of this large single-file document.
                return

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), ReplayHandler)
    bound_host, bound_port = server.server_address[:2]
    browser_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
    url = f"http://{browser_host}:{bound_port}/"
    print(f"Official Kaggriculture visualizer: {url}")
    print("Press Ctrl-C to stop the local viewer.")
    if open_browser and not webbrowser.open(url):
        print("A browser could not be opened automatically; use the URL above.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View an official replay or a fresh current-candidate game as the real Kaggriculture scene."
    )
    parser.add_argument("replay", nargs="?", type=Path, help="official Environment.toJSON replay file")
    parser.add_argument(
        "--current",
        action="store_true",
        help="run the current main.py candidate against the built-in starter first",
    )
    parser.add_argument("--seed", type=int, default=17, help="seed used with --current (default: 17)")
    parser.add_argument(
        "--candidate-seat",
        type=int,
        choices=(0, 1),
        default=0,
        help="current candidate seat used with --current (default: 0)",
    )
    parser.add_argument(
        "--save-replay",
        type=Path,
        help="save a --current game's underlying official replay (normally below replays/)",
    )
    parser.add_argument(
        "--initial-step",
        type=int,
        default=0,
        help="timeline step initially displayed (default: 0)",
    )
    parser.add_argument("--autoplay", action="store_true", help="start timeline playback immediately")
    parser.add_argument(
        "--output",
        type=Path,
        help="write a single HTML viewer and exit instead of serving it",
    )
    parser.add_argument("--host", default="127.0.0.1", help="local viewer host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="local viewer port; 0 chooses a free port")
    parser.add_argument("--no-open", action="store_true", help="do not try to open a browser")
    args = parser.parse_args()
    if bool(args.replay) == bool(args.current):
        parser.error("provide exactly one replay path or --current")
    if args.save_replay and not args.current:
        parser.error("--save-replay is only valid with --current")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    return args


def main() -> int:
    args = _parse_args()
    try:
        if args.current:
            print(
                f"Running current candidate vs starter: seed={args.seed}, "
                f"candidate seat={args.candidate_seat} ..."
            )
            replay = _run_current_candidate(args.seed, args.candidate_seat)
            if args.save_replay:
                replay_path = write_replay(replay, args.save_replay)
                print(f"Replay saved: {replay_path.resolve()}")
        else:
            replay = load_replay(args.replay)

        _print_summary(replay)
        html = render_replay_html(
            replay,
            initial_step=args.initial_step,
            autoplay=args.autoplay,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(html, encoding="utf-8")
            uri = args.output.resolve().as_uri()
            print(f"HTML viewer written: {args.output.resolve()}")
            if not args.no_open and not webbrowser.open(uri):
                print("A browser could not be opened automatically; open the file above.", file=sys.stderr)
            return 0

        _serve(html, args.host, args.port, not args.no_open)
        return 0
    except ReplayError as error:
        print(f"Replay error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Viewer error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
