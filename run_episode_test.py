import time
from kaggle_environments import make
from src.kaggriculture_agent.operating import DailyPlanningSession
from src.kaggriculture_agent.agent import decide

print("Starting D11+ single episode benchmark...", flush=True)
planning_session = DailyPlanningSession()
env = make("kaggriculture", configuration={"seed": 11}, debug=True)
obs = env.reset(2)[0].observation

step_times = []
def agent_wrapper(observation):
    start = time.perf_counter()
    action = decide(observation, session=planning_session)
    elapsed = time.perf_counter() - start
    step_times.append(elapsed)
    if observation.step % 24 == 0:
        day = observation.step // 24
        print(f"Step {observation.step} (Day {day}): decide took {elapsed:.4f}s", flush=True)
    return action

print("Running episode...", flush=True)
start_time = time.perf_counter()
env.run([agent_wrapper, "starter"])
total_time = time.perf_counter() - start_time

print(f"\nEpisode completed cleanly in {total_time:.2f} seconds.", flush=True)
print(f"Final rewards: {[state.reward for state in env.state]}", flush=True)
print(f"Planning session calls count: {len(planning_session.planning_diagnostics)}", flush=True)

for row in planning_session.planning_diagnostics:
    print(f"Day {row.get('day', '?')}: step={row.get('step')}, time={row.get('seconds', 0):.4f}s, reason={row.get('reason')}", flush=True)

max_replan = max((row.get('seconds', 0) for row in planning_session.planning_diagnostics), default=0)
avg_replan = sum(row.get('seconds', 0) for row in planning_session.planning_diagnostics) / max(1, len(planning_session.planning_diagnostics))
print(f"\nSummary stats:", flush=True)
print(f"Max replan time: {max_replan:.4f}s", flush=True)
print(f"Avg replan time: {avg_replan:.4f}s", flush=True)
