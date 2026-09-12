import sys
import runpy

import src.kaggriculture_agent.planner as planner
import time

old_loop = planner._long_candidate_loop
def wrapped_loop(state, programme, demand, pressure):
    print("STARTING LONG CANDIDATE LOOP", flush=True)
    t0 = time.perf_counter()
    res = old_loop(state, programme, demand, pressure)
    t1 = time.perf_counter()
    print(f"LONG CANDIDATE LOOP TOOK {t1-t0:.4f}s", flush=True)
    if t1-t0 > 5:
        print("EXITING TO PREVENT TIMEOUT", flush=True)
        sys.exit(0)
    return res

planner._long_candidate_loop = wrapped_loop
runpy.run_path("run_episode_test.py", run_name="__main__")
