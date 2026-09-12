import cProfile
import io
import pstats
from run_episode_test import env, get_agent, Agent
from kaggriculture_agent.observation import observation_to_state

print("Loading test state")
agent = get_agent()
obs = env.reset()
for _ in range(10):
    action = agent(obs)
    obs, reward, done, info = env.step(action)

state = observation_to_state(obs)
print(f"Profiling decide at step {state.step}...")

profiler = cProfile.Profile()
profiler.enable()
action = agent(obs)
profiler.disable()

s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
ps.print_stats(30)
with open('profile_stats.txt', 'w') as f:
    f.write(s.getvalue())
print("Profile saved to profile_stats.txt")
