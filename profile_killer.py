import subprocess
import time
import signal
import sys
p = subprocess.Popen(["python", "-m", "cProfile", "-s", "tottime", "run_episode_test.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
output = []
for line in p.stdout:
    sys.stdout.write(line)
    output.append(line)
    if "Step 240" in line:
        time.sleep(5) # wait 5s to profile step 240
        p.send_signal(signal.SIGINT)
        break
for line in p.stdout:
    sys.stdout.write(line)
p.wait()
