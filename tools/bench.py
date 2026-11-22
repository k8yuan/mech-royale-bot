"""Parallel benchmark: python3 tools/bench.py module [n_per_job] [jobs_per_hunt] -> mean score per hunt mode."""
import subprocess, sys, re
mod = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
procs = []
for h in ("normal", "fast"):
    for j in range(jobs):
        seed = 5000 + j * 1000
        procs.append((h, subprocess.Popen([sys.executable, "tools/sim.py", str(n), str(seed), "--bot", mod, "--hunt", h],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)))
res = {}
early = {}
for h, p in procs:
    out = p.communicate()[0]
    for line in out.splitlines():
        m = re.match(r"seed \d+ place\s+(\d+) died@\s*(\d+)", line)
        if m:
            res.setdefault(h, []).append(16 - int(m.group(1)))
            if int(m.group(2)) < 60 and int(m.group(1)) > 1:
                early[h] = early.get(h, 0) + 1
for h, v in res.items():
    print(f"{mod} {h}: mean {sum(v)/len(v):.2f} n={len(v)} early_deaths={early.get(h,0)}")
