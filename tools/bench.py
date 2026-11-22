"""Parallel benchmark: python3 tools/bench.py module [n_per_job] [jobs_per_hunt] -> mean score per hunt mode."""
import subprocess, sys, re
extra = [a for a in sys.argv[1:] if a.startswith("--")]
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--intel" in extra:
    pass
mod = pos[0]
n = int(pos[1]) if len(pos) > 1 else 24
jobs = int(pos[2]) if len(pos) > 2 else 2
intel = pos[3] if len(pos) > 3 else "omni"
hunts = ("normal",) if "--normal" in extra else (("squad",) if "--squad" in extra else ("normal", "fast"))
procs = []
for h in hunts:
    for j in range(jobs):
        seed = 5000 + j * 1000
        procs.append((h, subprocess.Popen([sys.executable, "tools/sim.py", str(n), str(seed), "--bot", mod, "--hunt", h, "--intel", intel] + extra,
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
