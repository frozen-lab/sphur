import re
from pathlib import Path
import sys

def parse_perf_file(path: Path):
    text = path.read_text()
    data = {}

    def grab(label, pat):
        m = re.search(pat, text)
        return float(m.group(1).replace(',', '')) if m else None

    data["task_clock"]     = grab("task-clock", r"([\d,.]+)\s+msec task-clock")
    data["cycles"]         = grab("cycles", r"([\d,]+)\s+cycles")
    data["instructions"]   = grab("instructions", r"([\d,]+)\s+instructions")
    data["branches"]       = grab("branches", r"([\d,]+)\s+branches")
    data["branch_misses"]  = grab("branch-misses", r"([\d,]+)\s+branch-misses")

    if data["cycles"] and data["instructions"]:
        data["ipc"] = data["instructions"] / data["cycles"]
        
    if data["task_clock"] and data["cycles"]:
        data["ghz"] = data["cycles"] / (data["task_clock"] * 1e6)
        
    return data

#
# main
# 

N = float(100_000_000)

base = Path(".local")
files = sorted(base.glob("stat_*.txt"))

if not files:
    print("⚠️ No stat_*.txt files found in .local/")
    sys.exit(1)

rows = []

for f in files:
    name = f.stem.replace("stat_", "")
    d = parse_perf_file(f)
    
    rows.append([
        name,
        f"{d['cycles'] / N:.2f}",
        f"{d['instructions'] / N:.2f}",
        f"{d['ipc']:.2f}",
        f"{d['ghz']:.2f}",
    ])

# Md table
headers = ["Example", "Cycles (per iter)", "Instructions (per iter)", "IPC", "GHz"]
widths = [max(len(r[i]) for r in ([headers] + rows)) for i in range(len(headers))]

def fmt_row(r):
    return "| " + " | ".join(f"{r[i]:<{widths[i]}}" for i in range(len(r))) + " |"

sep = "| " + " | ".join("-" * w for w in widths) + " |"

print(fmt_row(headers))
print(sep)

for r in rows:
    print(fmt_row(r))


