from pathlib import Path

# Project root = the folder that contains this config.py
ROOT = Path(__file__).resolve().parent

RAW      = ROOT / "data" / "raw"
GRAPH    = ROOT / "data" / "graph"
EXCHANGE = ROOT / "exchange"
RESULTS  = ROOT / "results"

for d in (RAW, GRAPH, EXCHANGE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)