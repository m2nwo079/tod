import json, time, os, sys, requests
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.environ["OPENALEX_API_KEY"]
BASE = "https://api.openalex.org/works"
TOPIC_ID = "T10462"

def reconstruct_abstract(inv):
    if not inv: return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs: pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))

def fetch_works(filter_str, max_records=4000, sort="cited_by_count:desc"):
    select = ("id,title,publication_year,cited_by_count,"
              "referenced_works,topics,abstract_inverted_index")
    params = {"filter": filter_str, "per_page": 200, "cursor": "*",
              "sort": sort, "api_key": API_KEY, "select": select}
    out = []
    while True:
        r = requests.get(BASE, params=params, timeout=60).json()
        out += r.get("results", [])
        nxt = r.get("meta", {}).get("next_cursor")
        if not nxt or len(out) >= max_records: break
        params["cursor"] = nxt; time.sleep(0.2)
    for w in out:
        w["abstract"] = reconstruct_abstract(w.pop("abstract_inverted_index", None))
    return out[:max_records]

if __name__ == "__main__":
    past = fetch_works(
        f"topics.id:{TOPIC_ID},from_publication_date:2015-01-01,"
        "to_publication_date:2019-12-31,language:en", max_records=4000)
    print("Past slice:", len(past), "works")
    (RAW / "past_works.json").write_text(json.dumps(past))
    print("Saved:", RAW / "past_works.json")