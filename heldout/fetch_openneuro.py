"""Download one OpenNeuro snapshot via the public GraphQL file listing + S3 URLs (no extra deps).
Usage: python3 -I fetch_openneuro.py <dsid> <tag> [dest_root]   -> dest_root/<dsid>/...
Skips files already present with the right size; prints FETCH_DONE <dsid> at the end.
"""
import json, os, sys, urllib.request, concurrent.futures as cf

API = "https://openneuro.org/crn/graphql"

def gql(q):
    req = urllib.request.Request(API, data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["data"]

def listing(ds, tag, tree=None, prefix=""):
    t = f', tree: "{tree}"' if tree else ""
    files = gql(f'{{ snapshot(datasetId: "{ds}", tag: "{tag}") {{ files(tree: {json.dumps(tree) if tree else "null"}) {{ id filename size directory urls }} }} }}')["snapshot"]["files"]
    out = []
    for f in files:
        path = prefix + f["filename"]
        if f["directory"]:
            out += listing(ds, tag, f["id"], path + "/")
        else:
            out.append((path, f["size"], (f["urls"] or [None])[0]))
    return out

def fetch(item, root):
    path, size, url = item
    dst = os.path.join(root, path)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "skip"
    if not url:
        return "nourl " + path
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(dst + ".part", "wb") as w:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    w.write(b)
            os.replace(dst + ".part", dst)
            return "ok"
        except Exception as e:  # noqa: BLE001
            err = e
    return f"fail {path}: {err}"

if __name__ == "__main__":
    ds, tag = sys.argv[1], sys.argv[2]
    root = os.path.join(sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/mne_data/live"), ds)
    items = listing(ds, tag)
    total = sum(s for _, s, _ in items)
    print(f"{ds} {tag}: {len(items)} files, {total/1e9:.2f} GB -> {root}", flush=True)
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda it: fetch(it, root), items))
    bad = [r for r in res if r.startswith(("fail", "nourl"))]
    print(f"ok={res.count('ok')} skip={res.count('skip')} bad={len(bad)}")
    for b in bad[:20]: print("  ", b)
    print(f"FETCH_DONE {ds}" if not bad else f"FETCH_INCOMPLETE {ds}")
