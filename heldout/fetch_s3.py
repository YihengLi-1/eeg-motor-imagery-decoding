"""Parallel OpenNeuro S3 fetch via bucket listing (public datasets whose files are exported to S3).
Usage: python3 -I fetch_s3.py <dsid> [threads]  -> ~/mne_data/live/<dsid>/...
"""
import os, re, sys, urllib.request, concurrent.futures as cf
ds = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 12
EXCLUDE = sys.argv[3].split(",") if len(sys.argv) > 3 else []   # top-level folders to skip, e.g. sourcedata,derivatives
root = os.path.expanduser(f"~/mne_data/live/{ds}")
keys, token = [], None
while True:
    u = f"https://s3.amazonaws.com/openneuro.org?list-type=2&prefix={ds}/" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
    import urllib.parse
    x = urllib.request.urlopen(u, timeout=120).read().decode()
    keys += [(k, int(s)) for k, s in zip(re.findall(r"<Key>(.*?)</Key>", x), re.findall(r"<Size>(\d+)</Size>", x))]
    m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
    if not m: break
    token = m.group(1)
keys = [(k, s) for k, s in keys if not any(k.startswith(f"{ds}/{e}/") for e in EXCLUDE)]
print(f"{ds}: {len(keys)} objects, {sum(s for _, s in keys)/1e9:.2f} GB", flush=True)
def get(item):
    k, size = item
    dst = os.path.join(root, k[len(ds) + 1:])
    if os.path.exists(dst) and os.path.getsize(dst) == size: return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(5):
        try:
            with urllib.request.urlopen("https://s3.amazonaws.com/openneuro.org/" + urllib.parse.quote(k), timeout=900) as r, open(dst + ".part", "wb") as w:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    w.write(b)
            if os.path.getsize(dst + ".part") == size:
                os.replace(dst + ".part", dst); return "ok"
        except Exception as e:  # noqa: BLE001
            err = e
    return f"fail {k}: {err}"
with cf.ThreadPoolExecutor(T) as ex:
    res = list(ex.map(get, keys))
bad = [r for r in res if r.startswith("fail")]
print(f"ok={res.count('ok')} skip={res.count('skip')} bad={len(bad)}"); [print("  ", b) for b in bad[:10]]
print(f"FETCH_DONE {ds}" if not bad else f"FETCH_INCOMPLETE {ds}")
