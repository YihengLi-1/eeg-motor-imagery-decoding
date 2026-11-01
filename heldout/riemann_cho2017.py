"""Strong classical baseline on the frozen Cho2017 held-out splits: shrinkage covariance -> Riemannian tangent space -> logistic regression.
Same cached windows, same subject-level splits (seed -> test 10 / val 5 / train rest) as step3d_matrix.py; transductive steps off
(tangent-space reference and scaler fit on training subjects only); C chosen on validation subjects only.
Two preprocessing conditions, reported separately: broad 0.1-75 Hz (the leaderboard band) and narrow 8-30 Hz (the usual MI band).
Usage: venv/bin/python heldout/riemann_cho2017.py   -> heldout/results_cho2017_riemann.json
"""
import os, json, pickle, time, numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = os.path.expanduser("~/bci-project/heldout")
META = pickle.load(open(f"{ROOT}/_cache_cho2017_meta.pkl", "rb")); clean = META["clean"]
SEEDS = [0, 1, 2, 3]; CS = [0.01, 0.1, 1.0, 10.0]
out = {"protocol": "shrinkage cov (OAS) -> tangent space (ref = train mean) -> StandardScaler -> LR; C by val subjects; windows and splits identical to step3d_matrix.py", "results": {}}
for band in ["broad_0.1-75", "narrow_8-30"]:
    D = pickle.load(open(f"{ROOT}/_cache_cho2017_{band}.pkl", "rb"))
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.RandomState(seed); perm = rng.permutation(clean)
        test_s, val_s, train_s = list(perm[:10]), list(perm[10:15]), list(perm[15:])
        g = lambda subs: (np.concatenate([D[s][0] for s in subs]).astype("float64"), np.concatenate([D[s][1] for s in subs]))
        Xtr, ytr = g(train_s); Xva, yva = g(val_s)
        cov = Covariances(estimator="oas")
        Ctr, Cva = cov.fit_transform(Xtr), cov.transform(Xva)
        ts = TangentSpace(metric="riemann").fit(Ctr)           # reference point from training subjects only
        Ftr, Fva = ts.transform(Ctr), ts.transform(Cva)
        best = None
        for C in CS:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Ftr, ytr)
            va = accuracy_score(yva, clf.predict(Fva))
            if best is None or va > best[0]: best = (va, C, clf)
        va, C, clf = best
        per = []
        for s in test_s:
            Xs, ys = D[s][0].astype("float64"), D[s][1]
            pred = clf.predict(ts.transform(cov.transform(Xs)))
            per.append({"subject": int(s), "acc": float(accuracy_score(ys, pred)), "kappa": float(cohen_kappa_score(ys, pred))})
        macc = float(np.mean([r["acc"] for r in per])); mk = float(np.mean([r["kappa"] for r in per]))
        rows.append({"seed": seed, "mean_acc": macc, "mean_kappa": mk, "val_acc": float(va), "C": C, "per_subject": per, "secs": round(time.time() - t0)})
        print(f"[{band} seed{seed}] test acc={macc*100:.1f}% kappa={mk:+.3f} (val {va*100:.1f}%, C={C}, {rows[-1]['secs']}s)", flush=True)
    a = np.array([r["mean_acc"] for r in rows])
    out["results"][band] = {"acc_mean": float(a.mean()), "acc_std": float(a.std()), "n_seeds": len(a), "runs": rows}
    print(f"== {band}: {a.mean()*100:.1f} ± {a.std()*100:.1f}% over {len(a)} splits", flush=True)
json.dump(out, open(f"{ROOT}/results_cho2017_riemann.json", "w"), indent=2)
print("RIEMANN_DONE")
