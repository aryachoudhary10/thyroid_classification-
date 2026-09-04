"""Recover the arm ladder from dermil-arms, and audit TN5000 for patient leakage.

Two jobs, both CPU-only and both inference-free.

1. The dermil-arms session finished, but `kaggle kernels output` returned only
   its checkpoint tree -- ~1.8 GB of adapter weights -- and truncated away
   results/ and registry.json. The numbers exist on Kaggle; they just will not
   come down alongside the weights. This kernel chains from that session, reads
   the results it wrote, prints the ladder, and re-emits ONLY the small files so
   the next fetch is a few MB and completes.

2. Every TN5000 number in this project rests on the assumption that the official
   train/val/test split does not put the same patient on both sides. The mirror
   ships no patient column -- patient_id is the filename stem -- so if one
   examination contributed several frames spread across the split, all of our
   external results are inflated, including the unexplained RCAF 0.960. This
   audit looks for that directly, by perceptual hash: near-duplicate images
   whose split labels differ.
"""
import os
import shutil
import sys
import time

T0 = time.time()


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


INPUT, WORK = "/kaggle/input", "/kaggle/working"
CODE = os.environ.get("DERMIL_CODE")
if CODE and CODE not in sys.path:
    sys.path.insert(0, CODE)

RUN = os.environ.get("DERMIL_RUN", "hires")
OUT = os.path.join(WORK, "report")
os.makedirs(OUT, exist_ok=True)

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402


def find_dir(root, *needles):
    hits = []
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            p = os.path.join(dp, d)
            if all(n.lower() in p.lower() for n in needles):
                hits.append(p)
    return hits


# =========================================================================== #
banner("1. ARM LADDER  (recovered from the chained dermil-arms session)")

res_roots = [p for p in find_dir(INPUT, "results") if os.path.isdir(p)]
print("results roots seen:", res_roots or "NONE")

found_any = False
for root in res_roots:
    for dp, _dn, fn in os.walk(root):
        for f in sorted(fn):
            if not f.endswith((".csv", ".json")):
                continue
            if not any(k in f for k in ("tn5000", "arm", "labelfree", "retrieval")):
                continue
            src = os.path.join(dp, f)
            rel = os.path.relpath(src, root).replace(os.sep, "__")
            try:
                shutil.copyfile(src, os.path.join(OUT, rel))
            except Exception:
                pass
            found_any = True
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(src)
                    print("\n--- %s" % os.path.relpath(src, root))
                    print(df.to_string(index=False))
                except Exception as e:
                    print("  could not read %s: %s" % (src, e))

if not found_any:
    print("no TN5000 result tables found in the chained input.")
    print("Chain from choudhary15/dermil-arms and re-run.")

# Registry tells us which stages actually completed, even if a table is missing.
for dp, _dn, fn in os.walk(INPUT):
    if "registry.json" in fn:
        src = os.path.join(dp, "registry.json")
        try:
            import json
            reg = json.load(open(src, encoding="utf-8"))
            keys = sorted(k for k in reg if "tn5000" in k or "seg/" in k)
            if keys:
                print("\ncompleted TN5000 / segmenter stages (%s):" % src)
                for k in keys:
                    print("   %-46s %s" % (k, reg[k].get("finished_at")))
                shutil.copyfile(src, os.path.join(OUT, "registry.json"))
        except Exception as e:
            print("  registry unreadable: %s" % e)

# =========================================================================== #
banner("2. TN5000 PATIENT-LEAKAGE AUDIT")

import cv2                                                      # noqa: E402

TN = None
for cand in ("main data", "main-data"):
    hits = find_dir(INPUT, cand)
    if hits:
        TN = hits[0]
        break
print("tn5000 root:", TN)


def dhash(path, size=8):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits(im[:, 1:] > im[:, :-1]).tobytes()


if TN:
    from src.data.tn5000 import build_tn5000_manifest as _bm
    try:
        man = _bm(TN)
    except Exception:
        import traceback
        traceback.print_exc()
        man = pd.DataFrame()

    if len(man):
        rows = []
        for _i, r in man.iterrows():
            for p in r["image_paths"]:
                if p:
                    rows.append({"path": p, "split": r.get("split"),
                                 "label": r.get("label"),
                                 "pid": str(r["patient_id"])})
        frames = pd.DataFrame(rows)
        print("frames: %d | splits: %s"
              % (len(frames), frames["split"].value_counts().to_dict()))

        print("\ncomputing perceptual hashes ...")
        hs, keep = [], []
        for i, p in enumerate(frames["path"].tolist()):
            h = dhash(p)
            if h is not None:
                hs.append(np.frombuffer(h, dtype=np.uint8))
                keep.append(i)
            if (i + 1) % 1000 == 0:
                print("   %d/%d" % (i + 1, len(frames)), flush=True)
        H = np.vstack(hs)
        F = frames.iloc[keep].reset_index(drop=True)
        print("hashed %d images" % len(F))

        # Exact-duplicate hashes first: cheap and unambiguous.
        key = [h.tobytes() for h in H]
        F["h"] = key
        dup = F.groupby("h").filter(lambda g: len(g) > 1)
        n_groups = dup["h"].nunique() if len(dup) else 0
        print("\nexact duplicate-hash groups: %d (%d images)"
              % (n_groups, len(dup)))
        cross = 0
        if len(dup):
            for _h, g in dup.groupby("h"):
                if g["split"].nunique() > 1:
                    cross += 1
            print("  ...of which span more than one split: %d" % cross)

        # Near-duplicates across the split boundary are the actual risk.
        bits = np.unpackbits(H, axis=1).astype(np.int8)
        val_idx = np.where(F["split"].values == "val")[0]
        tr_idx = np.where(F["split"].values == "train")[0]
        print("\nnear-duplicate scan: %d val vs %d train" % (len(val_idx), len(tr_idx)))
        THRESH = 5                       # Hamming distance over 64 bits
        near, worst = 0, []
        B = bits[tr_idx]
        for j, vi in enumerate(val_idx):
            d = np.abs(B - bits[vi]).sum(axis=1)
            m = int(d.min()) if len(d) else 64
            if m <= THRESH:
                near += 1
                worst.append((m, F.iloc[vi]["path"],
                              F.iloc[tr_idx[int(d.argmin())]]["path"]))
            if (j + 1) % 100 == 0:
                print("   %d/%d" % (j + 1, len(val_idx)), flush=True)

        frac = near / max(len(val_idx), 1)
        print("\nval images with a near-duplicate in train (Hamming <= %d): "
              "%d / %d = %.1f%%" % (THRESH, near, len(val_idx), 100 * frac))
        for d, a, b in sorted(worst)[:10]:
            print("   d=%d  %s  ~  %s"
                  % (d, os.path.basename(a), os.path.basename(b)))

        pd.DataFrame(worst, columns=["hamming", "val_image", "train_image"]).to_csv(
            os.path.join(OUT, "tn5000_near_duplicates.csv"), index=False)

        print("""
Verdict
-------
0%%          the split is clean; the external numbers stand on this axis.
< 1%%        incidental; note it and move on.
> 5%%        the official split leaks. Every TN5000 result in the project,
            including the unexplained RCAF 0.960, is inflated and the arms
            should be re-evaluated on a grouped split.
""")
else:
    print("TN5000 not attached -- skipping the audit")

print("\nsmall outputs written to %s (%d files)"
      % (OUT, len(os.listdir(OUT))))
banner("DONE in %.1f min" % ((time.time() - T0) / 60.0))
