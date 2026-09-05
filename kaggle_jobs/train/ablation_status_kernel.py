"""Ground truth for the ConvNeXt-Tiny ablation run (lesion_mil, mr_mil).

CPU-only, chained from dermil-convnext-ablation. The full-size download of
that session truncated (a 0-byte last.pt on lesion_mil/fold3, no fold4, no
mr_mil at all) -- but truncation has hit every large-output kernel run so far,
so a partial local download is not evidence of what happened server-side. This
reads the registry and any result tables directly from the chained input and
prints the whole thing, unfiltered, then re-emits only the small files.
"""
import json
import os
import shutil
import sys
import time

T0 = time.time()


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


INPUT, WORK = "/kaggle/input", "/kaggle/working"
OUT = os.path.join(WORK, "status")
os.makedirs(OUT, exist_ok=True)


def find_dir(root, *needles):
    hits = []
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            p = os.path.join(dp, d)
            if all(n.lower() in p.lower() for n in needles):
                hits.append(p)
    return hits


# =========================================================================== #
banner("REGISTRY -- every stage for the convnexttiny run, unfiltered")

found_registry = False
for dp, _dn, fn in os.walk(INPUT):
    if "registry.json" in fn:
        src = os.path.join(dp, "registry.json")
        try:
            reg = json.load(open(src, encoding="utf-8"))
        except Exception as e:
            print("  %s unreadable: %s" % (src, e))
            continue
        keys = sorted(k for k in reg if "convnexttiny" in k
                      or "lesion_mil" in k or "mr_mil" in k)
        if not keys:
            continue
        found_registry = True
        print("\nsource: %s" % src)
        for k in keys:
            art = reg[k]
            print("  %-42s %s  %s" % (k, art.get("finished_at"),
                                      {kk: vv for kk, vv in art.items()
                                       if kk != "finished_at"}))
        shutil.copyfile(src, os.path.join(OUT, "registry_%d.json" % (hash(dp) % 10000)))

if not found_registry:
    print("no registry.json with convnexttiny/lesion_mil/mr_mil keys found in "
          "the chained input -- the training kernel may not have written one "
          "yet, or it lives somewhere this walk did not reach.")

# =========================================================================== #
banner("RESULT TABLES -- anything the training kernel wrote for these models")

res_roots = [p for p in find_dir(INPUT, "results") if os.path.isdir(p)]
print("results roots seen:", res_roots or "NONE")

import pandas as pd  # noqa: E402

seen_any = False
for root in res_roots:
    for dp, _dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root)
        if "lesion_mil" not in rel and "mr_mil" not in rel and "_tables" not in rel:
            continue
        for f in sorted(fn):
            src = os.path.join(dp, f)
            dst_name = os.path.relpath(src, root).replace(os.sep, "__")
            try:
                shutil.copyfile(src, os.path.join(OUT, dst_name))
            except Exception:
                pass
            seen_any = True
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(src)
                    print("\n--- %s (%d rows)" % (os.path.relpath(src, root), len(df)))
                    print(df.head(20).to_string(index=False))
                except Exception as e:
                    print("  could not read %s: %s" % (src, e))
            elif f.endswith(".json"):
                try:
                    print("\n--- %s" % os.path.relpath(src, root))
                    print(json.dumps(json.load(open(src, encoding="utf-8")), indent=2)[:800])
                except Exception as e:
                    print("  could not read %s: %s" % (src, e))

if not seen_any:
    print("no lesion_mil / mr_mil result files exist yet under any results/ root.")
    print("Combined with the registry section above, this tells us whether "
          "training genuinely has not reached these outputs, or whether they "
          "exist but this walk missed them.")

# =========================================================================== #
banner("CHECKPOINT INVENTORY -- what actually got saved, regardless of results/")

for cdir in find_dir(INPUT, "checkpoints"):
    for model in ("lesion_mil", "mr_mil"):
        mp = os.path.join(cdir, "convnexttiny", model)
        if not os.path.isdir(mp):
            continue
        print("\n%s:" % mp)
        for sub in sorted(os.listdir(mp)):
            sp = os.path.join(mp, sub)
            if not os.path.isdir(sp):
                continue
            best = os.path.join(sp, "best.pt")
            last = os.path.join(sp, "last.pt")
            meta = os.path.join(sp, "meta.json")
            bsz = os.path.getsize(best) if os.path.exists(best) else -1
            lsz = os.path.getsize(last) if os.path.exists(last) else -1
            m = {}
            if os.path.exists(meta):
                try:
                    m = json.load(open(meta, encoding="utf-8"))
                except Exception:
                    pass
            print("  %-10s best=%10d  last=%10d  epoch=%s  best_metric=%s"
                  % (sub, bsz, lsz, m.get("epoch"), m.get("best_metric")))

print("\nsmall outputs written to %s (%d files)" % (OUT, len(os.listdir(OUT))))
banner("DONE in %.1f min" % ((time.time() - T0) / 60.0))
