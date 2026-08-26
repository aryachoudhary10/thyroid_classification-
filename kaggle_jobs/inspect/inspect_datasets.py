"""Kaggle inspection kernel: map ThyroidXL + TN5000 without downloading anything.

Answers the questions that decide whether patient bags can be built correctly:
  - where are images / masks / labels, and how many of each?
  - what is the filename convention, and can a patient id be derived from it?
  - what do the label files actually contain?
  - do masks pair 1:1 with images?
  - is there a metadata table with patient ids and pathology labels?
  - what does the official TN5000 ImageSets split look like?

Writes /kaggle/working/inspect.json and prints a human-readable report.
CPU only, no internet, ~1 minute.
"""
import collections
import json
import os
import re

ROOT = "/kaggle/input"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TXT_EXT = {".csv", ".tsv", ".json", ".txt", ".xml", ".yaml", ".yml", ".md"}

report = {}


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# --------------------------------------------------------------------------- #
def scan(root):
    per_dir = collections.defaultdict(lambda: collections.Counter())
    samples = collections.defaultdict(list)
    total = 0
    for dp, _dn, fn in os.walk(root):
        if "/.cache/" in dp.replace("\\", "/"):
            continue
        rel = os.path.relpath(dp, root)
        for f in fn:
            ext = os.path.splitext(f)[1].lower()
            per_dir[rel][ext] += 1
            total += 1
            if len(samples[rel]) < 4:
                samples[rel].append(f)
    return per_dir, samples, total


def preview(path, n=1200):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(n)
    except Exception as exc:
        return "<unreadable: %s>" % exc


# --------------------------------------------------------------------------- #
rule("0. WHAT IS MOUNTED")
mounts = sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []
print("mounts:", mounts)
report["mounts"] = mounts

for m in mounts:
    base = os.path.join(ROOT, m)
    rule("1. TREE: " + m)
    per_dir, samples, total = scan(base)
    print("total files (excluding .cache):", total)
    rows = sorted(per_dir.items(), key=lambda kv: -sum(kv[1].values()))
    tree = {}
    for rel, exts in rows[:40]:
        n = sum(exts.values())
        print("  %-52s %7d   %s" % (rel[:52], n, dict(exts.most_common(4))))
        for s in samples[rel][:3]:
            print("        e.g. %s" % s)
        tree[rel] = {"n": n, "ext": dict(exts), "samples": samples[rel]}
    report.setdefault("tree", {})[m] = tree

    # ---- any small text/tabular file gets previewed --------------------- #
    rule("2. TEXT / TABULAR FILES: " + m)
    shown = 0
    for dp, _dn, fn in os.walk(base):
        if "/.cache/" in dp.replace("\\", "/"):
            continue
        for f in sorted(fn):
            ext = os.path.splitext(f)[1].lower()
            if ext not in TXT_EXT:
                continue
            p = os.path.join(dp, f)
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            rel = os.path.relpath(p, base)
            # print every CSV/JSON; only a couple of examples per txt/xml dir
            is_table = ext in (".csv", ".tsv", ".json", ".yaml", ".yml", ".md")
            key = os.path.dirname(rel)
            if not is_table:
                cnt = report.setdefault("_txtseen", {})
                cnt[key] = cnt.get(key, 0) + 1
                if cnt[key] > 2:
                    continue
            if shown > 25:
                continue
            shown += 1
            print("\n--- %s  (%d bytes) ---" % (rel, size))
            print(preview(p, 900))
    report.pop("_txtseen", None)

# --------------------------------------------------------------------------- #
rule("3. THYROIDXL: FILENAME CONVENTION AND PATIENT GROUPING")
tx = [m for m in mounts if "thyroid" in m.lower()]
if tx:
    base = os.path.join(ROOT, tx[0])
    img_dirs, mask_dirs, label_dirs = [], [], []
    for dp, _dn, fn in os.walk(base):
        if "/.cache/" in dp.replace("\\", "/"):
            continue
        low = dp.lower().replace("\\", "/")
        n_img = sum(1 for f in fn if os.path.splitext(f)[1].lower() in IMG_EXT)
        if n_img == 0:
            continue
        if "mask" in low.split("/")[-1]:
            mask_dirs.append((dp, n_img))
        elif "label" in low.split("/")[-1]:
            label_dirs.append((dp, n_img))
        else:
            img_dirs.append((dp, n_img))
    print("image dirs:", [(os.path.relpath(d, base), n) for d, n in img_dirs])
    print("mask  dirs:", [(os.path.relpath(d, base), n) for d, n in mask_dirs])
    print("label dirs (image files):", [(os.path.relpath(d, base), n) for d, n in label_dirs])
    report["thyroidxl_dirs"] = {
        "images": [[os.path.relpath(d, base), n] for d, n in img_dirs],
        "masks": [[os.path.relpath(d, base), n] for d, n in mask_dirs],
        "labels": [[os.path.relpath(d, base), n] for d, n in label_dirs],
    }

    for d, n in img_dirs[:4]:
        names = sorted(f for f in os.listdir(d)
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        print("\n%s : %d images" % (os.path.relpath(d, base), len(names)))
        print("  first 12:", names[:12])
        stems = [os.path.splitext(f)[0] for f in names]

        # try a few patient-id conventions and report how many groups each gives
        cands = {
            "leading digits":      r"^(\d+)",
            "before first _":      r"^([^_]+)_",
            "before first -":      r"^([^-]+)-",
            "alnum prefix + num":  r"^([A-Za-z]*\d+)",
            "strip trailing _N":   r"^(.*?)_\d+$",
            "strip trailing -N":   r"^(.*?)-\d+$",
        }
        res = {}
        for label, pat in cands.items():
            g = [re.match(pat, s).group(1) for s in stems if re.match(pat, s)]
            if not g:
                continue
            c = collections.Counter(g)
            res[label] = {"groups": len(c), "matched": len(g),
                          "frames_per_group_max": max(c.values()),
                          "frames_per_group_mean": round(len(g) / len(c), 2)}
            print("  %-20s -> %5d groups, max %2d frames/group, mean %.2f"
                  % (label, len(c), max(c.values()), len(g) / len(c)))
        report.setdefault("patient_id_candidates", {})[os.path.relpath(d, base)] = res

        # mask pairing
        for md, _mn in mask_dirs:
            if os.path.dirname(md) != os.path.dirname(d):
                continue
            mnames = set(os.path.splitext(f)[0] for f in os.listdir(md))
            inter = len(set(stems) & mnames)
            print("  mask pairing with %s: %d/%d stems match exactly"
                  % (os.path.relpath(md, base), inter, len(stems)))
            if inter < len(stems):
                print("    unmatched examples:", list(set(stems) - mnames)[:5])
                print("    mask name examples:", sorted(mnames)[:5])

    # peek inside label files
    rule("4. THYROIDXL: WHAT IS IN labels/ AND labels_orig/ ?")
    for dp, _dn, fn in os.walk(base):
        if "/.cache/" in dp.replace("\\", "/"):
            continue
        leaf = os.path.basename(dp).lower()
        if "label" not in leaf or not fn:
            continue
        rel = os.path.relpath(dp, base)
        print("\n--- %s : %d files, extensions %s ---"
              % (rel, len(fn),
                 dict(collections.Counter(os.path.splitext(f)[1].lower()
                                          for f in fn).most_common(5))))
        for f in sorted(fn)[:3]:
            p = os.path.join(dp, f)
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                print("  %s  (image file, %d bytes)" % (f, os.path.getsize(p)))
            else:
                print("  %s:\n%s" % (f, preview(p, 400)))

# --------------------------------------------------------------------------- #
rule("5. TN5000: OFFICIAL SPLIT AND VOC ANNOTATIONS")
tn = [m for m in mounts if "thyroid" not in m.lower()]
if tn:
    base = os.path.join(ROOT, tn[0])
    for dp, _dn, fn in os.walk(base):
        if os.path.basename(dp).lower() == "main" and fn:
            print("ImageSets/Main contents:", sorted(fn))
            for f in sorted(fn):
                p = os.path.join(dp, f)
                txt = preview(p, 300)
                lines = [l for l in txt.split("\n") if l.strip()]
                try:
                    total_lines = sum(1 for _ in open(p, encoding="utf-8", errors="ignore"))
                except Exception:
                    total_lines = -1
                print("\n  %s : %d lines" % (f, total_lines))
                print("    first 5:", lines[:5])
                report.setdefault("tn5000_splits", {})[f] = {
                    "lines": total_lines, "head": lines[:5]}
    # one VOC annotation in full
    for dp, _dn, fn in os.walk(base):
        if os.path.basename(dp).lower() == "annotations" and fn:
            f = sorted(fn)[0]
            print("\n--- full VOC example %s ---" % f)
            print(preview(os.path.join(dp, f), 1500))
            break
    # class distribution across all XMLs
    import xml.etree.ElementTree as ET
    names = collections.Counter()
    ann_dir = None
    for dp, _dn, fn in os.walk(base):
        if os.path.basename(dp).lower() == "annotations" and fn:
            ann_dir = dp
            break
    if ann_dir:
        files = sorted(os.listdir(ann_dir))
        for f in files:
            try:
                r = ET.parse(os.path.join(ann_dir, f)).getroot()
                for o in r.findall("object"):
                    names[(o.findtext("name") or "").strip()] += 1
            except Exception:
                pass
        print("\nVOC object <name> distribution across %d files: %s"
              % (len(files), dict(names.most_common(10))))
        report["tn5000_classes"] = dict(names)

# --------------------------------------------------------------------------- #
rule("6. SUMMARY WRITTEN")
with open("/kaggle/working/inspect.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, default=str)
print("wrote /kaggle/working/inspect.json")
print("keys:", sorted(report.keys()))
