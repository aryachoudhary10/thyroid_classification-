"""Kaggle kernel 2: pin down the ThyroidXL label encoding and reproduce Table 1.

Kernel 1 established the layout. The open questions this one closes:

  - what does `conclusion` (1/2/3/...) actually mean?
  - which mapping reproduces the paper's 877/3354 dev and 353/739 test malignant?
  - do the official train/test patient lists + TIRADS reproduce Table 1 exactly?
  - does every patient's `images` list match files on disk, and do masks pair 1:1?
  - what are the mask value ranges (binary 0/255? multi-class?)
  - TN5000: official ImageSets sizes and VOC class distribution.

If Table 1 comes out exactly, the data adapter is provably correct and every
downstream patient-level number rests on solid ground.
"""
import collections
import json
import os

IMG_EXT = {".png", ".jpg", ".jpeg"}


def rule(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def find_dir(root, name):
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            if d == name:
                return os.path.join(dp, d)
    return None


report = {}
ROOT = "/kaggle/input"

TX = find_dir(ROOT, "ThyroidXL")
TN = find_dir(ROOT, "Main data")
print("ThyroidXL root:", TX)
print("TN5000    root:", TN)
report["roots"] = {"thyroidxl": TX, "tn5000": TN}

# --------------------------------------------------------------------------- #
rule("1. PATIENT LISTS")
def read_ids(p):
    with open(p, encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip()]

train_ids = read_ids(os.path.join(TX, "stats", "train_patients.txt"))
test_ids = read_ids(os.path.join(TX, "stats", "test_patients.txt"))
print("train patients:", len(train_ids), "| test patients:", len(test_ids))
print("overlap (must be 0):", len(set(train_ids) & set(test_ids)))
report["n_train_patients"] = len(train_ids)
report["n_test_patients"] = len(test_ids)
report["patient_overlap"] = len(set(train_ids) & set(test_ids))

# --------------------------------------------------------------------------- #
rule("2. id2info_eng.json STRUCTURE AND LABEL ENCODING")
info = json.load(open(os.path.join(TX, "stats", "id2info_eng.json"), encoding="utf-8"))
print("patients in id2info:", len(info))
print("covers all train+test:",
      len(set(train_ids + test_ids) - set(info)) == 0,
      "| missing:", len(set(train_ids + test_ids) - set(info)))

concl = collections.Counter(v.get("conclusion") for v in info.values())
print("\n'conclusion' code distribution:", dict(sorted(concl.items(), key=lambda kv: str(kv[0]))))

# what Conclusion string goes with each code?
pairs = collections.defaultdict(collections.Counter)
for v in info.values():
    nod = v.get("nodule_1") or {}
    pairs[v.get("conclusion")][str(nod.get("Conclusion"))] += 1
print("\ncode -> Conclusion string:")
for code in sorted(pairs, key=str):
    for s, n in pairs[code].most_common(4):
        print("   %-4s  %-45s %5d" % (code, s[:45], n))
report["conclusion_codes"] = {str(k): v for k, v in concl.items()}
report["code_to_string"] = {str(k): dict(v.most_common(6)) for k, v in pairs.items()}

# nodule_2 present?
n2 = sum(1 for v in info.values() if v.get("nodule_2"))
print("\npatients with a second nodule:", n2)
report["patients_with_nodule_2"] = n2

# --------------------------------------------------------------------------- #
rule("3. WHICH MAPPING REPRODUCES THE PAPER? (dev 877/3354, test 353/739)")
TARGET = {"dev_malignant": 877, "dev_n": 3354, "test_malignant": 353, "test_n": 739}
codes = sorted({v.get("conclusion") for v in info.values()}, key=str)
print("candidate malignant code sets, evaluated against Table 1:\n")
best = None
import itertools
for r in range(1, len(codes) + 1):
    for combo in itertools.combinations(codes, r):
        cs = set(combo)
        dm = sum(1 for p in train_ids if info.get(p, {}).get("conclusion") in cs)
        tm = sum(1 for p in test_ids if info.get(p, {}).get("conclusion") in cs)
        exact = (dm == TARGET["dev_malignant"] and tm == TARGET["test_malignant"])
        if exact or (abs(dm - 877) + abs(tm - 353)) < 60:
            print("   malignant = %-12s dev %4d (%.1f%%)  test %3d (%.1f%%)  %s"
                  % (str(sorted(cs, key=str)), dm, 100*dm/len(train_ids),
                     tm, 100*tm/len(test_ids), "<<< EXACT MATCH" if exact else ""))
        if exact:
            best = sorted(cs, key=str)
report["malignant_codes"] = best
print("\nRESOLVED malignant code set:", best if best else "NO EXACT MATCH -- inspect manually")

# --------------------------------------------------------------------------- #
rule("4. TIRADS DISTRIBUTION vs TABLE 1")
print("paper dev : T1=24  T2=328 T3=1090 T4=1024 T5=888")
print("paper test: T1=0   T2=45  T3=182  T4=215  T5=297\n")
for split, ids in (("dev", train_ids), ("test", test_ids)):
    c = collections.Counter()
    for p in ids:
        nod = (info.get(p) or {}).get("nodule_1") or {}
        c[nod.get("TIRADS")] += 1
    print("%-5s ours: %s" % (split, {k: c[k] for k in sorted(c, key=str)}))
    report.setdefault("tirads", {})[split] = {str(k): v for k, v in c.items()}

ages = [v.get("age") for v in info.values() if isinstance(v.get("age"), (int, float))]
gen = collections.Counter(v.get("gender") for v in info.values())
print("\nage: n=%d mean=%.1f min=%d max=%d   | gender codes: %s"
      % (len(ages), sum(ages)/len(ages), min(ages), max(ages), dict(gen)))
print("paper: mean age 48.1 +/- 12.7 (range 12-94), female 3650 (89.2%)")
report["age_mean"] = sum(ages)/len(ages)
report["gender_counts"] = {str(k): v for k, v in gen.items()}

# --------------------------------------------------------------------------- #
rule("5. FRAMES PER PATIENT vs TABLE 1 (dev mean 2.84, test 2.83, max 10)")
for split, ids, sub in (("dev", train_ids, "train"), ("test", test_ids, "test")):
    img_dir = os.path.join(TX, sub, "images")
    on_disk = collections.defaultdict(list)
    for f in os.listdir(img_dir):
        if os.path.splitext(f)[1].lower() in IMG_EXT:
            on_disk[f.split("_")[0]].append(f)
    counts = [len(on_disk.get(p, [])) for p in ids]
    listed = [len((info.get(p) or {}).get("images") or []) for p in ids]
    mismatch = sum(1 for a, b in zip(counts, listed) if a != b)
    print("%-5s images=%d  patients=%d  mean=%.2f  median=%d  min=%d  max=%d"
          % (split, sum(counts), len(counts), sum(counts)/len(counts),
             sorted(counts)[len(counts)//2], min(counts), max(counts)))
    print("      patients whose id2info image list != files on disk: %d" % mismatch)
    report.setdefault("frames", {})[split] = {
        "images": sum(counts), "patients": len(counts),
        "mean": sum(counts)/len(counts), "max": max(counts), "min": min(counts),
        "list_vs_disk_mismatch": mismatch}

# --------------------------------------------------------------------------- #
rule("6. MASK PAIRING AND PIXEL VALUES")
try:
    import numpy as np
    from PIL import Image
    for sub in ("train", "test"):
        img_dir = os.path.join(TX, sub, "images")
        msk_dir = os.path.join(TX, sub, "masks")
        imgs = set(os.path.splitext(f)[0] for f in os.listdir(img_dir))
        msks = set(os.path.splitext(f)[0] for f in os.listdir(msk_dir))
        print("%-5s images=%d masks=%d  exact stem matches=%d  unmatched images=%d"
              % (sub, len(imgs), len(msks), len(imgs & msks), len(imgs - msks)))
        vals, fracs, sizes = collections.Counter(), [], collections.Counter()
        for f in sorted(os.listdir(msk_dir))[:60]:
            a = np.array(Image.open(os.path.join(msk_dir, f)))
            vals.update(np.unique(a).tolist())
            fracs.append(float((a > 0).mean()))
            sizes[a.shape[:2]] += 1
        print("      mask unique values (60 sampled): %s" % sorted(vals)[:10])
        print("      lesion area fraction: mean %.4f  min %.4f  max %.4f"
              % (sum(fracs)/len(fracs), min(fracs), max(fracs)))
        print("      mask sizes: %s" % dict(list(sizes.items())[:4]))
        im = np.array(Image.open(os.path.join(img_dir, sorted(os.listdir(img_dir))[0])))
        print("      example image shape %s dtype %s" % (im.shape, im.dtype))
        report.setdefault("masks", {})[sub] = {
            "unmatched": len(imgs - msks), "values": sorted(vals)[:10],
            "mean_area_frac": sum(fracs)/len(fracs)}
except Exception as exc:
    print("mask check failed:", exc)

# --------------------------------------------------------------------------- #
rule("7. labels_orig CLASS STRINGS (per-image polygon labels)")
lo = os.path.join(TX, "test", "labels_orig")
lab = collections.Counter()
for f in sorted(os.listdir(lo))[:800]:
    try:
        d = json.load(open(os.path.join(lo, f), encoding="utf-8"))
        if isinstance(d, dict):
            lab[d.get("label")] += 1
        elif isinstance(d, list):
            for e in d:
                lab[e.get("label")] += 1
    except Exception:
        pass
print("label strings (800 sampled):", dict(lab.most_common(8)))
report["labels_orig_strings"] = dict(lab.most_common(8))

yolo = collections.Counter()
ld = os.path.join(TX, "test", "labels")
for f in sorted(os.listdir(ld))[:800]:
    for line in open(os.path.join(ld, f), encoding="utf-8"):
        if line.strip():
            yolo[line.split()[0]] += 1
print("YOLO class ids in labels/*.txt (800 sampled):", dict(yolo.most_common()))
report["yolo_classes"] = dict(yolo.most_common())

# --------------------------------------------------------------------------- #
rule("8. TN5000 OFFICIAL SPLITS AND CLASSES")
if TN:
    ms = os.path.join(TN, "ImageSets", "Main")
    for f in sorted(os.listdir(ms)):
        ids = read_ids(os.path.join(ms, f))
        print("  %-12s %5d ids   e.g. %s" % (f, len(ids), ids[:3]))
        report.setdefault("tn5000_splits", {})[f] = len(ids)
    import xml.etree.ElementTree as ET
    ann = os.path.join(TN, "Annotations")
    names, nobj = collections.Counter(), collections.Counter()
    for f in sorted(os.listdir(ann)):
        try:
            r = ET.parse(os.path.join(ann, f)).getroot()
            objs = r.findall("object")
            nobj[len(objs)] += 1
            for o in objs:
                names[(o.findtext("name") or "").strip()] += 1
        except Exception:
            pass
    print("\n  VOC class distribution:", dict(names.most_common(10)))
    print("  objects per image     :", dict(sorted(nobj.items())))
    report["tn5000_classes"] = dict(names)
    report["tn5000_objects_per_image"] = {str(k): v for k, v in nobj.items()}

    # class balance within each official split
    stem2cls = {}
    for f in sorted(os.listdir(ann)):
        try:
            r = ET.parse(os.path.join(ann, f)).getroot()
            o = r.find("object")
            if o is not None:
                stem2cls[os.path.splitext(f)[0]] = (o.findtext("name") or "").strip()
        except Exception:
            pass
    for f in sorted(os.listdir(ms)):
        ids = read_ids(os.path.join(ms, f))
        c = collections.Counter(stem2cls.get(i, "?") for i in ids)
        print("  %-12s class balance: %s" % (f, dict(c.most_common(5))))
        report.setdefault("tn5000_split_balance", {})[f] = dict(c.most_common(5))

# --------------------------------------------------------------------------- #
rule("9. DONE")
json.dump(report, open("/kaggle/working/verify.json", "w", encoding="utf-8"),
          indent=2, default=str)
print("wrote /kaggle/working/verify.json")
