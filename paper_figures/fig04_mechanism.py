"""Figure 4 -- what does the reliability mechanism actually learn?

All four panels come from the corrected reliability<->counterfactual-influence
analysis (src/eval/counterfactual.py), which computes influence as the change
in predicted probability when one (frame, region) evidence token is suppressed,
and correlates it with the model's own reliability score R for that token,
WITHIN region and WITHIN patient (across frames) so the region axis cannot
confound the result. Region names use the fixed order and colours established
in src/viz/explain.py.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, NEUTRAL, REGION_COLORS, style_ax

REGIONS = ["core", "margin", "peri", "global"]


def main() -> None:
    rel = {m: data.load_json(data.RELIABILITY_JSON[m]) for m in ("der_mil", "mr_mil")}
    tok_der = pd.read_csv(data.require(data.RELIABILITY_TOKENS["der_mil"]))
    ea = pd.read_csv(data.require(data.EVIDENCE_ABLATION))
    ea = ea[ea.suppressed != "none"].set_index("suppressed").loc[REGIONS].reset_index()

    cv = rel["der_mil"]["cross_view"]["per_region"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    # ---- (a) evidence ablation: cost of removing each region ----------------- #
    ax = axes[0, 0]
    x = np.arange(len(REGIONS))
    ax.bar(x, ea["mean_abs_delta_p"], color=[REGION_COLORS[r] for r in REGIONS],
          width=0.6)
    for i, v in enumerate(ea["mean_abs_delta_p"]):
        ax.text(i, v + 0.002, "%.4f" % v, ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in REGIONS], fontsize=10)
    style_ax(ax, "(a) Counterfactual influence by region\n(mean |Δp| when suppressed, DER-MIL)",
            ylabel="Mean |Δ predicted probability|")

    # ---- (b) region profile: mean R vs mean influence, both models ----------- #
    ax = axes[0, 1]
    for m, marker in (("der_mil", "o"), ("mr_mil", "s")):
        prof = rel[m]["region_profile"]
        for r in REGIONS:
            ax.scatter(prof[r]["mean_influence"], prof[r]["mean_R"],
                      s=140, marker=marker, color=REGION_COLORS[r],
                      edgecolor=INK, linewidth=0.8,
                      label=("%s" % r.capitalize()) if m == "der_mil" else None)
    ax.scatter([], [], marker="o", color=NEUTRAL, label="DER-MIL")
    ax.scatter([], [], marker="s", color=NEUTRAL, label="MR-MIL")
    ax.set_xscale("log")
    style_ax(ax, "(b) Learned reliability vs actual influence\n(per region, per model)",
            xlabel="Mean counterfactual influence (log scale)",
            ylabel="Mean reliability score R")
    ax.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))

    # ---- (c) cross-view rho per region, DER-MIL vs MR-MIL --------------------- #
    ax = axes[1, 0]
    w = 0.38
    x = np.arange(len(REGIONS))
    for m, off in (("der_mil", -w / 2), ("mr_mil", w / 2)):
        pr = rel[m]["cross_view"]["per_region"]
        vals = [pr[r]["mean_spearman"] for r in REGIONS]
        ax.bar(x + off, vals, w, color=MODEL_COLORS[m], alpha=0.9,
              label="DER-MIL" if m == "der_mil" else "MR-MIL")
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in REGIONS], fontsize=10)
    style_ax(ax, "(c) Reliability predicts influence?\nWithin-patient Spearman ρ, R vs |Δp| (across frames)",
            ylabel="Mean Spearman ρ")
    ax.legend(fontsize=9)
    cv_all = rel["der_mil"]["cross_view"]
    ax.text(0.02, 0.03,
           "DER-MIL overall: ρ=%.4f, n=%d correlations, p=%.4f"
           % (cv_all["mean_spearman"], cv_all["n"], cv_all["ttest_p"]),
           transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")

    # ---- (d) worked example: R across a real patient's frames, DER-MIL ------- #
    ax = axes[1, 1]
    sub3 = tok_der[tok_der.n_valid_frames >= 3]
    pid = sub3.groupby("patient_id").size().idxmax()
    one = sub3[sub3.patient_id == pid].pivot(index="frame", columns="region", values="R")
    one = one[REGIONS]
    for r in REGIONS:
        ax.plot(one.index, one[r], marker="o", color=REGION_COLORS[r],
               linewidth=1.8, markersize=6, label=r.capitalize())
    style_ax(ax, "(d) Reliability across frames, real patient %s\n(DER-MIL; %d valid frames)"
            % (pid, len(one)), xlabel="Frame index", ylabel="Reliability score R")
    ax.set_xticks(one.index)
    ax.legend(fontsize=8.5, ncol=2)

    fig.tight_layout()

    # How much does R actually vary across a patient's own frames, per region?
    # Recomputed live from the raw token dump: a correlation on a value that
    # never moves is noise on ties, not a signal, so this is load-bearing for
    # how panel (c) should be read.
    const_frac = {}
    for r in REGIONS:
        rng = (sub3[sub3.region == r].groupby("patient_id")["R"]
              .agg(lambda s: s.max() - s.min()))
        const_frac[r] = float((rng < 1e-4).mean())

    caption = (
        "Figure 4. What the reliability mechanism learns (DER-MIL, ResNet-50, "
        "ThyroidXL). (a) Counterfactual influence of each evidence region, "
        "measured by suppressing it and recording the mean absolute change in "
        "predicted probability: core and margin drive the prediction "
        "(|Δp|≈0.11-0.12); peri and global are nearly inert (|Δp|<0.002). "
        "(b) Mean reliability score R vs mean influence, per region, for "
        "DER-MIL (circles) and its reliability-disabled ablation MR-MIL "
        "(squares); a well-calibrated mechanism would place high-influence "
        "regions at high R. (c) Within-patient Spearman correlation between R "
        "and counterfactual influence, computed within region across a "
        "patient's frames (bags with >=3 valid frames) so the region axis "
        "cannot confound the statistic. Reliability predicts influence with "
        "the intended sign for margin (ρ=%.3f, %.0f%% of patients positive) "
        "and, more weakly, peri (ρ=%.3f, %.0f%% positive); it is close to "
        "chance for global (ρ=%.3f, %.0f%% positive) and inverted for core "
        "(ρ=%.3f, %.0f%% positive) -- the single region carrying the most "
        "influence. (d) Part of why global and peri are weak is visible "
        "directly: R varies by less than 1e-4 across a patient's own frames "
        "for global in %.0f%% of patients and for peri in %.0f%% of patients "
        "(recomputed from the raw per-token dump), leaving little genuine "
        "variation for the correlation to act on. Worked example shown for "
        "one real test patient with %d valid frames."
        % (cv["margin"]["mean_spearman"], 100 * cv["margin"]["frac_positive"],
           cv["peri"]["mean_spearman"], 100 * cv["peri"]["frac_positive"],
           cv["global"]["mean_spearman"], 100 * cv["global"]["frac_positive"],
           cv["core"]["mean_spearman"], 100 * cv["core"]["frac_positive"],
           100 * const_frac["global"], 100 * const_frac["peri"], len(one))
    )
    style.save(fig, "fig04_reliability_mechanism", caption)


if __name__ == "__main__":
    main()
