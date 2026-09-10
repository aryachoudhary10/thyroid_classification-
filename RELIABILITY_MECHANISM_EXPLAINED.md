# The Reliability Mechanism: How It Works and Why It Doesn't Contribute

*A summary for explaining the negative result to a supervisor.*

---

## Part 1: How Reliability Is Calculated

**Setup.** Each patient has a "bag" of ultrasound frames (median 3, max 10). Each frame is split into 4 evidence regions — **core** (inside the nodule), **margin** (its rim), **peri** (surrounding tissue), **global** (whole frame). A patient with 3 frames produces 3 × 4 = 12 "evidence tokens," each getting its own reliability score.

**For every token, four signals are computed:**

| Signal | How | Intuition |
|---|---|---|
| **I** (Importance) | Small neural network reads the token's features | "How predictive does this evidence claim to be, on its own?" |
| **U** (Uncertainty) | Another small network reads the same features | "How confident is the model in this observation?" — trained indirectly, with no ground truth for uncertainty itself |
| **S** (Support) | Compares this token to the *same region in the patient's other frames* (margin vs. margin, never margin vs. core) | "Do other views corroborate this one?" |
| **D** (Contradiction) | Same comparison, opposite direction | "Do other views disagree with this one?" |

These combine into one number:

```
R = sigmoid( f(I, S, D, U) )        R ∈ (0, 1)
```

**R does two jobs:** it decides how much to trust each region when merging them into a single frame representation, and it biases the attention that merges frames into one patient-level prediction.

**The design hypothesis, in one sentence:**
> *Trust a piece of evidence more if it's important, if other views back it up, and if the model isn't uncertain about it — trust it less if other views contradict it.*

The question the project then asked: **does the model actually learn to do that, and does it help accuracy?**

---

## Part 2: The Evidence It Isn't Contributing

Three independent tests, each harder to dismiss than the last.

### Test 1 — Ablation: does removing reliability change accuracy?

A twin model was built, identical in every way, with the reliability head switched off (called **MR-MIL**). Same backbone, same 4 regions, same everything — the only difference is 0.43M parameters (1.6% of the model).

| Model | ROC-AUC (internal test) |
|---|---|
| With reliability (DER-MIL) | 0.9911 |
| Without reliability (MR-MIL) | 0.9942 |

The version *without* the mechanism is at least as good, on two different backbones (ResNet-50 and ConvNeXt-Tiny), and the difference is not statistically significant either way (p ≈ 0.06–0.44).

**→ Turning it off costs nothing.**

### Test 2 — Does the reliability score track what actually matters?

This is the direct test: if the model calls a piece of evidence "reliable," removing it should hurt the prediction *more* than removing something it calls "unreliable."

Measured precisely: for every token, suppress it and record how much the prediction changes ("influence"), then correlate that against the model's own R score — **within the same region, across a patient's different frames only** (so it's comparing frame-to-frame within margin, or within core, never across regions).

| Region | Does R predict influence? |
|---|---|
| Margin | Yes, correctly (ρ = +0.33 on ResNet-50) |
| Core | Correct on ConvNeXt (+0.51), **backwards** on ResNet-50 (−0.41) |
| Peri | Correct on ResNet-50 (+0.20), **backwards** on ConvNeXt (−0.42) |
| Global | R barely varies across a patient's frames at all — correlation often undefined |

**The key finding:** which region gets it backwards *depends on which backbone is used*. If the mechanism had learned something real about thyroid anatomy, the same region should misbehave consistently across backbones. Instead, core breaks on one backbone and peri breaks on the other — the signature of noise, not a learned rule.

### Test 3 — Is this an architecture problem, not a concept problem?

The fusion formula was forced into a constrained form that can *only* increase trust for supportive evidence and *only* decrease it for contradictory/uncertain evidence, removing the model's freedom to learn something perverse:

```
R = sigmoid( I + a·Support − b·Contradiction − c·Uncertainty )     a, b, c ≥ 0 by construction
```

After full training, **a, b, c had moved only 1–3% from their starting values.** The model essentially never adjusted these weights — it found no reason to lean on support or contradiction at all. Accuracy under this constrained, "more honest" version was slightly *worse*, not better.

---

## Part 3: Why — The Likely Root Cause

Two structural reasons, and they compound.

1. **Not enough views to compare.** Support and Contradiction only mean something when there are multiple frames of the same patient to cross-check. Median bag size here is 3 frames. On the external validation dataset (TN5000), every patient has exactly **1** image — Support and Contradiction are mathematically forced to zero for the entire dataset. There's very little material for a "does this view agree with that view" signal to work with.

2. **Confound in how influence is distributed.** Core and margin regions drive roughly 100× more of the prediction than peri and global (measured directly by ablating each region). A reliability score meant to rank frames *within* a region has very little room to matter when the region itself barely affects the outcome — for peri and global, the score is often nearly flat across a patient's own frames, meaning there was almost nothing for it to rank.

---

## The Honest Framing

The mechanism was hypothesis-tested three independent ways — ablation, a direct correlation test with the region confound explicitly controlled for, and a constrained-architecture probe — and **all three converge on the same negative result.** That is not a failure of the experiment; it is what a well-designed negative result looks like.

The part of the architecture that *does* work, and carries the real accuracy gain over the published baseline, is the **four-region evidence encoding itself** (core/margin/peri/global replacing the original two-branch design). That is the genuine, statistically significant contribution (p = 0.023 vs. the baseline model).
