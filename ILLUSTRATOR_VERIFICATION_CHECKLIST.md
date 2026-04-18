# Illustrator Verification Checklist - Book 2 Figures

**Book:** Optimization for AI (The AI Engineer's Library)  
**Date Generated:** 2026-04-17  
**Chapters Covered:** 5 (Regularization), 6 (Loss Functions), 7 (Convergence)

---

## Chapter 5: Regularization (ch13)

### Figure Files Found

| # | File | Format |
|---|------|--------|
| 1 | regularization-taxonomy-flagship.png | PNG |
| 2 | regularization-taxonomy-flagship.pdf | PDF |
| 3 | regularization-taxonomy.png | PNG |
| 4 | regularization-taxonomy.pdf | PDF |
| 5 | l1-vs-l2.png | PNG |
| 6 | l1-vs-l2.pdf | PDF |
| 7 | dropout-train-vs-inference.png | PNG |
| 8 | dropout-train-vs-inference.pdf | PDF |
| 9 | batchnorm-train-vs-inference.png | PNG |
| 10 | batchnorm-train-vs-inference.pdf | PDF |
| 11 | overfitting-curves.png | PNG |
| 12 | overfitting-curves.pdf | PDF |

**Directory:** `/Users/vraghavan/Desktop/Book/publish/images/diagrams/ch13/`

### Verification Checklist

- [ ] **regularization-taxonomy-flagship.png** - FONT SIZE CHECK
  - Verify all text labels use 14pt+ fonts
  - Check that taxonomy hierarchy labels are legible at print size
  - Ensure category names are not truncated

- [ ] **l1-vs-l2.png** - GEOMETRY CHECK
  - Verify L1 diamond has clear, sharp corners (not rounded)
  - Check that the sparsity-inducing property is visually apparent
  - Confirm L2 circle is smooth and distinguishable from L1

- [ ] **dropout-train-vs-inference.png** - SCALING FACTOR CHECK
  - Verify the (1-p) scaling factor is explicitly shown
  - Check that training mode shows dropped neurons clearly
  - Confirm inference mode shows all neurons active with scaling applied

- [ ] **batchnorm-train-vs-inference.png** - General review
  - Verify training vs inference differences are clearly labeled
  - Check font sizes meet 14pt minimum

- [ ] **overfitting-curves.png** - General review
  - Verify training/validation curve distinction is clear
  - Check axis labels and legend are legible

- [ ] **regularization-taxonomy.png** - General review
  - Compare with flagship version for consistency
  - Verify hierarchy is complete and accurate

---

## Chapter 6: Loss Functions (ch14)

### Figure Files Found

| # | File | Format |
|---|------|--------|
| 1 | fig-focal-loss.png | PNG |
| 2 | fig-ce-vs-mse-gradients.png | PNG |
| 3 | loss-metric-mismatch.png | PNG |
| 4 | loss-metric-mismatch.pdf | PDF |
| 5 | perceptual-style-losses.png | PNG |
| 6 | perceptual-style-losses.pdf | PDF |

**Directory:** `/Users/vraghavan/Desktop/Book/publish/images/diagrams/ch14/`

### TikZ Figures (Inline in ch14-loss-functions.tex)

| # | Figure Reference | Description | Line |
|---|-----------------|-------------|------|
| 1 | fig:ch14-regression-losses | MSE, MAE, Huber comparison | ~212 |
| 2 | fig:ch14-margin-losses | Hinge, Logistic, Squared Hinge | ~555 |
| 3 | fig:ch14-triplet-loss | Triplet loss constraint diagram | ~838 |

**Source File:** `/Users/vraghavan/Desktop/Book/publish/chapters/part2/ch14-loss-functions.tex`

### Verification Checklist

#### TikZ Figures (require LaTeX compilation check)

- [ ] **fig:ch14-regression-losses (Huber piecewise curve)** - GAP CHECK
  - Verify no visible gaps at transition points (delta = 1, -1)
  - Check: Line 230-232 plots three segments for Huber
  - Segments: `domain=-4:-1`, `domain=-1:1`, `domain=1:4`
  - Confirm smooth visual continuity at joints
  - Verify Huber formula consistency: quadratic region `0.5*x^2`, linear region `abs(x) - 0.5`

- [ ] **fig:ch14-margin-losses (Hinge piecewise curve)** - GAP CHECK
  - Verify no visible gap at margin point (m = 1)
  - Check: Lines 569-570 for hinge segments
  - Segments: `domain=-2:1` (linear descent), `domain=1:3` (zero)
  - Confirm the two segments meet exactly at point (1, 0)

- [ ] **fig:ch14-triplet-loss** - BOUNDING BOX CHECK
  - Verify all labels fit within figure boundaries:
    - "Anchor a" label (below node)
    - "Positive p" label (above node)
    - "Negative n" label (above node)
    - Distance labels: "d(a,p)", "d(a,n)"
    - "margin m" bracket label
    - "valid region for n" annotation
  - Check number line labels at bottom are not clipped
  - Verify constraint notation is fully visible

#### PNG Figures

- [ ] **fig-focal-loss.png** - LEGEND CHECK
  - Verify gamma legend is present and shows multiple gamma values
  - Check that gamma = 0, 1, 2, 5 (or similar range) curves are labeled
  - Confirm legend position does not obscure curves

- [ ] **fig-ce-vs-mse-gradients.png** - AXIS LABELS CHECK
  - Verify Panel 1 has x-axis and y-axis labels
  - Verify Panel 2 has x-axis and y-axis labels  
  - Verify Panel 3 has x-axis and y-axis labels
  - Check that all three panels have consistent axis scaling
  - Confirm panel titles or descriptions are present

- [ ] **loss-metric-mismatch.png** - General review
  - Verify concept is clearly illustrated
  - Check font sizes meet standards

- [ ] **perceptual-style-losses.png** - General review
  - Verify diagram clarity
  - Check all text is legible

---

## Chapter 7: Convergence (ch15)

### Figure Files Found

| # | File | Format |
|---|------|--------|
| 1 | fig-convergence-rates-flagship.png | PNG |
| 2 | fig-convergence-rates-flagship.pdf | PDF |
| 3 | fig-convergence-rates.png | PNG |
| 4 | fig-convergence-rates.pdf | PDF |
| 5 | fig-acceleration-comparison.png | PNG |
| 6 | fig-acceleration-comparison.pdf | PDF |
| 7 | fig-condition-number.png | PNG |
| 8 | fig-condition-number.pdf | PDF |
| 9 | fig-condition-number-comparison.png | PNG |
| 10 | fig-condition-number-comparison.pdf | PDF |
| 11 | fig-lr-failure-modes.png | PNG |
| 12 | fig-lr-failure-modes.pdf | PDF |
| 13 | fig-momentum-effect.png | PNG |
| 14 | fig-momentum-effect.pdf | PDF |
| 15 | fig-sgd-gd-compute-flagship.png | PNG |
| 16 | fig-sgd-gd-compute-flagship.pdf | PDF |
| 17 | fig-sgd-vs-gd.png | PNG |
| 18 | fig-sgd-vs-gd.pdf | PDF |
| 19 | fig-smoothness.png | PNG |
| 20 | fig-smoothness.pdf | PDF |

**Directory:** `/Users/vraghavan/Desktop/Book/publish/images/diagrams/ch15/`

### Verification Checklist

#### Flagship Figure - Convergence Rates

- [ ] **fig-convergence-rates-flagship.png** - TABLE 15.1 CONSISTENCY CHECK
  - Verify the figure shows three distinct convergence rates:
    - [ ] Sublinear: O(1/t) - should show gradual asymptotic decay
    - [ ] Accelerated: O(1/t^2) - should show faster decay than sublinear
    - [ ] Linear: O(rho^t) where 0 < rho < 1 - should show exponential decay
  - Confirm rates are clearly labeled in legend or annotations
  - Check that visual representation matches mathematical definitions
  - Verify curves are distinguishable (different colors/line styles)

#### Grayscale Readability Test

**IMPORTANT:** All ch15 figures must be tested for grayscale readability. Convert each figure to grayscale and verify curves remain distinguishable.

- [ ] **fig-convergence-rates-flagship.png** - Grayscale test
- [ ] **fig-convergence-rates.png** - Grayscale test
- [ ] **fig-acceleration-comparison.png** - Grayscale test
- [ ] **fig-condition-number.png** - Grayscale test
- [ ] **fig-condition-number-comparison.png** - Grayscale test
- [ ] **fig-lr-failure-modes.png** - Grayscale test
- [ ] **fig-momentum-effect.png** - Grayscale test
- [ ] **fig-sgd-gd-compute-flagship.png** - Grayscale test
- [ ] **fig-sgd-vs-gd.png** - Grayscale test
- [ ] **fig-smoothness.png** - Grayscale test

#### Grayscale Testing Procedure
1. Open figure in image editor or use command: `convert input.png -colorspace Gray output_gray.png`
2. Verify all curves are distinguishable without color
3. Check that line styles (solid, dashed, dotted) provide sufficient differentiation
4. Confirm legends remain readable
5. Note any figures that rely solely on color for distinction

#### General Quality Checks (All ch15 figures)

- [ ] **fig-acceleration-comparison.png**
  - Verify momentum vs non-momentum comparison is clear
  - Check convergence trajectory visualizations

- [ ] **fig-condition-number.png** / **fig-condition-number-comparison.png**
  - Verify ill-conditioned vs well-conditioned cases shown
  - Check contour plots are distinguishable

- [ ] **fig-lr-failure-modes.png**
  - Verify oscillation (too high) and slow convergence (too low) shown
  - Check failure mode labels are clear

- [ ] **fig-momentum-effect.png**
  - Verify damping/acceleration effect is visible
  - Check trajectory comparison is clear

- [ ] **fig-sgd-gd-compute-flagship.png** / **fig-sgd-vs-gd.png**
  - Verify stochastic noise vs deterministic path shown
  - Check compute cost trade-off is illustrated

- [ ] **fig-smoothness.png**
  - Verify Lipschitz smoothness concept is illustrated
  - Check gradient bound visualization

---

## Summary Statistics

| Chapter | PNG Files | PDF Files | TikZ (Inline) | Total |
|---------|-----------|-----------|---------------|-------|
| Ch5 (Regularization) | 6 | 6 | 0 | 12 |
| Ch6 (Loss Functions) | 4 | 2 | 3 | 9 |
| Ch7 (Convergence) | 10 | 10 | 0 | 20 |
| **Total** | **20** | **18** | **3** | **41** |

---

## Priority Items

### High Priority (Specific Requirements)
1. [ ] Ch5: regularization-taxonomy-flagship.png - 14pt+ font verification
2. [ ] Ch5: l1-vs-l2.png - L1 diamond corner sharpness
3. [ ] Ch5: dropout-train-vs-inference.png - (1-p) scaling factor visible
4. [ ] Ch6: TikZ Huber/Hinge curves - no gaps at joints
5. [ ] Ch6: TikZ triplet loss - bounding box includes all labels
6. [ ] Ch6: fig-focal-loss.png - gamma legend present
7. [ ] Ch6: fig-ce-vs-mse-gradients.png - axis labels on all 3 panels
8. [ ] Ch7: fig-convergence-rates-flagship.png - matches Table 15.1 rates
9. [ ] Ch7: ALL figures - grayscale readability

### Medium Priority (General Quality)
- All PNG figures: 300 DPI minimum
- All figures: 14-18pt fonts for print
- All figures: Professional color scheme
- All figures: Marker styles visible at print size

---

## Sign-off

| Chapter | Reviewer | Date | Status |
|---------|----------|------|--------|
| Ch5 | | | [ ] Pending |
| Ch6 | | | [ ] Pending |
| Ch7 | | | [ ] Pending |

**Final Approval:** _________________ Date: _________
