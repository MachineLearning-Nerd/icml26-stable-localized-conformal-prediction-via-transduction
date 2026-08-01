# Source audit — Stable Localized Conformal Prediction via Transduction

## Provenance

| Item | Value |
|---|---|
| Paper | Stable Localized Conformal Prediction via Transduction |
| Authors | Yinjie Min (Nankai), Liuhua Peng (Melbourne), Changliang Zou (Nankai) |
| arXiv | 2605.01452 |
| OpenReview | https://openreview.net/forum?id=lSMTccAN61 |
| Source retrieved | `https://ar5iv.labs.arxiv.org/html/2605.01452` |
| Retrieval date | 2026-08-01 (UTC) |
| Retrieval method | `curl -sL -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"` |
| HTTP status / bytes | `200`, 1 204 691 bytes |
| SHA-256 of retrieved HTML | `d07bc37a6a81e0c74aef488fd566dfe5ebf4e0b94ad526c3449334000c2741a2` |
| Authors' code artifact | `https://github.com/OswinMin/StCP` (cited in Section 5 of the paper) |
| Code artifact commit pinned | `1d8df7614d49eada881426742688ba75fec631b9` |

The authors' artifact ships **both** the reference implementation (`Main/`,
`SimuAnalysis/`, `RealAnalysis/`) and the Table 1 datasets (`Dataset/`:
`crimedata.csv`, `proteinStructure.csv`, `achieve.csv`, `dermamnist.npz`,
`tissuemnist.npz`) plus the two pretrained image backbones
(`RealAnalysis/Para/{DermaMNIST,TissueMNIST}.pth`). Every experiment below is run
against that artifact, not against a paraphrase of it.

---

## Global experimental constants (Section 5, Appendix C.1–C.2)

| Symbol | Value | Anchor |
|---|---|---|
| Nominal level `1 − α` | `0.90` | §5, "The nominal coverage level is fixed at 1−α=90% throughout." |
| Repeats | `50` | §5, "All experiments are repeated 50 times." |
| `α_n` | `1 − (1−α)(n+1)/n` | Algorithm 1, step 5 |
| Conditional CDF learner `A` | engression network, hidden layers `(50,100,100,50)` | §5.1 |
| Tuned parameters `k₀` | `10000` (the `100×100` weight matrix only) | §5.1, Appendix C.1 |
| Penalty | `R(F_θ, F_θ̂) = ‖θ − θ̂‖₂² / k₀` | §5.1 |
| `α_tol` for `ours-sel` | `0.02` | §5.1 |
| Regression score | residual score | §5.1 |
| Classification score | `V(x,y) = 1 − μ̂(x)_y` | §5.1 |
| Std definition | sample sd with `1/(50−1)` denominator over the 50 repeats | Appendix C.1 |

### Two *different* percentage-improvement formulas — a reproduction trap

Appendix C.1 defines the Table 1 percentage explicitly, and it is **not**
`1 − a₁/a_base`:

> Denote by `a₁` the Std value of the reported StCP variant (either "ours" or
> "ours-sel") and `a₀` the Std value of the corresponding oracle baseline. Let
> `a_ref` denote the reference baseline, defined as the smallest Std among
> "base", SDCP, and PPI **whose marginal coverage remains within the acceptable
> range**. The reported improvement is `(1 − (a₁−a₀)/(a_ref−a₀)) × 100%`.

Confirmed in the authors' code at `RealAnalysis/sum_tab.py:61`:
`(np.min(base)-std[i])/(np.min(base)-ora)*100`.

Table 2 uses a **different**, simpler formula — `SimuAnalysis/sum_tab.py:93`:
`(base_std - value) / base_std * 100.0`, i.e. relative to `base` with no oracle
adjustment.

Cross-check of both formulas against published cells (published value in
parentheses):

| Cell | Formula | Recomputed | Published |
|---|---|---|---|
| Table 1 STAR / GLCP / ours | oracle-adj, `a_ref=8.78`, `a₀=1.48`, `a₁=5.25` | `48.4%` | `48.4%` |
| Table 1 CRIME / GLCP / ours | oracle-adj, `0.75, 0.16, 0.50` | `42.4%` | `42.9%` |
| Table 1 BIO / GLCP / ours | oracle-adj, `0.53, 0.07, 0.37` | `34.8%` | `35.7%` |
| Table 1 DERMA / GLCP / ours | oracle-adj, `0.68, 0.06, 0.49` | `30.6%` | `30.4%` |
| Table 1 TISSUE / GLCP / ours | oracle-adj, `0.83, 0.12, 0.73` | `14.1%` | `13.5%` |
| Table 2 LogAbs n=30 / GLCP / ours | plain, `1.12 → 0.77` | `31.25%` | `31.2%` |
| Table 2 LogAbs n=30 / CQR / ours | plain, `0.98 → 0.82` | `16.33%` | `16.3%` |

Residual discrepancies in the Table 1 rows are consistent with the table
reporting Std rounded to 2 decimals while the percentage is computed from
unrounded values. The two exactly-matching anchors (48.4%, 31.2%, 16.3%) pin the
formulas.

### Marginal-coverage annotation — the two tables use *different* thresholds

Both captions state the same rule: `−` below `1 − α − 0.01 = 0.89`, `+` above
`1 − α + 1/(n+1)`. The authors' code does not agree with the caption for
Table 1:

| Table | Code | Upper edge at `n = 30` |
|---|---|---|
| Table 1 | `RealAnalysis/sum_tab.py:34` — `data > 0.901 + 1/n` | `0.934333` |
| Table 2 | `SimuAnalysis/sum_tab.py:56` — `value > target + 1/(n+1)` | `0.932258` |

This is load-bearing, not cosmetic. DERMA / GLCP / `ours` has published marginal
coverage `0.933`. Under the caption's formula that is above the upper edge and
would be flagged `+` — which would also exclude it from consideration and change
the reference baseline `a_ref`. Under the code's formula it is below `0.934333`
and is printed unmarked, which is exactly what the paper shows. Reproducing the
published table therefore requires matching each table's own code, so that is
what the verifier does (`REAL_ANNOTATION_BAND` / `SIM_ANNOTATION_BAND` in
`src/published.py`).

A competitor carrying either mark is excluded from `a_ref` (Appendix C.1 and
`RealAnalysis/sum_tab.py:15–19`).

---

## Claim-by-claim source anchors

### Claim 1 — the StCP/SLCP formulation (Equation 7, Section 3)

Equation (7):

> `F̃_{S|X} = argmin_{F̃ ∈ ℱ} d( F̂_S^0(·), F̂_S^1(·; F̃) ) + λ R(F̃, F̂_{S|X})`

with, from Algorithm 1:

- `F̂_{S|X} = A(L'_N)` — initial conditional CDF estimator trained on **labeled
  source data**;
- `F̂_S^0(s) = n⁻¹ Σᵢ 1(Sᵢ ≤ s)` — empirical marginal CDF of the `n` **labeled
  target** calibration scores;
- `F̂_S^1(s; F̃) = m⁻¹ Σⱼ F̃(s | X̃ⱼ)` — transductive marginal estimate over the
  `m` **unlabeled target** covariates;
- `q̂_St = inf{ s : F̂_S^1(s; F̃_{S|X}) ≥ 1 − α_n }`, `α_n = 1 − (1−α)(n+1)/n`;
- `Ĉ_St(X_{n+1}) = { y : S(X_{n+1}, y) ≤ q̂_St }`.

Limiting behaviour asserted in §3: at `λ = 0` the finite-grid Wasserstein
alignment recovers the standard conformal quantile of `F̂_S^0`; as `λ → ∞`,
`F̃_{S|X} → F̂_{S|X}`. SLCP is the same construction with `S` replaced by a
localized score.

**Domain.** The claim is a statement about what the method *is* and that it runs
as specified — it is not universally quantified over distributions.

### Claim 2 — Theorem 4.2 (coverage robustness to λ)

Assumption 4.1: (i) `Θ` bounded in `‖·‖₂`; (ii) support of `F(·|x;θ)` bounded
uniformly in `x, θ`; (iii) `F(s|x;θ)` Lipschitz in `θ`; (iv) density
`f(·|·;θ)` uniformly bounded away from `0` and `∞`, with `∂f/∂θ` and `∂f/∂s`
uniformly bounded. `d` is the squared `∞`-Wasserstein distance and
`R(F_θ,F_θ̂) = ‖θ−θ̂‖₂²`.

Theorem 4.2: if there is `ϵ ≥ 0` with
`min_θ d(F̂_S^0, F̂_S^1(·;F_θ)) ≤ ϵ²` for any realization, and
`δ_S = |Q(1−α;F_S) − Q(1−α_n; F̂_S^1(·;F_θ̂))|`, then there exists `C > 0` with

> `| P(Y_{n+1} ∈ Ĉ_St(X_{n+1})) − (1−α) | ≤ C · min( ϵ + λ^{1/2} + n⁻¹, δ_S + λ^{−1/2} + n⁻¹ )`.

**Quantifier structure.** `∃C > 0` such that `∀λ`. The constant is not
specified, so the bound is *not* directly falsifiable at a single `λ`; what is
testable is (a) the two-regime shape — small-`λ` control via `ϵ`, large-`λ`
control via `δ_S` — and (b) the substantive consequence stated in Remark 4.3
and §3.1, namely that coverage stays near `1−α` **across a wide λ range**.

### Claim 3 — Theorem 4.6 (set stability rates)

Preconditions: Lemma 4.4 (density of `F_S` bounded away from zero; `q̂`
independent of `X_{n+1}`; `L(Ĉ) = L₀(q̂)` with `L₀` `C_L`-Lipschitz) and
Assumption 4.5 (`Θ` convex; `θ ↦ d(F_S,F_S^1(·;F_θ))` twice continuously
differentiable; `∇_{θθ} d ⪰ c_d I`).

Theorem 4.6:

- standard conformal: `Var(L(Ĉ) | D_aux) = O(n⁻¹)`;
- if `c_d > 0`: `Var(L(Ĉ_St) | D_aux) = O(m⁻¹ + {n(1+λ)²}⁻¹)`;
- if `c_d ≤ 0` and `λ ≥ 1 − c_d`: `Var(L(Ĉ_St) | D_aux) = O(m⁻¹ + (nλ²)⁻¹)`.

Stated consequence: "the first term depends on the number of unlabeled samples
and is negligible when `m ≫ n`, while the second term decreases as the
regularization level increases."

**Quantifier structure.** Asymptotic `O(·)` statements with unspecified
constants. Directly testable content: the **measured scaling exponent** of
standard-conformal set-size variance in `n`, and the **direction and magnitude**
of the StCP variance reduction as `λ` grows and as `m/n` grows.

### Claim 4 — Table 1 (five real datasets)

Datasets and splits (§5.1 and Appendix C.3): BIO (protein tertiary structure,
`d=9`, response-quantile partition with `α₀=0.9`), CRIME (communities & crime,
`d=15`, smallest-population district is the target agent), STAR (Tennessee
Student–Teacher Achievement Ratio, `d=15`, rural early-age students are the
target agent), DERMA (DermaMNIST, `28×28×3`, 7 classes), TISSUE (TissueMNIST,
`28×28×1`, 8 classes). Reported `n/m`: CRIME `30/500`, the other four `30/1000`.

For the three regression datasets the entry scripts pass a target pool of size
`n_cli`, which `procedure.py:308` splits as
`calTrAgent, calAgent = agent_tar.splitAgent(n_cli // 2)` — so the **calibration
size reported in Table 1 is `n_cli // 2`**. That is why the authors' own result
files are `P_60_500_1334_0_15` (CRIME), `P_60_1000_2000_4` (BIO) and
`P_60_1048_2446_10_1_1` (STAR) while the table column reads `n = 30`. The two
classification datasets pass `n` straight through
(`procedure.py:540`), hence `P_30_1000_2000_0.035` (DERMA) and
`P_30_1000_2000_0.07` (TISSUE).

**Exact quantified statement under test** (as recorded in the challenge claim
set): StCP/SLCP reduces prediction-set-size standard deviation by **20–48% when
built on GLCP** and by **6–29% when built on CQR**, while maintaining marginal
coverage near the nominal 90% level.

Published Table 1 Std percentages, recovered from the paper (GLCP-type /
CQR-type, `ours` and `ours-sel`):

| Dataset | GLCP ours | GLCP ours-sel | CQR ours | CQR ours-sel |
|---|---|---|---|---|
| CRIME | 42.9% | 27.7% | 25.0% | 19.4% |
| BIO | 35.7% | 8.3% | 29.3% | 12.3% |
| STAR | 48.4% | 42.9% | 7.3% | 6.7% |
| DERMA | 30.4% | 35.8% | 22.1% | 29.3% |
| TISSUE | 13.5% | 4.8% | 15.4% | 11.5% |

Note for honest scoring: the claim's stated GLCP band **20–48%** matches the
`ours` column on CRIME/BIO/STAR/DERMA but **not** TISSUE (13.5%), and does not
cover the `ours-sel` column. The CQR band **6–29%** does cover the observed
`ours` and `ours-sel` values (6.7%–29.3%). The reproduction must therefore
report the full per-cell matrix and adjudicate the band against it, rather than
report a single flattering number.

### Claim 5 — Table 2 (LogAbs simulation)

DGP fully specified in §5.2 / Appendix C.2, `d = 5`:

- `μ_s = 0`, `μ_t = 1_d / (2√d)`;
- `X' ~ N(μ_s, I_d)`, `Y' = (3/d) Σⱼ X'ⱼ + ε'`;
- `X ~ N(μ_t, I_d)`, `Y = (2/d) Σⱼ Xⱼ + ε`;
- `ε' | X' ~ N(0, σ²(X'; γ_s))`, `ε | X ~ N(0, σ²(X; γ_t))`;
- `σ_logabs(x; γ) = √γ · (Σⱼ log(1+|xⱼ|)) / √d`;
- `γ_s = 1.2`, `γ_t = 1.0`; `m = 500`, `n ∈ {30, 100, 500}`; 50 repeats.

Matches `SimuAnalysis/core.py:sigma(...,"logabs")` and `generate_agent(...)`
with `me_t = d/2`, `me_s = d/3`, `r = 0.5`.

**Exact quantified statement under test:** the largest stability gains occur at
`n = 30`, with Std reductions of **31.2% (GLCP-based)** and **16.3%
(CQR-based)**. Published `ours` percentages: GLCP `31.2 / 23.2 / 26.6` and CQR
`16.3 / 16.7 / 6.3` for `n = 30 / 100 / 500`. Note the CQR maximum over the
three `n` is `16.7%` at `n = 100`, marginally above the `n = 30` value of
`16.3%` — the "largest gains at n=30" wording is therefore exactly true for
GLCP and off by `0.4` points for CQR. This must be adjudicated, not smoothed
over.

### Claim 6 — Theorem 4.7 (data-driven λ selection)

Selection rule (§3.1): with `q_L = Q(1−α−α_tol; F̂_S^0)`,
`q_U = Q(1−α+α_tol; F̂_S^0)`,
`Λ_feas = { λ ∈ Λ : q̂_{St,λ} ∈ [q_L, q_U] }`, take `λ̂ = max Λ_feas` and
`q̂_{St-sel} = q̂_{St,λ̂}`.

Theorem 4.7: if `S₁,…,S_{n+1}` are exchangeable, then

> `P(Y_{n+1} ∈ Ĉ_{St-sel}(X_{n+1})) ∈ [1−α−α_tol, 1−α+α_tol+(n+1)⁻¹)`.

**Assumption.** Exchangeability of the `n` calibration scores and the test
score — nothing else. No parametric model, no accuracy condition on
`F̂_{S|X}`. This makes the theorem finite-sample and distribution-free, and its
prediction is a **concrete numeric interval**: with `α = 0.1`, `α_tol = 0.02`,
`n = 30` the band is `[0.88, 0.9522581)` (`0.9 + 0.02 + 1/31`).

Do not confuse this with the *annotation* band used by the Table 1/2 captions,
`[1−α−0.01, 1−α+1/(n+1)] = [0.89, 0.932258]`, which only decides where a `−`/`+`
superscript is printed. Theorem 4.7's guarantee band is the wider one.

**Quantifier structure.** Universally quantified over exchangeable score
sequences. A finite experiment corroborates but cannot prove it; a violation at
any assumption-satisfying instance would falsify it. The band is two-sided, so a
negative control that breaks exchangeability should be able to push coverage out
of the band — otherwise the check is vacuous.

## Implementation detail that changes what is testable: `check_order`

`Main/SLCP.py:8` defines `check_order(part1, delta)`, and `tune_lbd_list` calls it
after fitting every lambda. It scans all pairs `(i, j)` with `i < j` and flags any
that breaks the expected Pareto ordering — either `delta` failing to decrease
while `part1` also decreases, or both failing to move apart. The offending
tuner is then replaced by a deepcopy of its neighbour and re-trained at its own
lambda, and the scan repeats up to `3 * len(lbd_list)` times.

Consequence for evidence design: **the monotone regularisation path visible in
`tune_lbd_list` output is enforced, not emergent.** Any check of the form "delta
decreases in lambda" or "the discrepancy increases in lambda" run against that
output is circular — it verifies a repair loop, not Equation 7. The Claim 1
evidence therefore re-measures the path by calling `Tuner.tune_marginal`
directly for each lambda on independent deepcopies, which is exactly the call
`tune_lbd_list` makes in its own loop, minus the repair step. Both paths are
published so the size of the difference is visible to a reviewer.

## Units trap: `SLCP.q` is a probability, not a score

`SLCP.load_tuner` sets `self.q = np.quantile(self.beta, (1-alpha)(n+1)/n)` where
`beta` are CDF values, and `self.q` is then passed as the `q` argument of
`Generator.quantile(testX, self.q, ...)` (`Main/SLCP.py:90`), whose signature
documents `q: float = .9` as a probability. It is therefore a *level*, not a
score threshold, and must not be compared against `np.quantile(cal_scores, ...)`.
Doing so produced an apparent discrepancy of ~29 median score spacings that was
purely a units error. The tell was that the reported "quantiles" clustered near
0.90–0.93 — the nominal coverage level — across settings where the score
quantiles ranged over 1.7–2.7.
