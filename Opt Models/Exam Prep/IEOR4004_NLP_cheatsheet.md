# IEOR 4004 — NLP Cheat Sheet (One-Pager)

*For final exam, Mon May 11, 1:10–3:10 PM, IAB 417*

---

## 1. Convexity (the whole game hinges on this)

**Convex set:** `x, y ∈ S ⟹ λx + (1−λ)y ∈ S` for all `λ ∈ [0,1]`.

**Convex function:** `f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)`.

**Twice-differentiable test:** `f` convex `⟺ ∇²f(x) ⪰ 0` (PSD) on the domain.
Strictly convex if `∇²f ≻ 0`. Concave = `−f` convex.

**Operations that preserve convexity:**
- Non-negative weighted sum: `α₁f₁ + α₂f₂` (αᵢ ≥ 0) ✓
- Composition with affine: `f(Ax + b)` ✓
- Pointwise max: `max{f₁, …, fₖ}` ✓
- Affine functions are **both** convex and concave

**Common convex:** `x²`, `eˣ`, `−log x`, `‖x‖`, `xᵀQx` with `Q ⪰ 0`.
**Common concave:** `log x`, `√x` (on `x ≥ 0`).

---

## 2. Unconstrained Optimization — `min f(x)`

| Condition | Statement | Type |
|---|---|---|
| FONC | `∇f(x*) = 0` | necessary |
| SONC | `∇²f(x*) ⪰ 0` | necessary |
| SOSC | `∇f(x*) = 0` and `∇²f(x*) ≻ 0` | sufficient (strict local min) |

**If `f` is convex:** any stationary point is a **global** minimum. ⭐

---

## 3. Constrained Problem Setup

$$\min f(x) \quad \text{s.t.} \quad g_i(x) \le 0,\ i=1..m;\quad h_j(x) = 0,\ j=1..p$$

**Lagrangian:**
$$L(x,\lambda,\mu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \mu_j h_j(x)$$

- `λᵢ ≥ 0` (inequality multipliers — sign matters!)
- `μⱼ` free (equality multipliers)

---

## 4. KKT Conditions (the four)

At a local optimum `x*` (under constraint qualification, e.g. LICQ):

1. **Stationarity:** `∇f(x*) + Σ λᵢ ∇gᵢ(x*) + Σ μⱼ ∇hⱼ(x*) = 0`
2. **Primal feasibility:** `gᵢ(x*) ≤ 0`, `hⱼ(x*) = 0`
3. **Dual feasibility:** `λᵢ ≥ 0`
4. **Complementary slackness:** `λᵢ · gᵢ(x*) = 0` for each `i`

**Sufficiency:** If `f` and all `gᵢ` are convex and `hⱼ` are affine, then **KKT ⟺ global optimum**. ⭐⭐

---

## 5. KKT Solving Recipe

1. Write `L`.
2. Compute `∇ₓ L = 0` → equations linking `x*`, `λ`, `μ`.
3. **Case-split on complementary slackness.** For each inequality `gᵢ`:
   - **Active:** `λᵢ > 0` and `gᵢ(x*) = 0`
   - **Inactive:** `λᵢ = 0` and `gᵢ(x*) < 0`
4. Solve each case's system. Check:
   - all `λᵢ ≥ 0`
   - all original constraints satisfied
5. Compare `f(x*)` across surviving cases → pick the min.

> With `m` inequalities you have up to `2ᵐ` cases. Most collapse fast — start with "all active" and "all inactive" first.

---

## 6. Duality (if asked)

**Dual function:** `g(λ, μ) = infₓ L(x, λ, μ)`
**Dual problem:** `max g(λ, μ)` s.t. `λ ≥ 0`

- **Weak duality:** `dual opt ≤ primal opt` (always)
- **Strong duality:** equality holds if problem is convex **and** Slater's condition (∃ strictly feasible `x`) holds

---

## 7. Traps to avoid

- Forgetting `λᵢ ≥ 0` — losing this kills the answer
- Skipping the case-split on complementary slackness
- Using KKT as sufficient when problem isn't convex (it's only necessary then)
- Sign convention: standard form is `gᵢ ≤ 0` paired with `λᵢ ≥ 0`. If a problem writes `gᵢ ≥ 0`, flip signs first.
- For equality constraints `hⱼ = 0`, `μⱼ` is **free** in sign

---

## 8. 30-second sanity ritual before submitting any KKT problem

1. ✅ All four KKT conditions written?
2. ✅ All `λᵢ ≥ 0`?
3. ✅ Complementary slackness satisfied for every `i`?
4. ✅ Primal point feasible?
5. ✅ Is the problem convex? If yes, KKT point = global min (you're done).
