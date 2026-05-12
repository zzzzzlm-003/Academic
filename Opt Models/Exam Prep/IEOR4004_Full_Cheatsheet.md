# IEOR 4004 — Full Final Cheat Sheet

*Mon May 11, 1:10–3:10 PM, IAB 417 · NLP is the heaviest weight*

---

## A. LINEAR PROGRAMMING (LP)

### A1. Forms

**Standard form:**  `min cᵀx  s.t.  Ax = b,  x ≥ 0`

**Canonical (max) form:**  `max cᵀx  s.t.  Ax ≤ b,  x ≥ 0`

**Converting:**
- `≤` constraint: add **slack** `s ≥ 0`  →  `aᵀx + s = b`
- `≥` constraint: subtract **surplus** `s ≥ 0`  →  `aᵀx − s = b`
- Free variable `x`: write `x = x⁺ − x⁻`, `x⁺, x⁻ ≥ 0`
- `max cᵀx` ⟺ `−min(−cᵀx)`

### A2. Geometry

- Feasible region = **polyhedron** (intersection of half-spaces)
- **Basic Feasible Solution (BFS)** = vertex of polyhedron
- Build BFS: pick `m` basic vars from `n`, set `n−m` non-basic to 0, solve `B xᵦ = b`, need `xᵦ ≥ 0`
- **Fundamental theorem of LP:** if optimum exists and is finite, some BFS is optimal

### A3. Simplex method (max form)

1. Start at a BFS (tableau form: identity in basic columns)
2. **Reduced cost** of non-basic `j`:  `c̄ⱼ = cⱼ − cᵦᵀ B⁻¹ Aⱼ`
3. **Entering var:** any non-basic with `c̄ⱼ > 0` (for max) — pick most positive, or Bland's smallest-index for anti-cycling
4. **Leaving var (ratio test):**  `min { bᵢ / āᵢⱼ : āᵢⱼ > 0 }`
5. Pivot, repeat. **Stop** when all `c̄ⱼ ≤ 0`.

**Detect:**
- **Unbounded:** entering column has all `āᵢⱼ ≤ 0` (no leaving var)
- **Infeasible:** Phase-1 (or Big-M) ends with artificial > 0
- **Degenerate:** some basic var = 0; risk of cycling → use Bland's
- **Alt. optima:** some non-basic `c̄ⱼ = 0` at the optimum

### A4. Getting a starting BFS

**Two-phase method:** add artificial `aᵢ ≥ 0` to each row, **Phase 1:** `min Σ aᵢ`. If min = 0 → use that BFS for Phase 2 (original objective).

**Big-M:** add artificials with huge penalty `M` in the objective, solve in one pass.

---

## B. LP DUALITY

### B1. Primal–Dual pair

| Primal (min) | Dual (max) |
|---|---|
| `min cᵀx` | `max bᵀy` |
| `Ax ≥ b` | `Aᵀy ≤ c` |
| `x ≥ 0` | `y ≥ 0` |

**Conversion rules (min primal):**
- Primal `≥` constraint ↔ dual var `y ≥ 0`
- Primal `≤` constraint ↔ dual var `y ≤ 0`
- Primal `=` constraint ↔ dual var **free**
- Primal `x ≥ 0` ↔ dual `≤` constraint
- Primal `x` free ↔ dual `=` constraint

### B2. Duality theorems

- **Weak duality:** for any feasible `x, y`,  `cᵀx ≥ bᵀy`
- **Strong duality:** if either has finite optimum, both do and they are equal
- **Complementary slackness** (at optimal `x*, y*`):
  - `yᵢ* · (Aᵢx* − bᵢ) = 0`  for each `i`
  - `xⱼ* · (cⱼ − Aⱼᵀy*) = 0`  for each `j`

### B3. Sensitivity

- **Shadow price** of constraint `i` = optimal dual `yᵢ*` = marginal change in obj per unit change in `bᵢ` (within range)
- **RHS range:** how much can `bᵢ` change while current basis stays optimal? Use `B⁻¹` column.
- **Cost range:** how much can `cⱼ` change while reduced costs keep their signs?

---

## C. INTEGER PROGRAMMING (IP)

### C1. Modeling tricks (binaries `y ∈ {0,1}`)

- **Fixed charge:** cost is `f·𝟙{x>0} + c·x`. Use `x ≤ M·y`, add `f·y` to obj.
- **Either-or** (at least one of `aᵀx ≤ b₁` or `aᵀx ≤ b₂`):
  add `aᵀx ≤ b₁ + M(1−y)`, `aᵀx ≤ b₂ + M·y`
- **If A then B:** `y_A ≤ y_B`
- **At most k of N:** `Σ yᵢ ≤ k`
- **SOS1** (at most one nonzero): `Σ yᵢ ≤ 1`, `xᵢ ≤ M·yᵢ`

### C2. Branch and Bound

1. Solve LP relaxation. If integer → done.
2. Pick a fractional `xⱼ* = v`. **Branch:** `xⱼ ≤ ⌊v⌋` vs `xⱼ ≥ ⌈v⌉`
3. **Bound:** for each node solve LP relaxation. **Prune** if:
   - infeasible, or
   - LP optimum worse than incumbent, or
   - LP optimum is integer (update incumbent)
4. Stop when no live nodes remain.

**Gomory cuts:** add cut from simplex row to chop off fractional optimum without removing any integer point.

---

## D. NETWORK FLOWS

### D1. Shortest path

- **Dijkstra** (non-negative weights): greedy, `O(V²)` or `O(E log V)` with heap
- **Bellman–Ford** (handles negative edges, no negative cycle): relax all edges `V−1` times

### D2. Max flow

- **Ford–Fulkerson:** repeatedly find augmenting `s→t` path in residual graph, push min residual capacity
- **Max-flow Min-cut theorem:** max flow value = min capacity of any `s–t` cut

### D3. Transportation problem

- `m` supplies `sᵢ`, `n` demands `dⱼ`, balanced if `Σsᵢ = Σdⱼ` (add dummy row/col if not)
- **Initial BFS:** NW-corner rule, or min-cost rule, or Vogel's
- **Optimality:** compute potentials `uᵢ, vⱼ` with `uᵢ + vⱼ = cᵢⱼ` on basic cells; reduced cost on non-basic = `cᵢⱼ − uᵢ − vⱼ`; optimal if all ≥ 0 (for min)

### D4. Assignment problem

- Special transportation with `sᵢ = dⱼ = 1`
- **Hungarian method:** subtract row mins, then col mins, cover zeros with min lines; if lines < `n`, adjust uncovered entries; repeat

### D5. Total unimodularity (TU)

- If constraint matrix `A` is TU and `b` is integer → LP relaxation gives integer optimum automatically (no need for IP)
- Transportation, assignment, shortest path, max-flow LPs are all TU

---

## E. CONVEXITY (foundation of NLP)

### E1. Definitions

- **Convex set:** `x, y ∈ S ⟹ λx + (1−λ)y ∈ S` for all `λ ∈ [0,1]`
- **Convex function:** `f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)`
- **Concave:** `−f` convex
- **Affine** = both convex and concave

### E2. Tests

- **1st-order** (if `f` differentiable): `f` convex ⟺ `f(y) ≥ f(x) + ∇f(x)ᵀ(y−x)` ∀x,y
- **2nd-order** (if twice diff): `f` convex ⟺ `∇²f(x) ⪰ 0` (PSD) on domain
  - **Strictly** convex if `∇²f ≻ 0`
- **PSD check:** all eigenvalues ≥ 0, or all leading principal minors ≥ 0

### E3. Operations preserving convexity

- Non-negative weighted sum: `α₁f₁ + α₂f₂`, `αᵢ ≥ 0`
- Composition with affine: `f(Ax + b)`
- Pointwise max of convex functions
- Sum of convex is convex

### E4. Catalogue

- **Convex:** `x²`, `eˣ`, `−log x`, `‖x‖` (any norm), `xᵀQx` with `Q ⪰ 0`, `1/x` on `x>0`
- **Concave:** `log x`, `√x` on `x ≥ 0`

---

## F. UNCONSTRAINED NLP — `min f(x)`

| Condition | Statement | Type |
|---|---|---|
| FONC | `∇f(x*) = 0` | necessary |
| SONC | `∇²f(x*) ⪰ 0` | necessary |
| SOSC | `∇f(x*) = 0` and `∇²f(x*) ≻ 0` | **sufficient** (strict local min) |

**Key result:** if `f` is convex, any stationary point is a **global** minimum. ⭐

---

## G. CONSTRAINED NLP & KKT  ⭐⭐ (most important)

### G1. Problem setup

```
min  f(x)
s.t. gᵢ(x) ≤ 0,   i = 1..m
     hⱼ(x) = 0,   j = 1..p
```

**Lagrangian:**  `L(x, λ, μ) = f(x) + Σᵢ λᵢ gᵢ(x) + Σⱼ μⱼ hⱼ(x)`

- `λᵢ ≥ 0` for inequality constraints
- `μⱼ` **free** for equality constraints

### G2. KKT conditions (the four)

At a local optimum `x*` (under constraint qualification, e.g. LICQ):

1. **Stationarity:**  `∇f(x*) + Σ λᵢ ∇gᵢ(x*) + Σ μⱼ ∇hⱼ(x*) = 0`
2. **Primal feasibility:**  `gᵢ(x*) ≤ 0`, `hⱼ(x*) = 0`
3. **Dual feasibility:**  `λᵢ ≥ 0`
4. **Complementary slackness:**  `λᵢ · gᵢ(x*) = 0` for each `i`

**Sufficiency:** if `f` convex, all `gᵢ` convex, all `hⱼ` affine, then **KKT ⟺ global optimum**. ⭐⭐

### G3. KKT solving recipe

1. Write `L`.
2. `∇ₓ L = 0` → equations linking `x*, λ, μ`.
3. **Case-split on complementary slackness.** For each `gᵢ`:
   - **Active:** `λᵢ > 0`, `gᵢ(x*) = 0`
   - **Inactive:** `λᵢ = 0`, `gᵢ(x*) < 0`
4. Solve each case. Check `λᵢ ≥ 0` and all constraints.
5. Compare `f(x*)` across surviving cases → smallest wins.

> Start with "all inactive" (unconstrained min) and "all active" first — these usually pin down the answer.

### G4. NLP Duality

- **Dual function:** `g(λ, μ) = infₓ L(x, λ, μ)`  →  always concave (no matter what primal looks like)
- **Dual problem:** `max g(λ, μ)` s.t. `λ ≥ 0`
- **Weak duality:** `dual opt ≤ primal opt` always
- **Strong duality:** holds if primal is convex **AND** Slater's condition (∃ strictly feasible `x`)

---

## H. DYNAMIC PROGRAMMING (if on syllabus)

- Decompose into **stages**, with **state** `sₖ` at stage `k`, **decision** `xₖ`
- **Bellman recursion** (min):  `Vₖ(sₖ) = minₓₖ { cₖ(sₖ, xₖ) + Vₖ₊₁(f(sₖ, xₖ)) }`
- Boundary: `V_N(s) = 0` (or terminal cost)
- Solve backwards from `N` to `0`
- Knapsack DP: `V(i, w) = max(V(i−1, w),  vᵢ + V(i−1, w − wᵢ))`

---

## I. COMMON TRAPS

- KKT: forgetting `λᵢ ≥ 0`, or skipping case-split on complementary slackness
- KKT: using sufficiency when problem isn't convex (KKT is only necessary then)
- Sign convention: standard is `gᵢ ≤ 0` paired with `λᵢ ≥ 0`. If problem writes `gᵢ ≥ 0`, **flip the sign** before applying KKT.
- LP duality: getting the sign of `y` wrong for the constraint type (use the conversion table)
- Simplex: forgetting Bland's rule on degenerate problems → cycling
- IP big-M: picking `M` too small (cuts off feasible solutions) or absurdly large (numerical issues)
- Transportation: forgetting to add dummy row/column when unbalanced
- Convexity: checking only `∇²f ⪰ 0` at one point — needs to hold on the whole domain

---

## J. 60-second pre-submission sanity ritual

For every problem, before moving on:

1. ✅ Did I write **all** the optimality conditions? (e.g. all 4 KKT, or LP optimality + feasibility)
2. ✅ Are all multipliers the right sign?
3. ✅ Is my candidate `x*` **feasible** in the original problem?
4. ✅ Is the problem convex? If yes, my KKT point or LP optimum is global — say so.
5. ✅ Did I answer the actual question asked (value of `x*` AND value of `f(x*)`, or shadow price, etc.)?

---

*Tip while copying: write each section's formulas first, then come back to fill in the recipes and traps in your own words. The act of paraphrasing in the margin is what makes it stick.*
