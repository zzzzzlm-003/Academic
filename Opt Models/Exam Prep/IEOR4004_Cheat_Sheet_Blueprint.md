# IEOR4004 Cheat Sheet Blueprint
**A4 double-sided, handwritten only. Copy this onto paper — that's the rule.**

Layout below is split into 2 sides. Each side is what to write on one face of the A4.

---

## SIDE 1 — NLP (Q1 ammunition, ~70% of value)

### Block 1A: Convexity / Hessian (top-left, ~1/4 page)

```
GRADIENT  ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ          stationary pt: ∇f = 0
HESSIAN   H = [∂²f/∂xᵢ∂xⱼ]   (symmetric, n×n)

H eigenvalues       f is        critical pt
all > 0  (PD)       convex      LOCAL MIN
all < 0  (ND)       concave     LOCAL MAX
all ≥ 0  (PSD)      convex*     valley (inconcl.)
all ≤ 0  (NSD)      concave*    ridge (inconcl.)
mixed     (indef)   neither     SADDLE

2x2 PD test:  H = [[a,b],[b,c]]   ⇔ a>0 AND ac − b² > 0
3x3:  compute eigenvalues / leading principal minors all > 0

CONVEX problem (convex f, convex feasible set) ⇒ local min = GLOBAL min
```

### Block 1B: Lagrangian (equality) (top-right, ~1/4 page)

```
PROBLEM:  min f(x)   s.t.  hᵢ(x) = 0   i=1..m
LAGRANGIAN:  L(x,λ) = f(x) + Σ λᵢ hᵢ(x)

NECESSARY (∇L = 0):
  ∇ₓL = ∇f + Σ λᵢ ∇hᵢ = 0
  ∂L/∂λᵢ = hᵢ(x) = 0    (feasibility)

SOLUTION RECIPE:
  1. Form L
  2. Compute ∂L/∂xⱼ = 0  for all j
  3. Compute ∂L/∂λᵢ = 0  for all i  (= constraints)
  4. Solve the system (often by trial elimination)
  5. Verify: if f convex (resp. concave) on feasible set,
     critical pt is global min (resp. max).

EXAMPLE: max −2x²−y²+xy+8x+3y, s.t. 3x+y=10
  L = −2x²−y²+xy+8x+3y + λ(10−3x−y)
  ∂x: −4x+y+8−3λ=0   ∂y: −2y+x+3−λ=0   ∂λ: 10−3x−y=0
  → λ=1/4, x=69/28, y=73/28.  f concave (H eigvals −4,−2 < 0) ⇒ global max.
```

### Block 1C: KKT (inequality) — THIS IS THE BIG ONE (middle, ~1/3 page)

```
PROBLEM:  min f(x)   s.t.  gᵢ(x) ≤ 0   i=1..m
LAGRANGIAN:  L(x,λ) = f(x) + Σ λᵢ gᵢ(x)

KKT CONDITIONS (necessary at optimal x*):
  (1) PRIMAL FEASIBLE:   gᵢ(x*) ≤ 0  for all i
  (2) SIGN restriction:  λᵢ sign per table below
  (3) COMPLEMENTARY SLACKNESS:  λᵢ · gᵢ(x*) = 0  for all i
       (so either gᵢ active with =, or λᵢ = 0)
  (4) STATIONARITY:  ∇f(x*) + Σ λᵢ ∇gᵢ(x*) = 0

SIGN RESTRICTION TABLE:
  min  +  gᵢ ≤ 0   →   λᵢ ≥ 0
  min  +  gᵢ ≥ 0   →   λᵢ ≤ 0
  max  +  gᵢ ≤ 0   →   λᵢ ≤ 0
  max  +  gᵢ ≥ 0   →   λᵢ ≥ 0
  (mnemonic: λᵢ = ∂obj/∂(rhsᵢ); for min+≤, tightening rhs hurts → λ ≥ 0)

CASE-ANALYSIS RECIPE:
  For each constraint i:  either λᵢ = 0  OR  gᵢ(x*) = 0
  Branch on cases (2^m cases at most; many infeasible quickly)
  For each candidate, check: primal feas + sign restriction
  Best feasible candidate = optimum.

EXAMPLE: max f(x₁,x₂)  s.t.  x₁²+x₂² ≤ 5,  x₁−x₂ ≤ 1
  Stationarity from L = f + λ₁g₁ + λ₂g₂
  Try λ₁=0 → contradicts; λ₂=0 → infeasible
  ⇒ both active: x₁²+x₂²=5, x₁−x₂=1 → (2,1) [the (−1,−2) violates λ sign]
  λ₁ = −2/3, λ₂ = −1/3  (both ≤ 0 ✓ for max + ≤)
```

### Block 1D: Algorithms (bottom, ~1/4 page)

```
GRADIENT DESCENT (Steepest):
   xₖ₊₁ = xₖ − t · ∇f(xₖ)
   t = step size (line search or fixed e.g. 0.01)
   Stop when ||∇f|| < ε or step < δ
   Pros: simple. Cons: slow near optimum (zigzag).

NEWTON'S METHOD (Deflected gradient):
   xₖ₊₁ = xₖ − [H(xₖ)]⁻¹ · ∇f(xₖ)
   Fixed step size (curvature info baked in)
   Pros: fast quadratic convergence near optimum.
   Cons: needs PD Hessian; expensive to invert; can fail if x₀ far.

RAW EXAMPLE (NLP Ex 5):  min (x₁−3)² + (x₂−2)², t=0.01, x₀=(1,1)
   ∇f = [2(x₁−3), 2(x₂−2)]
   x₁ = (1,1) − 0.01·[−4,−2] = (1.04, 1.02)
   x₂ = (1.04,1.02) − 0.01·[−3.92,−1.96] = (1.0792, 1.0396)
   ... → (3,2) in ~59 iterations.
```

---

## SIDE 2 — LP / IP / Network (Q2 ammunition, pick by topic)

### Block 2A: LP (top-left, ~1/3 page)

```
LP STD FORM:     min cᵀx  s.t.  Ax = b, x ≥ 0
LP GRAPHICAL (2 vars):
   1. Plot constraint lines, shade feasible region
   2. Sweep iso-line of cᵀx in improving direction
   3. Optimum at vertex; solve binding lines for (x*,y*)

SIMPLEX (concept):
   - basic vars (m) + nonbasic (=0)
   - entering: most negative reduced cost cⱼ − cᵦ B⁻¹ Aⱼ (max problem)
   - leaving: min ratio b̄ᵢ / āᵢⱼ over āᵢⱼ > 0
   - unbounded: no entering positive ratio
   - infeasible: artificial var > 0 in optimal

DUALITY:
   PRIMAL:  max cᵀx s.t. Ax ≤ b, x ≥ 0
   DUAL:    min bᵀy s.t. Aᵀy ≥ c, y ≥ 0
   Weak:   cᵀx ≤ bᵀy  always
   Strong: at optimum  cᵀx* = bᵀy*
   Comp Slack:  yᵢ·(bᵢ − Aᵢx) = 0  AND  xⱼ·(c − Aᵀy)ⱼ = 0

SENSITIVITY:
   shadow price = yᵢ* = ∂(obj)/∂bᵢ   (within basis-stable range)
   reduced cost = c − Aᵀy gives range of cⱼ keeping current basis
```

### Block 2B: IP (top-right, ~1/3 page)

```
LOGICAL CONDITIONS:
   x ∈ {a,b,c}     →  x = a y₁+b y₂+c y₃, Σyᵢ=1, yᵢ binary
   x = 0 OR L≤x≤U  →  L y ≤ x ≤ U y, y binary
   "either A or B"  →  A − M(1−y) ≤ rhs, B − My ≤ rhs (big-M)
   fixed cost f if x>0  →  obj += f y, x ≤ M y
   ≤ k of n constraints active  →  use yᵢ + big-M

BRANCH & BOUND (max problem):
   1. Solve LP relaxation → z̄ (upper bound)
   2. If integer → done (incumbent)
   3. Else pick fractional xⱼ = vⱼ; branch:
        L: add  xⱼ ≤ ⌊vⱼ⌋
        R: add  xⱼ ≥ ⌈vⱼ⌉
   4. Solve each LP; FATHOM if:
        - infeasible
        - bound ≤ best incumbent (no improvement possible)
        - integer (update incumbent if better)
   5. Continue until all leaves fathomed.

CUTTING PLANES: add valid inequalities that cut off LP fractional sols
                but don't lose any integer feasible pts.
                Branch + Cut = B&B with cuts during tree.
```

### Block 2C: Network (bottom, ~1/3 page)

```
SHORTEST PATH s→t:    min Σ cᵢⱼ xᵢⱼ
   x ∈ {0,1}; flow conservation: Σⱼxᵢⱼ − Σⱼxⱼᵢ = bᵢ
   bₛ=1, b_t=−1, bᵢ=0 elsewhere.

MIN SPANNING TREE:
   yᵢⱼ ∈ {0,1}; Σyᵢⱼ = |V|−1; no cycles (or use cut constraints)
   Algorithms: Kruskal (sort edges, add if no cycle); Prim (grow from a node)

MAX FLOW s→t:       max  v
   0 ≤ fᵢⱼ ≤ uᵢⱼ
   Σⱼ fᵢⱼ − Σⱼ fⱼᵢ = 0  for i ≠ s,t
   Σⱼ fₛⱼ = v;  Σⱼ fⱼ_t = v
   Algorithm: Ford-Fulkerson (find augmenting path, push residual)
   Max-flow = Min-cut.

TRANSPORTATION (sources i, sinks j):
   min Σᵢⱼ cᵢⱼ xᵢⱼ
   Σⱼ xᵢⱼ ≤ sᵢ  (supply); Σᵢ xᵢⱼ ≥ dⱼ  (demand); xᵢⱼ ≥ 0
   Balanced if Σsᵢ = Σdⱼ.

ASSIGNMENT (n×n):
   xᵢⱼ ∈ {0,1}; Σⱼxᵢⱼ = 1, Σᵢxᵢⱼ = 1
   Algorithm: Hungarian method.

TSP (MTZ formulation):
   xᵢⱼ ∈ {0,1} arc used; uᵢ continuous
   Σⱼxᵢⱼ = 1 (out), Σᵢxᵢⱼ = 1 (in)
   uᵢ − uⱼ + n·xᵢⱼ ≤ n−1   (i≠j, both ≠1)   ← prevents subtours
```

---

## Sheet-writing tips

1. **Write small, but use boxes/dividers.** Pen-and-paper layout: divide each side into 4 quadrants with light pencil lines first.
2. **Worked examples > raw formulas.** Beside each formula, write a tiny example (2-3 lines) — it's much faster to recognize a problem pattern from a worked example.
3. **The KKT block is the most important** — give it the most space.
4. **Don't waste space on what's intuitive.** E.g. "x ≥ 0" non-negativity, basic algebra rules — skip.
5. **Include the sign-restriction table verbatim** — it's tiny and saves you in the exam.
6. **Color code if allowed** (check if "handwritten" prohibits color — usually OK):
   - black = formulas
   - blue = examples
   - red = "DO NOT FORGET"
7. **Last line of side 2:** write down the formulation pattern: "Decision vars → Objective → Constraints (capacity / demand / balance / logic / non-neg)" as a generic checklist for any modeling question.
