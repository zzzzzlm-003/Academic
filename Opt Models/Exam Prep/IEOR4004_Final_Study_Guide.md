# IEOR4004 Final Quiz — 4-Day Study Guide
**Exam: Mon May 11, 2026, 1:10–4:00 PM (75 min designed) | 1 handwritten A4 double-sided cheat sheet allowed**

---

## 0. Exam structure (per Prof. Kaya's announcement)

| # | Topic | Weight |
|---|---|---|
| Q1 | **Nonlinear Programming** — modeling, optimality conditions, algorithms | Heavy |
| Q2 | **Randomly chosen** from LP / IP / Network | Lighter |

**Strategy:** Spend ~70% of time on NLP, ~30% spread across LP/IP/Network.

---

## 1. NLP — what you must know cold (Q1)

### 1.1 Convexity check via Hessian

Given `f(x₁, ..., xₙ)`, compute:
- **Gradient** ∇f = vector of first partials
- **Hessian** H = matrix of second partials (symmetric)

Classify by **eigenvalues of H**:

| Hessian | Eigenvalues | Function is | Critical point is |
|---|---|---|---|
| Positive definite (PD) | All > 0 | Convex | Local **min** |
| Negative definite (ND) | All < 0 | Concave | Local **max** |
| Positive semi-def (PSD) | All ≥ 0 (some =0) | Convex | "Valley" (inconclusive) |
| Negative semi-def (NSD) | All ≤ 0 (some =0) | Concave | "Ridge" (inconclusive) |
| Indefinite | Mixed signs | Neither | Saddle point |

**Convex problem rule:** convex objective + convex feasible region ⇒ any local min is the **global** min.

**Worked example (NLP Ex Q3):**
f(x₁,x₂,x₃) = x₁² + x₂² + 3x₃² − x₁x₂ − x₂x₃ − x₁x₃

H = ⎡  2  −1  −1 ⎤
    ⎢ −1   2  −1 ⎥
    ⎣ −1  −1   4 ⎦

Eigenvalues ≈ [0.44, 3.00, 4.56] — all positive ⇒ **convex**.

For 2x2 H = [[a,b],[b,c]]: PD iff a>0 and det = ac−b² > 0. (Quick check.)

---

### 1.2 Modeling NLP word problems

**Pattern:** decision vars → objective (revenue − cost) → constraints (budget, capacity, demand, balance).

**Worked example (NLP Ex Q1):** Capital K at $4/unit, labor L at $1/unit, $8 budget, output = KL.
```
max  z = K·L
s.t. 4K + L ≤ 8
     K, L ≥ 0
```
Quadratic objective — NLP. (Could also solve with Lagrangian: K*=1, L*=4.)

**Worked example (NLP Ex Q9 — Markowitz QP):** Portfolio with stocks i=1,2,3, return Sᵢ.
- Decision: xⱼ = $ in stock j
- Variance of portfolio = Σᵢⱼxᵢxⱼ·cov(Sᵢ,Sⱼ) — quadratic
- min variance, s.t. expected return ≥ target, Σxⱼ = budget, xⱼ ≥ 0

**Tip:** if you see "variance" or "ratio" or "product of decision vars" → it's NLP.

---

### 1.3 Equality constraints — Lagrangian

**Setup:** min f(x)  s.t. h(x) = 0

**Lagrangian:** L(x, λ) = f(x) + λ·h(x)   (or − λ·h(x); both work, sign of λ flips)

**Necessary condition:** ∇L = 0, i.e.,
- ∂L/∂xⱼ = 0  for all j   →  ∇f = −λ·∇h (gradient of obj is linear combo of constraint gradients)
- ∂L/∂λ = 0   →  h(x) = 0 (feasibility)

**Worked example (NLP Ex Q7):** max z = −2x² − y² + xy + 8x + 3y, s.t. 3x + y = 10.

L = −2x² − y² + xy + 8x + 3y + λ(10 − 3x − y)

∂L/∂x: −4x + y + 8 − 3λ = 0   ...(1)
∂L/∂y: −2y + x + 3 − λ = 0    ...(2)
∂L/∂λ: 10 − 3x − y = 0        ...(3)

Solve linear system → λ = 1/4, x = 69/28, y = 73/28.

**Confirm optimum:** Hessian of f is [[−4,1],[1,−2]] ⇒ eigenvalues both negative ⇒ f concave ⇒ this critical point is the **global max**.

---

### 1.4 Inequality constraints — KKT conditions

**Setup:** min f(x)  s.t. gᵢ(x) ≤ 0  for i=1..m

**Lagrangian:** L(x, λ) = f(x) + Σᵢ λᵢ·gᵢ(x)

**The four KKT conditions** (necessary for any optimal x*):

1. **Primal feasibility:** gᵢ(x*) ≤ 0 for all i
2. **Sign restriction on λᵢ** (depends on min/max + ≤/≥): see table below
3. **Complementary slackness:** λᵢ · gᵢ(x*) = 0 for each i (either constraint is **active** with equality, or λᵢ=0)
4. **Stationarity:** ∇f(x*) + Σᵢ λᵢ ∇gᵢ(x*) = 0

#### Sign-restriction table (memorize!)

| Problem | Constraint form | λᵢ sign |
|---|---|---|
| **min** | gᵢ(x) ≤ 0 | λᵢ ≥ 0 |
| **min** | gᵢ(x) ≥ 0 | λᵢ ≤ 0 |
| **max** | gᵢ(x) ≤ 0 | λᵢ ≤ 0 |
| **max** | gᵢ(x) ≥ 0 | λᵢ ≥ 0 |

(Mnemonic: λᵢ is the "shadow price" — how the obj changes when RHS increases. For a min problem with ≤, tightening ↑ obj, so λ ≥ 0.)

**KKT solution algorithm — case analysis on λᵢ:**

For each constraint i, branch:
- **Case A:** λᵢ = 0 (constraint not binding) → solve stationarity ignoring constraint i
- **Case B:** λᵢ ≠ 0, so gᵢ(x*) = 0 (constraint binding) → use it as equality

Try cases, find candidate (x*, λ*), check sign restrictions and primal feasibility.

**Worked KKT example (NLP3 lecture):** max f(x₁,x₂), s.t. x₁² + x₂² ≤ 5  AND  x₁ − x₂ ≤ 1

After case analysis (λ₁=0 leads to contradictions): both constraints active.
From x₁² + x₂² = 5 and x₁ − x₂ = 1: candidates (2,1) and (−1,−2).
Sub into stationarity → (2,1) gives λ₁ = −2/3, λ₂ = −1/3 — both ≤ 0 ✓ (correct sign for max + ≤).
**Optimal:** (x₁*, x₂*, λ₁*, λ₂*) = (2, 1, −2/3, −1/3).

---

### 1.5 Algorithms — Gradient Descent (Steepest Descent)

For unconstrained min f(x):

```
1. Pick starting point x₀, step size t (e.g., 0.01), tolerance ε
2. Compute gradient g = ∇f(xₖ)
3. If ||g|| < ε, stop; else
4. xₖ₊₁ = xₖ − t·g                ← key update
5. Go to 2
```

**Worked example (NLP Ex Q5):** min Z = (x₁−3)² + (x₂−2)², start (1,1), t = 0.01.

∇f = [2(x₁−3), 2(x₂−2)]

Iter 1: g = [2(1−3), 2(1−2)] = [−4, −2]. x₁ = (1,1) − 0.01·[−4,−2] = (1.04, 1.02)
Iter 2: g = [2(1.04−3), 2(1.02−2)] = [−3.92, −1.96]. x₂ = (1.04, 1.02) − 0.01·[−3.92,−1.96] = (1.0792, 1.0396)
... (converges to (3,2) in ~59 iterations)

**Pros/cons:** simple, but slow near the optimum.

---

### 1.6 Algorithms — Newton's Method (Deflected Gradient)

For unconstrained min f(x), uses **Hessian** for curvature:

```
xₖ₊₁ = xₖ − H(xₖ)⁻¹ · ∇f(xₖ)
```

Step size is **fixed** (no t·); curvature info baked in.

**Pros/cons:** very fast convergence near optimum, but expensive (compute & invert H), and can fail if starting far from optimum or H not PD.

**For exam:** know the formula, know it uses Hessian, know it's "deflected gradient" (correcting the steepest-descent direction with curvature).

---

## 2. LP — quick refresher (in case Q2 = LP)

### 2.1 Graphical method (≤ 2 vars)
1. Graph constraints, shade feasible region
2. Draw an iso-line of objective; sweep in improving direction
3. Optimum at a vertex (corner) where iso-line just touches feasible region
4. Solve the 2 binding constraints simultaneously to get coordinates

### 2.2 Simplex method — concept-level
- Walks from vertex to vertex along edges
- At each step picks entering variable (most negative reduced cost for max) and leaving variable (min ratio test)
- Stops when no improving direction
- For exam: understand **basic vs nonbasic**, ratio test, when LP is **unbounded** (no leaving var) or **infeasible**

### 2.3 Duality
- Every LP (P) has a dual (D). max P ↔ min D, ≤ ↔ ≥, etc.
- **Weak duality:** any feasible primal x and any feasible dual y satisfy cᵀx ≤ bᵀy (max case)
- **Strong duality:** if both have optima, val(P) = val(D)
- **Complementary slackness:** at optimum, for each i: yᵢ·(slackᵢ) = 0 and xⱼ·(reduced costⱼ) = 0
  - Use to solve dual from primal solution

### 2.4 Sensitivity analysis
- How does optimum change if cⱼ, bᵢ, or aᵢⱼ shifts?
- **Ranges**: of cⱼ where current basis stays optimal; of bᵢ where current basis stays feasible
- **Shadow price** = dual value of constraint i = ∂(obj)/∂bᵢ

---

## 3. IP — quick refresher (most likely Q2 if not LP)

### 3.1 Modeling logical conditions

| Want | Modeling |
|---|---|
| x ∈ {2, 7, 12} | x = 2y₁ + 7y₂ + 12y₃, Σyᵢ = 1, yᵢ ∈ {0,1} |
| x ∈ {0, 2, 7, 12} | as above with Σyᵢ ≤ 1 |
| x = 0 OR L ≤ x ≤ U | L·y ≤ x ≤ U·y, y ∈ {0,1} |
| Either constraint A OR B | A − M(1−y) ≤ rhs and B − My ≤ rhs, big-M |
| If x>0 then fixed cost f | obj += f·y, with x ≤ M·y, y∈{0,1} |
| At most k of n constraints | introduce yᵢ for each, Σyᵢ ≥ n−k, deactivate via big-M |

### 3.2 Branch & Bound algorithm

```
1. Solve LP relaxation (drop integrality). Call its optimum z̄ (this is an upper bound for max problem).
2. If solution is integer → done.
3. Else, pick a fractional variable xⱼ = vⱼ. Branch into 2 subproblems:
       — Subproblem A: add constraint xⱼ ≤ ⌊vⱼ⌋
       — Subproblem B: add constraint xⱼ ≥ ⌈vⱼ⌉
4. Solve each subproblem (LP relaxation).
5. Fathom (prune) a branch if:
       — infeasible
       — LP relaxation bound is worse than current best integer solution
       — solution is integer (update incumbent if better)
6. Continue until all branches fathomed.
```

**Best integer found so far** = "incumbent". The bound from LP relaxation is what you compare against.

### 3.3 Cutting planes (concept)
- Add valid inequalities that cut off fractional LP solutions but keep all integer feasible points
- "Branch and cut" = combine B&B with cuts during the tree search
- The best cuts are **facets** of the convex hull of integer solutions

---

## 4. Network — quick refresher (least likely Q2 but possible)

| Problem | Decision var | Key constraint |
|---|---|---|
| **Shortest path** s→t | xᵢⱼ ∈ {0,1} use arc | Flow conservation: 1 leaves s, 1 enters t, 0 elsewhere |
| **Min spanning tree** (Kruskal/Prim) | yᵢⱼ ∈ {0,1} include edge | Σyᵢⱼ = n−1, no cycles (tree) |
| **Max flow** s→t | fᵢⱼ ≥ 0 flow on arc | 0 ≤ fᵢⱼ ≤ uᵢⱼ; flow conservation at intermediate nodes; max f_out(s) |
| **Transportation** (sources i, sinks j) | xᵢⱼ ≥ 0 ship | Σⱼxᵢⱼ ≤ supplyᵢ, Σᵢxᵢⱼ ≥ demandⱼ |
| **Assignment** (n people ↔ n tasks) | xᵢⱼ ∈ {0,1} | Σⱼxᵢⱼ = 1, Σᵢxᵢⱼ = 1 |
| **TSP (MTZ)** | xᵢⱼ ∈ {0,1} traverse arc, uᵢ continuous | degree=1 in/out, MTZ: uᵢ − uⱼ + n·xᵢⱼ ≤ n−1 (no subtour) |

**Min-cost flow general form:** min Σ cᵢⱼ·xᵢⱼ, s.t. flow conservation: Σⱼ xⱼᵢ − Σⱼ xᵢⱼ = bᵢ (bᵢ>0 sink, <0 source, =0 transshipment), 0 ≤ xᵢⱼ ≤ uᵢⱼ.

---

## 5. Four-day plan

### Day 1 (Wed May 6) — NLP scaffolding
- Read **WrapUp.pdf** pages 37–63 (NLP section, 27 pages)
- Read **NLP1.pdf** focusing on: convexity, Hessian, QP examples (skim NLP code)
- Re-do **NLP_Exercises Q3** (convexity), **Q1** (modeling)
- Goal: be able to write the convexity table from memory + form a Lagrangian

### Day 2 (Thu May 7) — NLP optimality conditions
- Read **NLP2.pdf** (Lagrangian, equality case)
- Read **NLP3.pdf** pages 1–24 (KKT, KKT example)
- Re-do **NLP_Exercises Q7** (Lagrangian), **Q4** (concavity + max)
- Practice: write all 4 KKT conditions with sign-restriction table from memory
- This is the highest-yield day. KKT will almost certainly appear on the exam.

### Day 3 (Fri May 8) — Algorithms + Q2 prep
- Morning: **NLP3.pdf** pages 25–54 (gradient descent, Newton's method)
- Re-do **NLP_Exercises Q5** (gradient descent — do 3 iterations by hand on paper)
- Afternoon: review LP/IP/Network using **WrapUp.pdf** pages 3–36
  - LP: graphical method + duality theorems
  - IP: branch & bound (do **In Class Exercise 5** on B&B by hand)
  - Network: just memorize the formulation table above
- Build draft of cheat sheet (see Section 6 below)

### Day 4 (Sat May 9 / Sun May 10) — Cheat sheet + mock
- Finalize handwritten cheat sheet (Section 6)
- Do a 75-min timed mock using one NLP exercise + one IP exercise from the practice PDFs
- Re-read your cheat sheet 3x; visualize where each formula lives

### Sunday night
- Light review only. Sleep.

---

## 6. Cheat sheet blueprint (A4 double-sided, handwritten)

See `IEOR4004_Cheat_Sheet_Blueprint.md` — that's a separate file with everything you should copy onto your A4.

---

## 7. Why your current grade is OK

- Assignments: 3 × 100/100, plus 4–5 graded → 15% locked.
- Project 1 final report: 96.22/100 → most of project 1's 22.5% is locked.
- Project 2 submitted Apr 26.
- Quiz 1: 91 (Section 2 adjusted) → already good.
- Final Quiz is at most 15% of final grade. Even a mediocre final won't tank you.

**Target:** comfortable B+ to A−. Not aiming for perfect — aiming for "I knew enough KKT and B&B to write coherent answers."

---

## 8. Files in your folder you should reference

| File | Purpose |
|---|---|
| `WrapUp.pdf` | THE review session slides (May 4 Panopto) — your most valuable file |
| `NLP1.pdf` / `NLP2.pdf` / `NLP3.pdf` | Full lecture decks, all NLP content |
| `IEOR4004_NLP_Exercises.pdf` | 9 NLP problems with full solutions — gold |
| `IEOR4004_LP_Exercises.pdf` / `_IP_Exercises.pdf` / `_Network_Exercises.pdf` | Topic practice |
| `IEOR4004_InClassExercise5.pdf` | B&B by hand (do this!) |
| `IEOR4004_InClassExercises6.pdf` | Network MIP formulation (network practice) |
| `Solutions to Assignments & Quizzes/` | Past quiz answers — see how the prof grades |
