# NLP 理论框架 (讲义总结图扩展版)

## 🌳 NLP 总分类图

```
                    Nonlinear Programming Problems
                   /                                \
        Without Constraints                    With Constraints
         (Unconstrained)                         (Constrained)
              │                                       │
       ┌──────┴──────┐                       ┌────────┴────────┐
       │             │                       │                 │
   Quadratic    Non-Quadratic           Equality         Inequality
    forms          forms              constraints        constraints
       │             │                       │                 │
       ▼             ▼                       ▼                 ▼
    Hessian      Hessian + 迭代算法    Lagrange Multipliers    KKT
    (闭式)        (gradient descent /                       Conditions
                  Newton's method)
```

---

# 1. 凸性 (Convexity) — 整套理论的核心

## 1.1 凸集

集合 `S ⊂ ℝⁿ` 是 **convex set** ⟺ 
```
∀ x, y ∈ S, ∀ λ ∈ [0, 1]:    λx + (1−λ)y ∈ S
```
即"任意两点的连线段也在集合里"。

## 1.2 凸函数 / 凹函数

`f: S → ℝ` 是 **convex function** ⟺
```
f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)
```
即"secant 在 graph 上方"。

**concave** = `−f` convex（"secant 在 graph 下方"）。

**affine** function (`aᵀx + b`) **既凸又凹**。

## 1.3 凸性的二阶判定 (twice-differentiable)

| 条件 | 结论 |
|---|---|
| Hessian `H(x) ⪰ 0` (PSD) ∀x | f convex |
| `H(x) ≻ 0` (PD) ∀x | f strictly convex |
| `H(x) ⪯ 0` (NSD) ∀x | f concave |
| `H(x) ≺ 0` (ND) ∀x | f strictly concave |

## 1.4 保 convexity 的操作

- **非负加权和：** `α₁f₁ + α₂f₂` (`αᵢ ≥ 0`) convex if `fᵢ` convex
- **仿射复合：** `f(Ax + b)` convex if `f` convex
- **逐点最大：** `max{f₁, ..., fₖ}` convex if 所有 `fᵢ` convex
- **常见 convex 函数：** `x²`, `eˣ`, `−log x`, `‖x‖`, `xᵀQx (Q ⪰ 0)`

## 1.5 Local vs Global Optima ⭐

**核心定理：**
- 若 `f` convex, **任何 local min 都是 global min**
- 若 `f` concave, **任何 local max 都是 global max**
- 一般 nonconvex 问题，local 可能不是 global → 必须分析

**Convex Program 定义：**
- min f, s.t. gᵢ ≤ 0, hⱼ = 0
- f convex + 所有 gᵢ convex + 所有 hⱼ affine → **convex program**
- 在 convex program 中：**KKT 是充分条件**，找到 KKT 点即 global optimum

---

# 2. Unconstrained NLP

## 2.1 三大最优条件

| 条件 | 公式 | 必要 / 充分 |
|---|---|---|
| **FONC** (First Order Necessary) | `∇f(x*) = 0` | 必要 |
| **SONC** (Second Order Necessary) | `H(x*) ⪰ 0` | 必要 |
| **SOSC** (Second Order Sufficient) | `∇f(x*) = 0` AND `H(x*) ≻ 0` | **充分** (strict local min) |

⭐ **凸函数的特例：** `f` convex → **FONC 即充分** (FONC + convex → global min)

## 2.2 Hessian → Critical Point Nature

```
∇f(x*) = 0 (stationary point), 然后看 H(x*):
```

| Hessian | x* 的性质 |
|---|---|
| **PD** (正定) | local minimum (strict) |
| **ND** (负定) | local maximum (strict) |
| **PSD** | valley (一片连续相等的最小点) |
| **NSD** | ridge (一片连续相等的最大点) |
| **Indefinite** | **saddle point** (鞍点) |

## 2.3 PD / PSD 判定方法

**方法 A: Sylvester 主子式法 (手算优先)**
```
PD ⟺ 所有 leading principal minors 严格 > 0
ND ⟺ 主子式符号交替 −, +, −, +, ...
PSD ⟺ 所有 (不只 leading) principal minors ≥ 0
```

**方法 B: 特征值法**
```
PD ⟺ 所有特征值 > 0
ND ⟺ 所有特征值 < 0
PSD ⟺ 所有特征值 ≥ 0
Indefinite ⟺ 有正有负
```

**速记：** 对角矩阵 → 特征值就是对角元；一般矩阵不能这么做。

## 2.4 算法（针对 non-quadratic 或大维问题）

### Gradient Descent (Steepest Descent)
```
Δx = −∇f(x_k)
x_{k+1} = x_k + t · Δx       (t = step size, 通常需要 line search)
```
- 只用 ∇f
- 线性收敛 (慢)

### Newton's Method
```
Δx = −H⁻¹(x_k) · ∇f(x_k)
x_{k+1} = x_k + Δx            (t = 1, fixed)
```
- 用 ∇f 和 Hessian
- 二次收敛 (快, 但每次贵)
- ⭐ 二次函数 1 步收敛
- 远离最优时可能 fail (H 不 PD)

### 关键对比

| | Gradient Descent | Newton |
|---|---|---|
| 信息 | ∇f | ∇f + H |
| 步长 | 调 t (line search) | t=1 fixed |
| 速度 | 线性收敛 (慢) | 二次收敛 (快) |
| 成本 | 便宜 | 贵 (求 H⁻¹) |
| 鲁棒 | 远离最优也稳 | 远离最优可能 fail |

---

# 3. Equality-Constrained NLP — Lagrange Multipliers

## 3.1 问题形式

```
min f(x)
s.t. hᵢ(x) = 0,  i = 1, ..., m
```

## 3.2 Lagrangian 函数

```
L(x, λ) = f(x) + Σᵢ λᵢ hᵢ(x)
```
- `λᵢ` 称为 Lagrange multiplier
- **`λᵢ` 任意符号 (free)**（等式约束的特点）

## 3.3 必要条件 (FONC for constrained)

```
∇ₓ L = 0     →     ∇f + Σᵢ λᵢ ∇hᵢ = 0
∇_λ L = 0    →     hᵢ(x) = 0  (即原约束)
```

**几何直觉：** 在最优点, 目标函数的梯度 ∇f 是约束梯度 ∇hᵢ 的线性组合。

## 3.4 解题套路

```
Step 1: 写 Lagrangian
Step 2: ∂L/∂xⱼ = 0 (n 个方程)
Step 3: ∂L/∂λᵢ = 0 (即约束方程, m 个)
Step 4: 解 n+m 方程组 (通常 case split)
Step 5: 比较所有 candidate 的 f(x*), 选最优
```

---

# 4. Inequality-Constrained NLP — KKT Conditions ⭐⭐

## 4.1 一般 NLP 问题形式

```
min f(x)
s.t. gᵢ(x) ≤ 0,  i = 1, ..., m
     hⱼ(x) = 0,  j = 1, ..., p
```

## 4.2 Lagrangian

```
L(x, λ, μ) = f(x) + Σᵢ λᵢ gᵢ(x) + Σⱼ μⱼ hⱼ(x)
```

## 4.3 KKT 4 大条件 (Necessary 在 LICQ 下)

### 条件 1: Primal Feasibility (原可行)
```
gᵢ(x*) ≤ 0, hⱼ(x*) = 0
```

### 条件 2: Dual Feasibility (Sign Restriction)
```
λᵢ 的符号取决于优化方向 + 约束方向:
```

| 问题 | 约束 | `λᵢ` 符号 |
|---|---|---|
| min | `gᵢ ≤ 0` | `λᵢ ≥ 0` |
| min | `gᵢ ≥ 0` | `λᵢ ≤ 0` |
| max | `gᵢ ≥ 0` | `λᵢ ≥ 0` |
| **max** | **`gᵢ ≤ 0`** | **`λᵢ ≤ 0`** ⭐ |

`μⱼ` 永远自由 (equality 约束没有 sign restriction)。

### 条件 3: Complementary Slackness (互补松弛)
```
λᵢ · gᵢ(x*) = 0   ∀i
```
即对每个 i: 要么 `λᵢ = 0` (约束不绑) 要么 `gᵢ(x*) = 0` (约束绑)

### 条件 4: Stationarity (平稳)
```
∇f(x*) + Σᵢ λᵢ ∇gᵢ(x*) + Σⱼ μⱼ ∇hⱼ(x*) = 0
```
即"目标 gradient = 约束 gradients 的线性组合"。

## 4.4 KKT 必要性的前提：Constraint Qualification (CQ)

KKT 仅在"约束行为良好"时是必要条件。最常用的 CQ:
- **LICQ** (Linear Independence Constraint Qualification): active 约束的 gradients 线性无关
- **Slater's Condition** (适用 convex): 存在严格内点 (`gᵢ(x) < 0`)

不满足 CQ → KKT 可能写不出来 (虽然 x* 仍是最优)。

## 4.5 KKT 的充分性 ⭐

**Convex Program 下:** KKT 必要 + 充分。即:
- `f` convex
- `gᵢ` convex
- `hⱼ` affine

→ KKT 解 = global optimum

## 4.6 Case Split 解题套路

```
Step 1: 写 Lagrangian (注意 sign convention!)
Step 2: 写出 4 个 KKT 条件
Step 3: 对每个 i, case split:
   • Case A: λᵢ = 0 (约束 i 不绑)
   • Case B: gᵢ(x*) = 0 (约束 i 绑)
Step 4: 对每种 case 解方程组, 检查:
   - 是否原可行
   - sign restriction 是否满足
Step 5: 矛盾的 case 淘汰
Step 6: 凸性检查写充分性陈述
Step 7: 报告 x*, λᵢ*, f(x*)
```

---

# 5. NLP Duality (简版, 知道即可)

## 5.1 Dual Function
```
g(λ, μ) = inf_x L(x, λ, μ)
```
⭐ `g` 永远 **concave**（不管原问题 convex 与否）。

## 5.2 Dual Problem
```
max g(λ, μ)
s.t. λ ≥ 0     (假设 min primal + ≤ 约束)
```

## 5.3 弱对偶 / 强对偶

**Weak Duality:** `g(λ, μ) ≤ f(x)` 对任意 dual feasible `(λ, μ)` 和 primal feasible `x`。

**Strong Duality:** 在 **convex program + Slater's** 下，`max g = min f` (等号成立)。

---

# 6. Quadratic Programming (QP) — NLP 的重要特例

## 6.1 QP 形式

```
min  (1/2) xᵀ Q x + cᵀ x
s.t. Ax ≤ b,  Ex = d
```
- `Q` 是 symmetric matrix
- 目标二次, 约束线性

## 6.2 QP 凸性判定

```
Q ⪰ 0 (PSD) → 目标 convex → convex QP → KKT 充分 → unique global
Q ≺ 0 (ND)  → 目标 concave → concave QP (max problem 才有意义)
Q indefinite → non-convex QP → 难，可能多 local optima
```

## 6.3 经典 QP 例子: Markowitz Portfolio

```
min  xᵀ Σ x       (variance, Σ = covariance matrix, always PSD)
s.t. μᵀ x ≥ R     (return ≥ target)
     1ᵀ x = budget
     x ≥ 0        (no short)
```

⭐ Covariance matrix **总是 PSD** → 目标 convex → convex QP。

---

# 7. 速答陈述模板

考试时下面这些句子直接搬用，能拿条件分：

### 凸性陈述
> "The Hessian `H(x)` is positive definite for all x because all leading principal minors are positive. Therefore `f` is strictly convex."

### 充分性陈述
> "Since `f` is convex, all `gᵢ` are convex, and the equality constraints are affine, this is a convex program. The KKT conditions are sufficient, and `(x*, λ*, μ*)` is the global optimum."

### Concave maximization (Markowitz, monopolist 等)
> "Since `f` is concave and the constraint set is convex, this is a convex program (in disguise). The KKT/FONC conditions are sufficient for global optimum."

### Newton 优势
> "Newton's method incorporates curvature via the Hessian inverse, achieving quadratic convergence near the optimum. For quadratic objectives, it converges in one iteration."

### Gradient Descent 适用场景
> "Gradient descent only uses first-order information, making it cheap per iteration but slow to converge. It is suitable when the Hessian is unavailable or too expensive to compute."

---

# 30 秒理论速记

**NLP 分类树**: 无约束 (Hessian) vs 有约束 (Lagrange/KKT)

**FONC vs SOSC**: ∇f=0 必要; ∇f=0 + H≻0 充分

**Convex program**: f凸 + gᵢ凸 + hⱼ仿射 → KKT充分 → global

**KKT 4 条件**: Primal feas / Sign / Comp slack / Stationarity

**Sign convention 表 (max + ≤ → λ ≤ 0)** ⭐

**算法**: GD (一阶, 慢但便宜) vs Newton (二阶, 快但贵)

**QP**: Q⪰0 → convex; Markowitz 是 convex QP

---

**这份理论框架 + 之前的题型 cheat sheet, NLP 知识结构完整。**
