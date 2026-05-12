# IEOR 4004 考前最终版 Cheat Sheet
*按题型组织 · 每题型配例 + 公式 + 答题套路*

---

# PART 1 — NLP (Q1, heavy emphasis)

## 题型 N1：判断函数 convexity

**例：** `f(x₁, x₂, x₃) = x₁² + x₂² + 3x₃² − x₁x₂ − x₂x₃ − x₁x₃`. Is `f` convex?

**模板（3 步）：**

```
Step 1: Compute Hessian H
   ∂²f/∂xᵢ² 放对角线
   ∂²f/∂xᵢ∂xⱼ 放 (i,j) 和 (j,i)

Step 2: 检验 H 是 PD/PSD/ND/NSD/Indefinite
   首选 Sylvester判定法（手算快）:
   • PD: 所有 leading principal minors > 0
   • ND: leading minors 符号交替 −, +, −, +...

Step 3: 下结论
   PD → strictly convex; PSD → convex
   ND → strictly concave; NSD → concave
   Indefinite → neither
```

**做这道：**
- `H = [[2,−1,−1],[−1,2,−1],[−1,−1,6]]`
- 1×1: 2 > 0 ✓ ; 2×2 det: `2·2 − (−1)(−1) = 3 > 0` ✓ ; 3×3 det: 12 > 0 ✓
- **All positive → PD → f is strictly convex.**

**Sylvester 速记表：**

| Leading 主子式符号 | 结论 |
|---|---|
| `+, +, +, ...` | PD → strictly convex |
| `−, +, −, +, ...` | ND → strictly concave |
| 不满足任一 | 看具体情况 |

⭐ **convex function 常见例：** `x²`, `eˣ`, `−log x`, `‖x‖`, `xᵀQx (Q ⪰ 0)`
⭐ **保持 convexity 的运算：** 非负加权和 / 仿射复合 `f(Ax+b)` / pointwise max

---

## 题型 N2：Unconstrained Optimization

**例：** Find **max** of `f(q₁, q₂) = −4q₁² + 55q₁ − 15q₂² + 135q₂ − 100`.
*(注：f 是 concave 的，所以是 max 问题；concave 函数没有有限 min。)*

**模板：**

```
Step 1: ∇f = 0 (FONC)
Step 2: 求解 → candidate x*
Step 3: 算 H, 验证 PD (for min) 或 ND (for max)
   • H PD → x* is strict local min
   • H ND → x* is strict local max
Step 4: 如果 f convex (PD on whole domain) → global ⭐
```

**做这道：**
- `∂f/∂q₁ = −8q₁ + 55 = 0 → q₁* = 55/8 = 6.875`
- `∂f/∂q₂ = −30q₂ + 135 = 0 → q₂* = 4.5`
- `H = [[−8,0],[0,−30]]` → 特征值 −8, −30 → ND → f strictly concave
- (concave + maximization 这里其实是 max 问题) → **global max** at `(6.875, 4.5)`

**速记：**
| 优化方向 | 想要 Hessian 是 |
|---|---|
| min | PD / PSD (convex) |
| max | ND / NSD (concave) |

---

## 题型 N3：Lagrangian with Equality Constraint

**例：** min `x₁² + x₂²`  s.t.  `(x₁ − 2)² + x₂² = 1`.

**模板：**

```
Step 1: 写 Lagrangian
   L(x, λ) = f(x) + λ · h(x)
   (等式约束: λ 任意符号, 自由)

Step 2: 设三组偏导 = 0
   ∂L/∂xᵢ = 0  ∀i
   ∂L/∂λ = h(x) = 0  (即原约束)

Step 3: 解方程组 (常需 trial & error)
   - 从最简单的方程入手
   - 因式分解, case split

Step 4: 比较所有 candidate 的 f 值, 选最优
```

**做这道：**

```
L = x₁² + x₂² + λ[(x₁−2)² + x₂² − 1]

∂L/∂x₁ = 2x₁ + 2λ(x₁−2) = 0  → x₁(1+λ) = 2λ  ...(A)
∂L/∂x₂ = 2x₂ + 2λx₂ = 0       → x₂(1+λ) = 0  ...(B)
∂L/∂λ = (x₁−2)² + x₂² − 1 = 0                  ...(C)

从(B): x₂ = 0  或  λ = −1
  • 若 λ = −1: (A) 变 0 = −2 矛盾 → 弃
  • 故 x₂ = 0

代 x₂ = 0 入 (C): (x₁−2)² = 1 → x₁ = 1 或 3
  • x₁=1: f = 1
  • x₁=3: f = 9

→ Min at (x₁*, x₂*) = (1, 0), f* = 1
```

---

## 题型 N4：KKT with Inequality Constraints ⭐⭐ **(最可能考的大题)**

**例：** max `3x₁ + x₂`  s.t.  `x₁² + x₂² ≤ 5`, `x₁ − x₂ ≤ 1`.

**Sign Convention 表（必看）：**

| 问题 | 约束形式 | `λᵢ` 符号 |
|---|---|---|
| min | `gᵢ ≤ 0` | `λᵢ ≥ 0` |
| min | `gᵢ ≥ 0` | `λᵢ ≤ 0` |
| max | `gᵢ ≥ 0` | `λᵢ ≥ 0` |
| **max** | **`gᵢ ≤ 0`** | **`λᵢ ≤ 0`** ⭐ |

**模板（4 个 KKT 条件 + case split）：**

```
Step 1: 写 Lagrangian
   L = f + Σ λᵢ gᵢ + Σ μⱼ hⱼ

Step 2: 写出 KKT 4 条件
   (1) Stationarity: ∇ₓ L = 0
   (2) Primal feasibility: gᵢ ≤ 0, hⱼ = 0
   (3) Sign restriction on λᵢ (查上表!)
   (4) Complementary slackness: λᵢ · gᵢ = 0

Step 3: Case split on Comp. Slackness
   对每个 i: 要么 λᵢ = 0 要么 gᵢ = 0
   - 通常先试 "λᵢ = 0"（约束不绑），不行再试"约束绑"
   - 验证符号 + feasibility, 矛盾就淘汰

Step 4: 凸性检查 (sufficiency)
   • 凸目标 + 凸inequality + 仿射equality → KKT 充分, x* = global opt
```

**做这道（NLP3 讲义例题，完整 walkthrough）：**

```
L = 3x₁ + x₂ + λ₁(x₁² + x₂² − 5) + λ₂(x₁ − x₂ − 1)

KKT:
(C) ∂L/∂x₁ = 3 + 2λ₁ x₁ + λ₂ = 0
(D) ∂L/∂x₂ = 1 + 2λ₁ x₂ − λ₂ = 0
(A) λ₁(x₁² + x₂² − 5) = 0
(B) λ₂(x₁ − x₂ − 1) = 0
   primal feas: 两个约束 ≤ 0
   sign:  λ₁, λ₂ ≤ 0  (max + ≤)

Case split on (A):
   λ₁ = 0?  → (C) 给 λ₂ = −3, (D) 给 λ₂ = 1, 矛盾 ✗
   → λ₁ ≠ 0, 故 x₁² + x₂² = 5  ...(E)

Case split on (B):
   λ₂ = 0?  → (C)(D) 解出 x₁ = −3/(2λ₁), x₂ = −1/(2λ₁)
      代(E): λ₁² = 1/2, sign 要 λ₁ ≤ 0 → λ₁ = −√2/2
      算 x₁ = 3√2/2 ≈ 2.12, x₂ = √2/2 ≈ 0.71
      验证 x₁ − x₂ = √2 ≈ 1.41 > 1 → 违反约束 ✗
   → λ₂ ≠ 0, 故 x₁ − x₂ = 1  ...(F)

联立 (E)+(F): x₁² + (x₁−1)² = 5
   2x₁² − 2x₁ − 4 = 0 → x₁² − x₁ − 2 = 0 → x₁ = 2 或 −1
   候选 (2, 1) 或 (−1, −2)

验证:
   (−1, −2): 代(C) → λ₁ = 2/3 > 0 违反 sign ✗
   (2, 1):   代(C)(D) → λ₁ = −2/3, λ₂ = −1/3, 都 ≤ 0 ✓

⭐ 答: (x₁*, x₂*) = (2, 1), λ₁* = −2/3, λ₂* = −1/3
   Max value = 3(2) + 1 = 7
```

**充分性陈述（**必写！考试加分点**）：**

> Since `f = 3x₁ + x₂` is linear (both convex and concave), `g₁ = x₁² + x₂² − 5` is convex, and `g₂` is affine, this is a convex maximization problem... wait — 这里是 max linear 在 convex feasible region，目标既凸又凹， feasible region 凸 → KKT 充分 → `(2, 1)` 是 global maximum.

---

## 题型 N5：Convex Programming Verification

**形式：** "Is this a convex program? Justify."

**模板（标准答题 4 句话）：**

```
For minimization:
1. Objective f is convex because [Hessian PD/PSD argument 或 凸函数运算保持]
2. Each inequality gᵢ ≤ 0 has gᵢ convex (or gᵢ ≥ 0 has gᵢ concave)
3. Each equality hⱼ is affine
→ This is a convex program. KKT conditions are sufficient.

For maximization:
1. Objective f is concave
2. Inequality side same (g still convex if gᵢ ≤ 0)
3. Equalities affine
→ Convex program.
```

---

## 题型 N6：Absolute Value Linearization

**触发条件：** 看到 `|·|` 在 min objective 或 `≤` constraint。

**例：** min `Σᵢ |aᵢ x − bᵢ|` (L1 regression)

**模板：**

```
对每个 |xᵢ − targetᵢ|:
  引入 zᵢ ≥ 0
  把 |xᵢ − targetᵢ| 换成 zᵢ
  添加约束:
    zᵢ ≥  (xᵢ − targetᵢ)
    zᵢ ≥ −(xᵢ − targetᵢ)

→ NLP 变 LP, 整个问题 convex
```

**例：HVAC**
原: `min Σᵢ rate · |Tᵢⁱⁿ − Tᵢᵒᵘᵗ|`
变: `min Σᵢ rate · zᵢ`, `zᵢ ≥ ±(Tᵢⁱⁿ − Tᵢᵒᵘᵗ)`

**⚠️ 适用范围：** 只在 "min 一个 |·|" 或 "约束 |·| ≤ b" 时能用。`max |·|` 或 `|·| ≥ b` 不行（非 convex）。

---

## 题型 N7：Algorithm — One Iteration

### Gradient Descent (Steepest)

**模板：**
```
1. 算 ∇f(x_k)
2. Δx = −∇f(x_k)
3. x_{k+1} = x_k + t · Δx   (t 题目给)
```

**例：** `f(x₁, x₂) = (x₁−3)² + (x₂−2)²`, `x₀ = (1, 1)`, `t = 0.01`
- `∇f(1,1) = (2(1−3), 2(1−2)) = (−4, −2)`
- `x₁ = (1, 1) − 0.01 · (−4, −2) = (1.04, 1.02)`

### Newton's Method

**模板：**
```
1. 算 ∇f(x_k) 和 Hessian H(x_k)
2. 求 H⁻¹
3. Δx = −H⁻¹ · ∇f
4. x_{k+1} = x_k + t · Δx   (Newton 通常 t = 1)
```

**例：** `f(x₁, x₂) = x₁² + x₁x₂ + 3x₂²`, `x₀ = (3, 3)`
- `∇f = (2x₁ + x₂, x₁ + 6x₂)` → `∇f(3,3) = (9, 21)`
- `H = [[2, 1], [1, 6]]`, `det = 11`, `H⁻¹ = (1/11) · [[6,−1],[−1,2]]`
- `Δx = −(1/11) · (33, 33) = (−3, −3)`
- `x₁ = (3, 3) + (−3, −3) = (0, 0)` ← **一步就到 global min**

**⭐ Newton 在 quadratic function 上 1 步收敛**（最大卖点）。

### Initialization Methods (概念题)
- **Random**: 没先验 / 非凸多重启
- **Grid Search**: 小维度
- **Warm Start**: 参数调优 / 序列问题
- **Clustering**: 多模态
- **Heuristic / Informed**: 有 domain knowledge

---

## 题型 N8：Formulation (Markowitz / 类QPP)

**例：** $1000 投 3 只股票, `E(Sᵢ)`、`Var(Sᵢ)`、`Cov(Sᵢ,Sⱼ)` 给出. 最小化 portfolio variance, 期望收益 ≥ 12%.

**模板：**

```
变量: xⱼ = 投资到股票 j 的钱

目标 (variance):
   Var(Σ xⱼSⱼ) = Σ xⱼ² Var(Sⱼ) + 2 Σ_{i<j} xᵢxⱼ Cov(Sᵢ,Sⱼ)
   = xᵀ Σ x  (Σ 是 covariance matrix, PSD)

约束:
   Σ xⱼ · E(Sⱼ) ≥ target_return × budget   (期望收益)
   Σ xⱼ = budget                            (预算)
   xⱼ ≥ 0                                   (no short selling)
```

**⭐ 关键性质：** Covariance matrix `Σ` **always PSD** → 目标 convex → convex QPP → KKT 充分 → unique global min.

---

# PART 2 — Network (Q2 候选)

## 题型 W1: Shortest Path 

**LP Formulation 模板（最常考）：**

```
变量: x_{ij} ≥ 0  (每条边)

min Σ c_{ij} x_{ij}

Flow conservation:
  起点 s: Σⱼ x_{sj} − Σⱼ x_{js} = 1
  终点 t: Σⱼ x_{tj} − Σⱼ x_{jt} = −1
  中间 i: Σⱼ x_{ij} − Σⱼ x_{ji} = 0
```

**⭐ Property:** LP 约束矩阵 totally unimodular → LP 自动给整数解.

**Dijkstra 模板（边权 ≥ 0）：**

```
1. d[s]=0, d[u]=∞ ∀u≠s, all unmark
2. Repeat:
   - pick unmark u with smallest d[u]
   - for each (u,v): d[v] = min(d[v], d[u]+c(u,v))
   - mark u
3. Stop when t marked
```

⚠️ Dijkstra **不能处理负权**（贪心假设破坏）。

---

## 题型 W2: TSP + MTZ

**模板：**

```
变量:
   x_{ij} ∈ {0,1}  (是否走 i→j)
   uᵢ ∈ ℝ, i = 2,…,n  (访问次序)

min Σ c_{ij} x_{ij}

s.t.
   Σⱼ x_{ij} = 1  ∀i         (每城市出一次)
   Σᵢ x_{ij} = 1  ∀j         (每城市进一次)
   uᵢ − uⱼ + n · x_{ij} ≤ n−1
      ∀ i,j ∈ {2,…,n}, i≠j   (MTZ消子环)
   1 ≤ uᵢ ≤ n−1  ∀i
```

**MTZ 直觉：** 走 `i→j` 强制 `uⱼ ≥ uᵢ + 1` → 访问次序递增 → 不可能成子环.

---

## 题型 W3: Project Management / Critical Path

**模板（**这是讲义的独家技巧**）：**

```
1. 节点 = checkpoints (任务汇合时刻)
2. 边 = 任务, 边权 = − duration
3. 找 start→finish 的最长 path
   = 取负后的最短 path
   = critical path
4. 总工期 = path 长度的绝对值
```

**例（NM2）：** A=6, B=9, C=8 (or 7), D=7 (or 8), E=10, F=12
- Critical path: **B → D → E → F**
- 工期: **9 + 7 + 10 + 12 = 38 天**

**追问题型：**
- Critical 任务延迟 k 天 → 总工期 +k
- 非 critical 在 slack 内 → 不影响

---

## 题型 W4: MST + Prim/Kruskal

**IP Formulation：**

```
变量: x_{ij} ∈ {0,1}  (边选不选)

min Σ c_{ij} x_{ij}
s.t.
   Σ x_{ij} = n−1                         (n−1 条边)
   Σ_{i,j∈S} x_{ij} ≤ |S|−1  ∀S ⊂ V       (subtour elim)
   x_{ij} ∈ {0,1}

LP relaxation: x_{ij} ≥ 0
```

**Kruskal 模板（手算优先用这个）：**

```
1. 边按权重排序
2. 从小到大, 加边 if 不成环 (用 Union-Find 查), else 跳过
3. 加满 n−1 条边停
```

**Prim 模板：**

```
1. 选任意起点 v₀, T = {v₀}
2. Fringe = 与 T 相邻但不在 T 的边
3. 选 fringe 中最小边, 加入 T
4. 重复 n−1 次
```

**⭐ 概念辨析（极易考）：**

| | MST | Shortest Path |
|---|---|---|
| 目标 | 全网骨架最便宜 | 两点间最便宜 |
| 算法 | Prim / Kruskal | Dijkstra |
| 边数 | 恰好 n−1 | 不定 |
| 图类型 | 通常无向 | 通常有向 |

**Prim vs Kruskal：**
- Prim node-based (像 Dijkstra), 稠密图快
- Kruskal edge-based (sort + Union-Find), 稀疏图快

---

# PART 3 — 答题流程通用 SOP

任何 NLP 题都按这套流程答：

```
Step 0: 读题, 确认 min / max
Step 1: Formulate (变量, 目标, 约束) — 注意标准形式 g ≤ 0
Step 2: 凸性分析 — 算 Hessian + 用 Sylvester
Step 3: 写充分性句子 — "Convex program, KKT sufficient"
Step 4: 应用相应方法
   • 无约束 → FONC
   • 等式约束 → Lagrangian
   • 不等式约束 → KKT + case split
Step 5: 验证 — sign restriction + feasibility + 比较所有候选
Step 6: 写答案 — x*, f(x*), 必要时 λ*
```

---

# 30 秒最终自检（考前 5 分钟）

✅ Sign convention 表能默写 (4 行)
✅ KKT 4 条件能默写 (Stationarity / Primal feas / Dual feas / Comp slack)
✅ Convex sufficiency 那句话: "f convex + gᵢ convex + hⱼ affine → KKT sufficient → global"
✅ 看到 `|·|` 在 min 目标 → 引入 z, 拆两个 linear constraint
✅ Hessian PD → 严格 convex → global min
✅ MTZ: `uᵢ − uⱼ + n·x_{ij} ≤ n−1`
✅ Shortest path LP: flow conservation
✅ MST: n−1 条边, subtour elimination

---

**抄完这个 = 全部题型都备完。** 
预计抄完用时 45-60 分钟，刚好 10:45 之前完成 → 吃饭 + 休息 → 出门。
