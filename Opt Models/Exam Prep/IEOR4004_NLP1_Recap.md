# NLP1 速查 (5/9 上午复盘)

> NLP1 = "**什么是 NLP + 凸性 + 二次规划**"
> 核心：会判一个无约束 NLP 的最优点是 min/max/saddle，会判函数是否凸

---

## 0. NLP 标准型 (p5)

$$\min\ f(x) \quad \text{s.t.}\ g_i(x) \leq b_i,\ h_j(x) = b'_j$$

至少有一个 f, g, h 非线性 → NLP。三件武器对三种形态：

| NLP 形态 | 武器 |
|---|---|
| 无约束 | **Hessian** 判 min/max/saddle |
| 只有等式约束 | **Lagrangian** |
| 有不等式约束（最一般） | **KKT**（Lagrangian 的升级版） |

---

## 1. 梯度 ∇ 和 Hessian (p40-46)

**梯度**：偏导组成的向量。指向上升最快的方向。
$$\nabla f = \left(\tfrac{\partial f}{\partial x_1}, \ldots, \tfrac{\partial f}{\partial x_n}\right)$$

**Hessian**：二阶偏导组成的对称方阵。
$$H = \nabla^2 f, \quad H_{ij} = \tfrac{\partial^2 f}{\partial x_i \partial x_j}$$

**Hessian 永远对称**（混合偏导可交换） → 特征值都是实数

---

## 2. 凸性 (p26-39) ⭐ 核心

**关键直觉**：凸性 = 不存在"出去又进来" = **局部最优 = 全局最优** = 简单

**核心式子**（凸组合 convex combination）：
$$\lambda x + (1-\lambda)y, \quad \lambda \in [0,1]$$

| 对象 | 定义 |
|---|---|
| 凸集 | $x, y \in S \Rightarrow \lambda x + (1-\lambda) y \in S$ |
| 凸函数 | $f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)$ ⬅ **琴生不等式** |
| 凹函数 | 把 ≤ 换成 ≥ |

**凸函数 = ⌣ 形（朝上开口的碗）**，min 问题用凸；max 问题用凹。

---

## 3. 判凸（核心考点）⭐⭐⭐ (p46-50)

**Quadratic 函数** $f(x) = x^T A x + b^T x + c$（slide 约定不带 ½）：
→ Hessian H = 2A 是常数，判一次就够了

**Non-quadratic 函数**：H(x) 随 x 变 → 必须保证 H(x) 在**整个域上**每点都 PSD

### 提取 A 的窍门
- $x_1^2$ 的系数 → A₁₁
- $x_1 x_2$ 的系数 → 拆成 A₁₂ + A₂₁ = 2·A₁₂（对称）
- 线性项 + 常数项 **不影响凸性**（不进 Hessian）

### 三种判正定方法

| 方法 | 怎么算 | 速度 |
|---|---|---|
| **顺序主子式** ⭐ 推荐 | 从左上角取 1×1, 2×2, ..., n×n 子矩阵算 det，全 > 0 → PD | 快 |
| **特征值法** | 解 det(A − λI) = 0，全部 λ > 0 → PD | 慢但稳 |
| **Solver** | Gurobi 直接告诉你 | 仅 coding 题用 |

### 完整对照表

| 矩阵性质 | 函数性质 | critical 点 |
|---|---|---|
| 全 λ > 0（PD） | 严格凸 | **局部 min = 全局 min** |
| 全 λ ≥ 0（PSD） | 凸 | min（含半凸边界） |
| 全 λ < 0（ND） | 严格凹 | **局部 max = 全局 max** |
| 全 λ ≤ 0（NSD） | 凹 | max |
| 有正有负（不定） | 既不凸也不凹 | **saddle point** |

⚠️ 顺序主子式法只对 PD/ND 严格判定可靠，PSD 边界情况要用特征值法。

---

## 4. 必背：2×2 与 3×3 行列式

**2×2**：$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$

**3×3 沿第一行展开**：
$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = a(ei-fh) - b(di-fg) + c(dh-eg)$$

⚠️ **顺序主子式 ≠ 余子式**
- 顺序主子式：从左上角往右下扩张，取 k×k 子矩阵的 det → **判正定用**
- 余子式：划掉第 i 行第 j 列剩下的 det → **算行列式用**

---

## 5. QP（二次规划）的概念 (p40-58)

**QP = quadratic objective**（可能 + 线性约束）：
$$\min\ x^T A x + b^T x \quad \text{s.t. (linear constraints)}$$

- 若 A 正定（min 问题）→ 凸 QP → Gurobi 容易解
- 凸 QP 经典应用：**Markowitz 投资组合**（最小化方差）
- "Quadratic + 等式约束" → KKT 退化为线性方程组（Lagrangian 闭式可解）

考试不考写 Gurobi 代码，只需会建模 + 判凸。

---

## 30 秒自测

1. f(x,y) = 4x² + 5y² + 2xy + 3x + 1 凸吗？
   → A=[[4,1],[1,5]]，D₁=4>0, D₂=19>0 → **凸 ✓**

2. f(x) = x³ 凸吗？
   → H(x)=6x，x>0 时正、x<0 时负 → **既不凸也不凹**

3. 凸集和凸函数的定义有什么共同点？
   → 都用了凸组合 λx + (1-λ)y。凸集要求该组合还在集合里；凸函数要求 f 在该点的值 ≤ secant line（琴生）

---

## 下一站

NLP2 = **无约束优化的迭代算法**（GD + Newton's method） — 用今天学的 ∇ 和 H 真正动手解题
