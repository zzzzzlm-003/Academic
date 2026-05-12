# NLP3 速查 (5/9 下午复盘)

> NLP3 = "**KKT（不等式约束）+ GD/Newton 迭代算法**"
> 核心：会用 KKT 解一道有 1-2 条不等式约束的 NLP，并背 sign restriction 表

---

## 1. KKT 四件套（必背 / cheat sheet）

问题：$\min f(x)$ s.t. $g_i(x) \leq 0$，$h_j(x) = 0$

Lagrangian：
$$L(x, \lambda, \mu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \mu_j h_j(x)$$

| 条件 | 数学 | 干啥用的 |
|---|---|---|
| **1. Primal feasibility** | $g_i(x^*) \leq 0,\ h_j(x^*) = 0$ | 解必须在围栏里 |
| **2. Sign restriction** | $\lambda_i$ 符号查表 | 保证方向一致 |
| **3. Complementary slackness** | $\lambda_i \cdot g_i(x^*) = 0$ | **KKT 灵魂**：决定每条约束 active 还是 inactive |
| **4. Stationarity** | $\nabla_x L = 0$ | 梯度平衡（n 个方程） |

### Sign Restriction 表 ⭐⭐⭐

| 方向 | 不等式 | λ 符号 |
|---|---|---|
| **min** | **g ≤ 0** | **λ ≥ 0** ⭐ |
| min | g ≥ 0 | λ ≤ 0 |
| max | g ≤ 0 | λ ≤ 0 |
| max | g ≥ 0 | λ ≥ 0 |

记忆口诀："**min ≤ 是正**"，其它三种翻转推导。

⚠️ 等式约束的 μ_j **没有符号限制**（任意实数）。

---

## 2. KKT 解题流程（核心套路）⭐

```
Step 1. 写 Lagrangian L = f + Σ λg + Σ μh
Step 2. 列 4 类方程：
        (A) ∂L/∂xⱼ = 0       — Stationarity（每个变量一个）
        (B) gᵢ(x) ≤ 0,...    — Primal feasibility
        (C) λᵢ · gᵢ(x) = 0   — Complementary slackness（每条不等式一个）
        (D) λᵢ 符号           — Sign restriction
Step 3. 用 (C) 分 case：
        每条不等式 → 假设 active (g=0) 或 inactive (λ=0)
        m 条不等式 → 最多 2^m 个 case
Step 4. 对每个 case 解出 (x*, λ*)
Step 5. 用 (B) 验可行 + 用 (D) 验符号 → 筛掉无效候选
Step 6. 比较剩下候选的 f(x*)，选最优
```

### 经典例题（NLP3 p17-23）模板

> max f(x₁, x₂) s.t. x₁²+x₂² ≤ 5，x₁−x₂ ≤ 1

- 4 种 case：(λ₁=0, λ₂=0), (λ₁=0, g₂=0), (g₁=0, λ₂=0), (g₁=0, g₂=0)
- 通过试错和矛盾排除：**只有"两条都 active" case 可行**
- 解联立方程：x₁²+x₂²=5 ∩ x₁−x₂=1 → 候选 (2,1) 和 (-1,-2)
- 用 sign restriction (max+≤ → λ≤0) 验证：(-1,-2) 推出 λ=2/3 > 0 ❌淘汰；(2,1) 推出 λ₁=-2/3, λ₂=-1/3 ✓
- **最优：(x*, λ*) = (2, 1, -2/3, -1/3)**

---

## 3. 迭代算法 (p27-46)

非凸/non-quadratic 通常没闭式解，**用迭代算法**从起点 x₀ 一步步逼近最优。

### Gradient Descent（最速下降法）
$$x_{k+1} = x_k - t \cdot \nabla f(x_k)$$

- $t$ = 步长（line search 或固定值）
- 只用**一阶信息**（梯度）
- 优点：简单、内存省
- 缺点：**最优附近收敛慢**（zigzag 锯齿）

### Newton's Method
$$x_{k+1} = x_k - [H(x_k)]^{-1} \cdot \nabla f(x_k)$$

- 用**二阶信息**（Hessian）
- 步长**不需要选**（曲率自带）
- 优点：最优附近**二次收敛**（快）
- 缺点：要算 Hessian 并求逆（贵）；H 必须 PD；离起点远时可能不收敛

### 对比

| 项目 | GD | Newton |
|---|---|---|
| 用什么信息 | ∇f | ∇f + H |
| 收敛速度 | 慢（线性） | 快（二次） |
| 每步代价 | 低 | 高（求逆 H） |
| 起点要求 | 宽松 | 必须靠近最优 |
| 对 H 要求 | 不要求 | **必须 PD**（否则下降方向错） |

---

## 4. 考试常见陷阱

1. **Sign restriction 用错**：max+≤ 是 λ ≤ 0，**不是** ≥ 0。slide 例题就是用它淘汰 (-1,-2)。

2. **Complementary slackness 漏 case**：每条不等式都有 2 种可能，记得**穷举**。

3. **等式约束 μ 不要加 sign restriction**：μ 可以是任意实数。

4. **Primal feasibility 验证**：解出来的 x* 必须代回**所有**约束（包括没被设成 active 的那些）检查。

5. **Newton's Method 要求 H 正定**：如果 H 不正定（含负特征值），Newton 方向可能朝上山方向走，不收敛。

---

## 5. 30 秒自测

1. min 问题 + g(x) ≥ 0，λ 应该是？ → **λ ≤ 0**（min/≤ 是正，反方向→翻转）

2. KKT 4 个条件里哪一个是 KKT 比 Lagrange 多出来的？ → **Sign restriction + Complementary slackness**（Lagrange 等式约束没有这俩）

3. Newton's Method 的 update rule 写出来 → $x_{k+1} = x_k - H^{-1} \nabla f$

4. 一道题有 3 条不等式约束，最多多少个 case 要分？ → **2³ = 8 个**（实际很多会被矛盾迅速淘汰）
