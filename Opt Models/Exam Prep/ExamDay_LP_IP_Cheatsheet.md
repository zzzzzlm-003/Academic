# IEOR 4004 LP + IP 考前 Cheat Sheet
*按题型组织 · 每题型配例 + 公式 + 答题套路*

---

# PART 1 — LP (Linear Programming)

## 题型 L1：图解法 (Graphical Method, 2 变量)

**例：** max `z = 3x₁ + 2x₂`  s.t.  `x₁ + x₂ ≤ 4`,  `x₁ + 3x₂ ≤ 6`,  `x₁, x₂ ≥ 0`.

**模板（4 步）：**

```
Step 1: 画约束直线
   每个约束化等式 → 画直线
   ≤ 在直线下方阴影; ≥ 在上方
   x₁, x₂ ≥ 0 → 限制在第一象限

Step 2: 确定可行域
   所有约束阴影的交集 = 可行多边形

Step 3: 画目标函数等值线
   z = c₁x₁ + c₂x₂  → 选一个 z 值画直线
   平行移动等值线

Step 4: 找最优顶点
   max → 沿梯度 (c₁, c₂) 方向移到最远顶点
   min → 沿 (-c₁, -c₂) 方向
   ⭐ 最优解一定在顶点 (corner point) 上
```

**做这道：**
- 可行域顶点：(0,0), (4,0), (0,2), 和 `x₁+x₂=4` 与 `x₁+3x₂=6` 交点 = (3, 1)
- 算 z 值：z(0,0)=0, z(4,0)=12, z(0,2)=4, z(3,1)=**11**
- **答：** max at (4, 0), z* = 12

**特殊情况速记：**
- **唯一最优**：等值线在唯一顶点切到
- **多重最优**：等值线与可行域**一条边**重合
- **无界**：可行域无限延伸 + 目标可无限改进
- **不可行**：约束矛盾, 可行域为空

---

## 题型 L2：化标准形

**LP 标准形要求：** 
1. **max** 目标
2. **等式约束** (Ax = b)
3. **变量 ≥ 0**
4. **b ≥ 0**

**转换速查：**

| 原始 | 改成 |
|---|---|
| `min cᵀx` | `max (−cᵀx)` |
| `aᵀx ≤ b` | `aᵀx + s = b`, `s ≥ 0` (**slack 松弛**) |
| `aᵀx ≥ b` | `aᵀx − s = b`, `s ≥ 0` (**surplus 剩余**) |
| `x` free (无界) | `x = x⁺ − x⁻`, `x⁺, x⁻ ≥ 0` |
| `x ≤ 0` | `x' = −x ≥ 0` |
| `b < 0` | 两边乘 -1, 同时翻转不等号 |

⭐ **Slack vs Surplus 经济含义：**
- Slack > 0 → 资源**有富余**
- Surplus > 0 → 约束**超额满足**

---

## 题型 L3：写对偶问题 (Dual)

**对偶速查表（**必背**）：**

| Primal (max) | Dual (min) |
|---|---|
| `max cᵀx` | `min bᵀy` |
| `Ax ≤ b` | `Aᵀy ≥ c` |

| Primal 约束 → | Dual 变量 |
|---|---|
| `i-th 约束 ≤` | `yᵢ ≥ 0` |
| `i-th 约束 ≥` | `yᵢ ≤ 0` |
| `i-th 约束 =` | `yᵢ` free |

| Primal 变量 → | Dual 约束 |
|---|---|
| `xⱼ ≥ 0` | `j-th 约束 ≥` |
| `xⱼ ≤ 0` | `j-th 约束 ≤` |
| `xⱼ` free | `j-th 约束 =` |

**口诀：** 约束类型 → 变量符号；变量符号 → 约束类型。

**例：** 
```
Primal:
   max  3x₁ + 5x₂
   s.t. x₁ + 2x₂ ≤ 4      (y₁)
        2x₁ + x₂ = 5      (y₂)
        x₁ ≥ 0, x₂ free
```

**Dual:**
- 2 个约束 → 2 个 dual 变量 y₁, y₂
- 约束1 是 ≤ → y₁ ≥ 0
- 约束2 是 = → y₂ free
- x₁ ≥ 0 → dual 约束 ≥
- x₂ free → dual 约束 =

```
Dual:
   min  4y₁ + 5y₂
   s.t. y₁ + 2y₂ ≥ 3      (来自 x₁)
        2y₁ + y₂ = 5      (来自 x₂)
        y₁ ≥ 0, y₂ free
```

---

## 题型 L4：弱/强对偶 + 验证最优性

**弱对偶 (Weak Duality):** 
- 对任意 primal 可行 `x` 和 dual 可行 `y`：**`cᵀx ≤ bᵀy`** (max primal)
- 推论：
  - Primal 无界 → Dual 不可行
  - Dual 无界 → Primal 不可行

**强对偶 (Strong Duality):**
- 若 Primal 有有限最优 → Dual 也有，且 **最优值相等**

**对偶关系表：**

|  | Primal 不可行 | Primal 有界 | Primal 无界 |
|---|---|---|---|
| **Dual 不可行** | 可能 | 不可能 | **一定** |
| **Dual 有界** | 不可能 | **一定** | 不可能 |
| **Dual 无界** | **一定** | 不可能 | 不可能 |

---

## 题型 L5：互补松弛 (Complementary Slackness) ⭐

**核心：** 在最优解, **松弛量 × 对应对偶变量 = 0**

**两组条件：**

| 条件 | 意义 |
|---|---|
| **原约束 i 不紧** (`Aᵢx < bᵢ`, slack > 0) | → **`yᵢ = 0`** |
| **原约束 i 紧** (`Aᵢx = bᵢ`) | → `yᵢ` 可 > 0 |
| **原变量 `xⱼ > 0`** | → **dual 约束 j 取等号** `(Aᵀ)ⱼ y = cⱼ` |
| **原变量 `xⱼ = 0`** | → dual 约束可严格不等 |

**经济解释：** 
- 资源有富余 → 影子价格 = 0（资源不值钱）
- 影子价格 > 0 → 资源用尽（约束 binding）

**验证最优性 4 步：**

```
Step 1: 计算每个约束的 slack: sᵢ = bᵢ − Aᵢx*
        sᵢ > 0 → yᵢ = 0
        sᵢ = 0 → yᵢ 待定

Step 2: 看每个原变量:
        xⱼ* > 0 → (Aᵀ)ⱼ y = cⱼ 必成立
        xⱼ* = 0 → (Aᵀ)ⱼ y ≥ cⱼ 即可

Step 3: 解方程组求 y

Step 4: 验证 y 是 dual 可行 (满足 sign restriction + 所有 dual 约束)
        是 → x* 是最优
        否 → 不是最优
```

---

## 题型 L6：Shadow Price / 灵敏度分析

**Shadow price `yᵢ*` 含义：**
- **`bᵢ` 增加 1 单位 → 最优 objective 改变 `yᵢ*` 个单位**（在范围内）

**常考问法：**
- "如果约束 i 的 RHS 增加 5，objective 变多少？" → `Δz = 5 · yᵢ*`
- "约束 i 不紧（有 slack）时 shadow price 是多少？" → `yᵢ* = 0`（资源富余, 加更多没用）
- "shadow price 何时有效？" → 在 RHS 变化的某个范围内（current basis 保持最优）

---

## 题型 L7：LP Modeling 经典模式

### Production Planning (产能 LP)
```
变量: xⱼ = 产品 j 的产量
目标: max Σ profitⱼ · xⱼ
约束: Σ resourceᵢⱼ · xⱼ ≤ capacityᵢ  (资源)
      xⱼ ≥ 0
```

### Blending Problem
```
变量: xⱼ = 原料 j 的使用量
目标: min Σ costⱼ · xⱼ
约束: 营养i下限/上限
      总量约束
      xⱼ ≥ 0
```

### Transportation Problem
```
变量: x_{ij} = 从源 i 到目的地 j 运量
目标: min Σ c_{ij} x_{ij}
约束: Σⱼ x_{ij} ≤ sᵢ  (源 i 供应)
      Σᵢ x_{ij} ≥ dⱼ  (目的地 j 需求)
      x_{ij} ≥ 0
```

⚠️ 这种网络 LP 的约束矩阵 TU → LP 自动给整数解。

---

# PART 2 — IP (Integer Programming)

## 题型 I1：IP Modeling Tricks ⭐⭐

**触发条件：** 二元 yes/no 决策，固定成本，either-or, if-then, 计数限制

### Trick 1: Binary 决策变量
```
yⱼ = 1 if 选择/激活/做事件 j, 0 otherwise
```

### Trick 2: Fixed Charge (固定成本 / Big-M)
"生产 → 付固定成本 + 可变成本"

```
变量: xⱼ ≥ 0 (产量), yⱼ ∈ {0,1} (是否启动)
约束: xⱼ ≤ M · yⱼ   ← Big-M 链接约束
目标: min Σ (cⱼ · xⱼ + fⱼ · yⱼ)
       变量成本 + 固定成本
```

**Big-M 选择：** `M` 要够大让 `yⱼ=1` 时 `xⱼ` 不受限，但又不能太大（数值问题）。建议 `M = 实际上限`。

### Trick 3: 二选一约束 (Either-Or / Disjunction)
"约束 1 满足 **或** 约束 2 满足"

```
约束 1:  a₁ᵀ x ≤ b₁ + M(1 − y)
约束 2:  a₂ᵀ x ≤ b₂ + M · y
y ∈ {0,1}

y=1 → 约束 1 active, 约束 2 relaxed
y=0 → 约束 2 active, 约束 1 relaxed
```

### Trick 4: If-Then (蕴含)
"如果 yA=1, 那 yB 必须 = 1"

```
yA ≤ yB    或等价  yA − yB ≤ 0
```

### Trick 5: At Most / At Least k
```
"至多 k 个 yⱼ = 1":  Σ yⱼ ≤ k
"至少 k 个":  Σ yⱼ ≥ k
"恰好 k 个":  Σ yⱼ = k
```

### Trick 6: SOS1 (Special Ordered Set 1)
"最多一个 xⱼ 可以非零"

```
xⱼ ≤ M yⱼ,  Σ yⱼ ≤ 1
```

### Trick 7: Piecewise Linear Cost
分段线性成本 → 用 binary 切换不同段。

---

## 题型 I2：Branch and Bound 算法 ⭐

**模板（4 步）：**

```
Step 1: LP Relaxation
   去掉整数约束, 解 LP, 得最优值 z_LP
   • max 问题: z_LP 是 IP 最优的 上界 (upper bound)
   • min 问题: z_LP 是 IP 最优的 下界 (lower bound)

Step 2: 检查是否整数
   全部整数 → 找到 IP 最优, 完成
   有分数 → 选一个分数变量分支

Step 3: Branch (分支)
   选分数变量 xⱼ*, 假设 xⱼ* = 5.6
   分两个子问题:
     左子: 加 xⱼ ≤ 5
     右子: 加 xⱼ ≥ 6
   (这两个分支无重叠无遗漏)

Step 4: 对每个节点剪枝 (Prune)
   剪掉的条件:
   • LP 不可行                     → infeasibility
   • LP 最优 ≤ 当前 incumbent 值   → bound
     (max 问题; min 反过来 ≥)
   • LP 最优是整数                 → 整数解, 更新 incumbent

Step 5: 重复直到所有 live nodes 处理完
```

**例：** max `z = 3x₁ + 2x₂`  s.t.  `x₁ + x₂ ≤ 4`, `x₁ + 3x₂ ≤ 6`, `x₁, x₂ ≥ 0` 整数.

```
Node 0 (root): LP 解 = (3, 1), z=11 (整数!) → 这道题 LP 直接给整数
   → 最优 (3, 1), z*=11

   (若 LP 给 (3.5, 0.5), z=12.5, 则:
    分支 x₁ ≤ 3 和 x₁ ≥ 4, 各自解 LP, 继续...)
```

**关键术语：**
- **Incumbent**: 目前已知的最好整数可行解
- **Pruning by bound**: max 问题, 若节点 LP 上界 ≤ incumbent → 剪枝
- **Pruning by infeasibility**: 节点 LP 无解 → 剪枝
- **Pruning by integrality**: 节点 LP 最优是整数 → 更新 incumbent, 该节点完成

---

## 题型 I3：Cutting Plane (Chvátal-Gomory Cut)

**思想：** 给 LP 加一条不等式（"cut"）使得：
- 削除当前的分数 LP 最优
- 不消除任何整数可行解

**CG Cut 构造步骤：**

```
Step 1: 选一个非负向量 u (multiplier)
Step 2: 把约束乘 u 累加: uᵀA x ≤ uᵀb
Step 3: 因 x 整数, 向下取整: ⌊uᵀA⌋ x ≤ ⌊uᵀb⌋
        ← 这就是 CG cut
Step 4: 加入 LP, 重新求解
```

**例：** 
```
max 3x₁ + 2x₂
s.t. 10x₁ + 6x₂ ≤ 45
     x₁, x₂ ≥ 0 整数

LP 最优: (4.5, 0), z=13.5 (分数 ✗)

取 u = 0.5:
  0.5 · (10x₁ + 6x₂) ≤ 0.5 · 45
  5x₁ + 3x₂ ≤ 22.5

向下取整:
  5x₁ + 3x₂ ≤ 22     ← CG Cut

验证:
  • LP 解 (4.5, 0): 5(4.5)+0 = 22.5 > 22 → 被削除 ✓
  • 整数 (4, 0):    5(4)+0 = 20 ≤ 22 → 保留 ✓
```

**Branch & Cut：** 现代 solver 把 B&B + Cutting Plane 结合用。

---

## 题型 I4：LP Relaxation 性质（概念题）

**Q: LP relaxation 给出什么界？**
- Max IP: LP relaxation 是**上界** (`z_LP ≥ z_IP`)
- Min IP: LP relaxation 是**下界** (`z_LP ≤ z_IP`)

**Q: 什么时候 LP relaxation 直接给 IP 最优？**
- 约束矩阵 totally unimodular (TU) + RHS 整数 → LP 自动整数
- 经典 TU 问题：assignment, transportation, shortest path, max flow, MST 的 spanning tree formulations

**Q: 强 formulation 是什么意思？**
- 同一 IP 可以有多个等价 formulation
- LP relaxation 越接近 IP 解 → formulation 越**强**（"tight"）
- 强 formulation → B&B 收敛快

---

## 题型 I5：TSP 完整 IP (Subtour Elimination)

**两种 formulation：**

### A. Subset-Based (子集约束, 指数增长但 LP 界紧)

```
min Σ d_{ij} x_{ij}
s.t. Σⱼ x_{ij} = 1  ∀i           (每城市出一次)
     Σᵢ x_{ij} = 1  ∀j           (每城市进一次)
     Σ_{i,j ∈ S} x_{ij} ≤ |S| − 1
         ∀ S ⊂ V, 3 ≤ |S| < n    (subtour elimination)
     x_{ij} ∈ {0,1}
```

⚠️ Subset 约束**指数级**：n=20 已有 100万+ 子集 → 实际不可行

### B. MTZ Formulation (多项式约束)

```
变量增加: uᵢ ∈ ℝ, i = 2,...,n

约束改为:
     Σⱼ x_{ij} = 1, Σᵢ x_{ij} = 1
     uᵢ − uⱼ + n · x_{ij} ≤ n − 1
        ∀ i,j ∈ {2,...,n}, i ≠ j
     1 ≤ uᵢ ≤ n − 1
```

**优点：** 约束数 `O(n²)` 而非指数
**缺点：** LP relaxation 较弱

---

# PART 3 — 通用 SOP 答题流程

```
任何 LP/IP 题:
Step 0: 读题, 识别 min/max, 决策变量, 资源约束
Step 1: 写决策变量定义 (明确单位)
Step 2: 写目标函数
Step 3: 写约束 (资源限制, 需求, 守恒, 非负)
Step 4: 加 integrality / binary (若是 IP)

如果让求解:
   LP 小问题 → 图解法 / 顶点枚举
   LP 验最优 → 互补松弛
   IP 小问题 → B&B 或 cutting plane

如果让分析:
   写 dual → 用对偶表
   shadow price → ∂z*/∂bᵢ
   sensitivity → ranging
```

---

# 30 秒考前自检

✅ LP 标准形：max, =, x≥0, b≥0
✅ Dual 转换表：约束→变量符号, 变量→约束类型
✅ Strong duality: 最优值相等
✅ Complementary slackness: slack × dual = 0
✅ Shadow price = `yᵢ*` = `∂z*/∂bᵢ`
✅ B&B 3 种剪枝: infeasible / bound / integer-found
✅ CG cut: `⌊uᵀA⌋ x ≤ ⌊uᵀb⌋`
✅ Big-M: `xⱼ ≤ M yⱼ` 链接连续与二元变量
✅ Either-Or: `aᵀx ≤ b + M(1−y)`
✅ TSP 必须有 subtour elimination

---

# 题型快速识别 (一句话锁定答题路径)

| 题目说... | 用什么 |
|---|---|
| "用图解法" | 画约束 + 顶点枚举 |
| "找最优 + 2 变量" | 图解法 |
| "写对偶问题" | 对偶转换表 |
| "验证 x* 是最优" | 互补松弛 |
| "增加 1 单位 bᵢ" | shadow price |
| "yes/no decision" | binary variable |
| "fixed cost + variable cost" | Big-M + binary |
| "either A or B" | 两个 big-M 约束 + binary |
| "解 IP" | B&B 或 cutting plane |
| "为什么 LP 不够" | 整数性 |
| "TSP 公式" | MTZ 或 subtour elim |

---

**抄完 = LP+IP 全题型覆盖。加上之前的 NLP + Network 全套, 考试两道题 100% 命中。**
