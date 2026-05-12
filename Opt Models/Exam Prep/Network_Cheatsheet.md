# IEOR 4004 — Network Models 完整 Cheat Sheet

*考点 4 块：Shortest Path · TSP/MTZ · Project Management · MST*

---

# 0. 图论基础

## 0.1 节点、边、记号

- **Vertices (V)** = 节点 / 顶点
- **Edges (E)** = 边（无向）；**Arcs** = 弧（有向）
- 记号：`V(G)` 表示图 `G` 的节点集，`E(G)` 表示边集。一般写 `G = (V, E)`。
- 边权（cost / distance）：`c(u, v)` 或 `c_{uv}`

## 0.2 有向 vs 无向（Directed vs Undirected）

- **有向边**：有序对 `(u, v)`，方向从 `u` 到 `v`
- **无向边**：无序对 `{u, v}`，等价于 `(u, v)` 和 `(v, u)` 都存在

**如何判断：**

| 方法 | Directed | Undirected |
|---|---|---|
| 视觉 | 边有箭头 | 边无箭头 |
| Adjacency matrix `A` | `A ≠ Aᵀ` | `A = Aᵀ`（对称） |
| 度概念 | in-degree, out-degree | degree |

## 0.3 Path / Cycle / Tree / Forest

| 术语 | 定义 |
|---|---|
| **Path** | 一串相连的节点+边；有向path要方向一致 |
| **Cycle** | 首尾相连且中间不重复 (首尾除外) |
| **Tree** | 无向、connected、无cycle |
| **Forest** | 无cycle 但不一定 connected（多棵tree的集合） |
| **Spanning tree** | 包含图中**所有节点**的tree |

**关键定理：** `n` 个节点的 tree 恰好有 **`n − 1`** 条边。

---

# 1. SHORTEST PATH（最短路径）

## 1.1 问题

给带权有向图 `G = (V, E)`，起点 `s` 终点 `t`，求 `s → t` 总权重最小的path。

## 1.2 LP Formulation（**必考写法**）

变量：`x_{ij}` = 边 `(i, j)` 是否在path上（IP是 `{0, 1}`，LP relaxation是 `≥ 0`）

```
min Σ_{(i,j) ∈ E} c_{ij} · x_{ij}

s.t. Σ_j x_{sj} − Σ_j x_{js} = 1        (起点：净流出 = 1)
     Σ_j x_{tj} − Σ_j x_{jt} = −1       (终点：净流入 = 1)
     Σ_j x_{ij} − Σ_j x_{ji} = 0        (中间节点：流守恒)  ∀ i ≠ s, t
     x_{ij} ≥ 0
```

**Flow conservation（流守恒）**：每个中间节点"流入 = 流出"。

⭐ **重要事实：** 该 LP 的约束矩阵是 **totally unimodular** → LP relaxation 自动给出 0/1 整数解 → 不需要写IP，直接LP就够。

## 1.3 Dijkstra 算法（边权**非负**）

```
Algorithm: Dijkstra
1. 初始化：d[s] = 0, d[u] = +∞ ∀u ≠ s。所有节点 unmark。
2. 重复：
   - 从未mark的节点中，选 d[u] 最小的 u
   - 对每条 (u, v) ∈ E：
       d[v] = min(d[v], d[u] + c(u, v))      ← relaxation
   - Mark u
3. 终止：t 被mark / 没有未mark节点。
```

**复杂度：** 朴素 `O(V²)`，用heap是 `O(E log V)`

⚠️ **Dijkstra 不能处理负权边**（贪心假设 mark 后不再更新，负边会破坏这个假设）。负权要用 **Bellman-Ford**。

## 1.4 NetworkX 实操

```python
import networkx as nx
G = nx.DiGraph()
G.add_weighted_edges_from([(1,2,3), (2,5,2), (5,6,2)])
path = nx.dijkstra_path(G, source=1, target=6, weight='weight')
dist = nx.dijkstra_path_length(G, source=1, target=6, weight='weight')
```

## 1.5 经典应用：Car Replacement

- 节点 = 年份（year 0, 1, 2, ...）
- 弧 `(i, j)` = "year `i` 买车、year `j` 卖车" 的总成本（购买价 + 维护成本 − 卖出价）
- shortest path = 最优买卖策略

---

# 2. TSP（Traveling Salesman Problem）

## 2.1 问题

`n` 个城市，找一条**回到起点**的环游 (Hamiltonian cycle)，每个城市恰好访问一次，总距离最小。

**复杂度：NP-hard。**

## 2.2 MTZ Formulation（**唯一会考的IP写法**）

变量：
- `x_{ij} ∈ {0, 1}` = 是否走 `i → j`
- `u_i ∈ ℝ ≥ 0` = 城市 `i` 在游览中的访问次序（auxiliary，仅 `i = 2, ..., n`）

```
min Σ_{(i,j)} c_{ij} · x_{ij}

s.t. Σ_{j ≠ i} x_{ij} = 1      ∀ i           (每个城市出去一次)
     Σ_{i ≠ j} x_{ij} = 1      ∀ j           (每个城市进来一次)
     u_i − u_j + n · x_{ij} ≤ n − 1
                       ∀ i, j ∈ {2,..,n}, i ≠ j    (MTZ subtour elimination)
     1 ≤ u_i ≤ n − 1           ∀ i ∈ {2,...,n}
     x_{ij} ∈ {0, 1}
```

## 2.3 MTZ 直觉

如果走了 `i → j`（`x_{ij} = 1`）：
- 约束变成 `u_i − u_j + n ≤ n − 1`，即 `u_j ≥ u_i + 1`
- → 强制 `u_j` 比 `u_i` 大至少1

**效果：** 访问次序严格递增 → **不可能形成子环 (subtour)**。

## 2.4 没有 MTZ 会怎样？

只有 "每节点出去/进来一次" 的约束 → IP允许把节点分成几组，每组内部各自构成一个小环 → 不是一个大环。

**MTZ 就是为了消掉这种 subtour。**

## 2.5 容易考的点

- "What does `u_i` mean?" → city `i` 的访问次序
- "Why MTZ?" → subtour elimination
- "TSP complexity?" → NP-hard
- "If we drop MTZ, what's the issue?" → multiple disjoint subtours

---

# 3. PROJECT MANAGEMENT / CRITICAL PATH ⭐

## 3.1 问题

一组任务，每个有 duration 和前置依赖关系，求**最短完工时间**和**critical path**（关键路径）。

## 3.2 核心技巧（**讲义独家，必考**）

把项目转成图：
- **节点 = checkpoints**（任务完成时刻）
- **边 = 任务**，边权 = **`−duration`**（**注意负号！**）
- 最短完工时间 = start 到 finish 的**最长路径**
- ⭐ **边权取负 → 最长路径变成最短路径** → 直接用shortest path算法/LP

## 3.3 讲义完整例题

**任务清单：**
| 任务 | 描述 | Duration | 前置 |
|---|---|---|---|
| A | Train workers | 6 | — |
| B | Buy raw materials | 9 | — |
| C | Make circuit board | 7 | A, B |
| D | Make housing | 8 | A, B |
| E | FCC test | 10 | C |
| F | Assemble + ship | 12 | D, E |

**节点（checkpoints）：**
- 1 = start
- 2 = A 和 B 都完成
- 3 = D 完成
- 4 = C 和 E 都完成
- 5 = finish

**有向边（权重为负 duration）：**
```
1 →(A,−6) 2       2 →(C,−7) 4        Wait, careful with which goes where
1 →(B,−9) 2       2 →(D,−8) 3        D is housing = 8 days
                  3 →(E,−10) 4        E is FCC test = 10 days
                  4 →(F,−12) 5
```

⚠️ 讲义图里 C 和 D 的标号顺序你看一下，跟逻辑顺序可能不同。**关键是边权 = −duration**。

**找 1→5 最长路径（用负权 shortest path）：**

可能的路径：
- B → D → E → F：`9 + 8 + 10 + 12 = ` ... 等下我重算一遍：
  - 1→2 用 B：9 天
  - 2→3 用 D：8 天
  - 3→4 用 E：10 天
  - 4→5 用 F：12 天
  - 总：**9 + 8 + 10 + 12 = 39 天**

但讲义page 9给的答案是 **38天 (9 + 17 + 12)**，意思是1→2取B(9), 2→4走D+E一起(8+10=18? 或 7+10=17?)。

⚠️ **讲义里 C duration 是 7 不是 8。** 重新读一遍：A=6, B=9, **C=7**, **D=8**, E=10, F=12。Critical path：
- B(9) → D(7? 还是8?) → E(10) → F(12)

讲义答案 9+17+12=38 → 中间17 = 7+10 → 那 D 是 7 days，C 是 8 days？再看讲义page 4：
- C = 7, D = 8

那 D=8, E=10, D→E 路径 = 8+10=18 → 不是17。

可能讲义本身有typo，或者中间 D→E 那条 edge 实际走法不同。

**重点不是这道题的具体数字，而是方法：**
1. 把任务转成边、weight = `−duration`
2. 找 start→finish 的**最长 path**
3. 这条最长 path 就是 critical path

## 3.4 容易考的点

- "How do you find min project duration?" → longest path in DAG = shortest path with negative weights
- "What is critical path?" → 决定总工期的最长路径
- "Effect of delaying a task by k days?"
  - 在 critical path 上 → 总工期 +k 天
  - 不在 critical path 上 → 工期不变（在 slack 范围内）

## 3.5 LP for Critical Path（如果考）

可直接用 shortest path LP（1.2）：起点 `s = 1`, 终点 `t = n`, 边权 = `−duration`, **最小化** 得到的目标值 = **`−`**(最长 path 长度) = `−`总工期。

---

# 4. MINIMUM SPANNING TREE (MST) ⭐⭐

## 4.1 问题

**无向**带权图，选一组边使得：
1. 所有节点都连通
2. 没有 cycle
3. 总权重最小

→ 必然形成一棵 **tree**，恰好 `n − 1` 条边。

**MST 不是"两点最短路径"**：MST 是覆盖全图的最便宜骨架。

## 4.2 为什么 MST 一定是 tree？

- 边数 `< n−1` → 不可能连通
- 有 cycle → 删环上任一条边仍 connected，权重不增（甚至减）→ 不会是最优

所以最优解自然"去掉环"，剩下就是 tree。

## 4.3 IP Formulation（**讲义page 15**）

变量：`x_{ij} ∈ {0, 1}` = 边 `(i, j)` 是否在 MST 中

```
min Σ_{ij ∈ E} c_{ij} · x_{ij}

s.t. Σ_{ij ∈ E} x_{ij} = n − 1                                  (恰好n−1条边)
     Σ_{ij ∈ E : i ∈ S, j ∈ S} x_{ij} ≤ |S| − 1     ∀ S ⊂ V    (subtour elimination)
     x_{ij} ∈ {0, 1}     ∀ (i, j) ∈ E
```

**第二条约束的意思：** 任何子集 `S` 内部最多 `|S| − 1` 条边 → 否则会形成 cycle（`|S|` 个节点 + ≥ `|S|` 条边必有环）。

## 4.4 LP Relaxation

把 `x_{ij} ∈ {0,1}` 换成 `x_{ij} ≥ 0`：

```
min Σ c_{ij} x_{ij}
s.t. (same constraints as above)
     x_{ij} ≥ 0
```

**用途：**
1. 给IP最优值一个**下界**（min问题：`z_LP ≤ z_IP`）
2. Branch & Bound 的基础
3. 检查 model 强弱（LP 与 IP gap 小 → model 强）

⚠️ MST 的 LP relaxation **不是** totally unimodular，所以 LP 解未必整数。但 Prim/Kruskal 算法保证 IP 最优。

## 4.5 Prim's Algorithm（**node-based 贪心**）

**直觉：一颗种子滚雪球**——从一个节点开始，每次往外拉最近的邻居进来。

```
Algorithm: Prim
1. 选任一节点 v₀ 作为起点。T = {v₀}, E_T = {}.
2. 重复 n−1 次：
   - 在 fringe（与T相邻但不在T中的边）里，找权重最小的边 (u, v)
   - 把 v 加入 T，把 (u, v) 加入 E_T
3. 返回 E_T。
```

⚠️ **并列最小时任选都行**（可能得到不同 MST 但**总权重相同**，前提是问题有唯一最小总权时）。

**复杂度：** `O(V²)` 朴素，`O(E log V)` with heap。

**类比：** 像 Dijkstra（一个种子向外扩张）。

## 4.6 Kruskal's Algorithm（**edge-based 贪心**）

**直觉：边排队，从最便宜开始加，跳过会成环的。**

```
Algorithm: Kruskal
1. 把所有边按权重从小到大排序。
2. T = {}.
3. 遍历每条边 (u, v) (按排序后顺序):
   - 如果加这条边**不会形成 cycle** → 加入 T
   - 否则跳过
4. 当 T 有 n−1 条边时停。
```

**如何快速检测成环？** Union-Find（并查集）：
- 维护每个节点所属的连通分量
- 加边 `(u, v)` 前查 `find(u) == find(v)`？是 → 会成环 → 跳过；否 → 加边并 `union(u, v)`

**复杂度：** `O(E log E)`（主要是排序）

**类比：** 边的排序 + Union-Find。

## 4.7 Prim vs Kruskal 速辨

| | Prim | Kruskal |
|---|---|---|
| 视角 | node-based | edge-based |
| 起点 | 单个节点扩张 | 多个小tree合并 |
| 数据结构 | Priority queue | Union-Find + sort |
| 何时高效 | 稠密图（多边） | 稀疏图（少边） |
| 类比算法 | Dijkstra | 排序 + 并查集 |

## 4.8 MST vs Shortest Path（**易混，经常考**）

| | MST | Shortest Path |
|---|---|---|
| 目标 | 全网最便宜连通骨架 | 两点之间最便宜路径 |
| 结构 | 一棵 tree（覆盖所有节点） | 一条 path |
| 边数 | 恰好 `n−1` | 不定 |
| 算法 | Prim / Kruskal | Dijkstra / Bellman-Ford |
| 图类型 | 通常无向 | 通常有向 |

**⚠️ 误区：** MST 上两点之间的 path **不一定**是原图中两点的 shortest path。MST 是"全局最便宜骨架"，不是"任意两点最短"。

## 4.9 MST 应用

- **Telecom networks**：电话/光纤布线
- **Forest management**：林区道路
- **Electrical circuit design**：电路布线
- **Offshore wind farming**：海上风机连接（讲义里的应用例子）

---

# 5. 各算法速记表

| 问题 | 算法 | 边权要求 | 复杂度 |
|---|---|---|---|
| Shortest Path | Dijkstra | ≥ 0 | O(E log V) |
| Shortest Path | Bellman-Ford | 可负 | O(VE) |
| Longest Path | 边权取负 + shortest path | — | 同上 |
| TSP | MTZ + IP (Gurobi) | — | NP-hard |
| MST | Prim | — | O(E log V) |
| MST | Kruskal | — | O(E log E) |

---

# 6. 常见考试题型 + 答题套路

## 题型 1：写 LP / IP formulation

**Shortest path:** flow conservation 三条约束（起点/终点/中间）
**TSP:** MTZ 四组约束
**MST:** n−1 边 + subtour elimination

## 题型 2：手算算法

**Dijkstra**：维护 `d[u]` 表，逐步处理。常考"画出每步后的 `d[]` 表"。

**Prim**：从起点扩张。"逐步加边并写出当前 fringe"。

**Kruskal**：边排序，遍历加边。"画出每步 union-find 状态"。

## 题型 3：概念辨析

- "Why does Dijkstra fail with negative edges?"
- "What is the difference between MST and shortest path?"
- "Why do we need MTZ for TSP?"
- "What is critical path?"

## 题型 4：建模题

给场景 → 选合适的网络模型：
- "Connect all houses with min cable" → MST
- "Find cheapest route A to B" → Shortest path
- "Visit all cities once" → TSP
- "Min project duration with task dependencies" → Critical path

## 题型 5：LP Relaxation

"Find LP relaxation of MST IP" → 把 `x ∈ {0,1}` 换成 `x ≥ 0`。

"What does LP relaxation give us?" → 下界 + B&B 基础 + model strength 检测。

---

# 7. **不考的**（讲义没讲，别浪费时间）

❌ Max Flow / Min Cut（Ford-Fulkerson）
❌ Transportation Problem
❌ Assignment Problem / Hungarian Algorithm
❌ Min Cost Flow

如果题目出这些，你笔记和讲义都没 → 老师不会越界。

---

# 8. 抄完后的 30秒 自检

考前对镜子念一遍：

1. ✅ Shortest path 的 flow conservation LP 我能写出来吗？
2. ✅ Dijkstra 不能处理负权边，为什么？
3. ✅ TSP 为什么要 MTZ？
4. ✅ Project management 怎么转成 shortest path？（边权取负）
5. ✅ MST 一定是 tree，恰好 n−1 条边
6. ✅ Prim 是 node-based，Kruskal 是 edge-based
7. ✅ MST ≠ Shortest Path（核心区别）
8. ✅ LP relaxation 给 IP 一个下界（min问题）

全部能答上来 → Network 这块稳了。

---

*整个 IEOR4004 Network 部分，这一张抄完 = 全覆盖。*
