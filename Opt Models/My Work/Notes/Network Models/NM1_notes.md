% NM1 课程笔记 - 网络模型与方法

作者: 整理自 `NM1.pdf` 与课堂对话

--

## 课程提要（Outline）
- 介绍网络模型（图的基本概念）
- 旅行推销员问题（TSP）回顾（LP / MTZ）
- 最短路径问题（LP 表述、Dijkstra 算法）
- NetworkX 库简介与实操提示
- 实例：汽车更换问题、项目管理网络

## 重要定义与记法
- `vertices`：顶点 / 节点（中文常写“顶点”或“节点”）。
- `edges`：边（无向图），在有向图中通常称为弧（arc）。
- 记法 `V(G)`, `E(G)`：
  - `V(G)` 表示图 `G` 的顶点集合，`E(G)` 表示图 `G` 的边集合。
  - 加上 `(G)` 是为了把集合与具体的图绑定，方便同时讨论多个图时不歧义。

## 有向图 vs 无向图（Directed vs Undirected）
- 有向边写作有序对 `(u, v)`（方向从 `u` 指向 `v`）；无向边可写作无序对 `{u, v}`，等同于 `(u,v)` 和 `(v,u)`。
- 判断图是否有向的实用方法：
  1. 视觉检查：图上边是否有箭头。
  2. 边表/邻接表：如果存在 `(u,v)` 但没有 `(v,u)`，通常是有向图。
  3. 邻接矩阵 `A`：若 `A != A^T`（非对称）则为有向图；若 `A = A^T` 则为无向图。
  4. 度的概念：有向图有入度（in-degree）和出度（out-degree）；无向图只有 degree。

### NetworkX 检测示例
```python
import networkx as nx
G = nx.read_edgelist("edges.txt")  # 或者创建时指定 create_using=nx.DiGraph()
nx.is_directed(G)  # True = 有向, False = 无向

import numpy as np
A = nx.to_numpy_array(G)
np.allclose(A, A.T)  # True -> 无向（对称）
```

## 路径、环、树与森林
- 路径（path）：依次连接的一系列顶点与边；有向路径要求边方向一致。
- 环（cycle）：首尾相连且不重复节点（除首尾）。
- 树（tree）：无向、连通且无环；含 n 个节点的树恰有 n−1 条边。
- 森林（forest）：无环但不一定连通的图（即多棵树的集合）。

## 最短路径问题（Shortest Path）
- 问题：给定有权图 `G=(V,E)` 与起点 `s`、终点 `t`，求 `s` 到 `t` 的最短路径（权重和最小）。
- LP 表述（常用思想）：对每条边设置决策变量 `x_{ij}`，并写流守恒约束（source/sink），该 LP 在无负循环时成立。

### Dijkstra 算法（要点）
- 适用条件：边权非负。
- 基本思想：贪心 — 每步选择当前未处理的、距离 `s` 最近的顶点 `u`，对 `u` 的邻居进行松弛（relaxation），直至所有顶点处理完或到达 `t`。
- 伪步骤：
  1. 初始化：`d[s]=0`，其它 `d[u]=+inf`；所有顶点标记为未处理。
  2. 重复：选择未处理顶点 `u` 使 `d[u]` 最小；对每个 `(u,v)` 执行 `d[v]=min(d[v], d[u]+w(u,v))`；标记 `u` 为已处理。
  3. 终止：当 `t` 被处理或没有未处理顶点可选时。

### NetworkX 使用示例（最短路径）
```python
import networkx as nx
G = nx.DiGraph()
G.add_weighted_edges_from([(1,2,1),(2,5,2),(5,6,2)])
path = nx.dijkstra_path(G, source=1, target=6, weight='weight')
dist = nx.dijkstra_path_length(G, source=1, target=6, weight='weight')
```

## 旅行推销员问题（TSP）简要
- 问题：在 n 个城市间找一条巡回（回到起点），使总距离最小。是 NP-hard 的组合优化问题。
- 常见数学表述：整数规划（binary x_{ij} 表示是否走从 i 到 j）。
- 子回路（subtour）问题：裸 IP 会产生多个子回路，需要额外约束消除。常用的 MTZ（Miller–Tucker–Zemlin）约束通过辅助变量 `u_i` 指定访问次序以防止子回路。

## 应用示例小结（笔记中提到的例子）
- 汽车更换问题：把年份当作节点，买卖动作作为边，边权为成本（购买价差 + 维护），问题转为图上的最短路径。
- 项目管理（关键路径）：任务与节点构建网络，边权为持续时间，关键路径法（CPM）用于计算最短完工时间与关键任务链。

## 常用符号速查
- `V(G)`：顶点集合；`E(G)`：边集合。  
- `c(e)` 或 `c(u,v)`：边 `e` 或弧 `(u,v)` 的权重/费用。

## 我们对话中的实用建议（行动项）
- 想实际练习的话：在 notebook 中用 `networkx` 建一个有向/无向图，练习 `is_directed()`, `dijkstra_path()`。  
- 若需要，我可以把本笔记转成更系统的学习计划、练习题和答案（或把某页内容逐句详解）。

--

文件位置：
`Opt Models/Lecture/NM1_notes.md`
