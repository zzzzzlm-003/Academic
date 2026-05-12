# IP5 课堂笔记（TSP + 网络优化）

## 课程定位
IP5 主要围绕 **旅行商问题（TSP）** 与 **MTZ 子回路消除**，并结合 `gurobipy` / `networkx` 做建模和求解。

参考材料：
- [Opt Models/Lecture/Integer Programming/IP5/IP5.pdf](../Lecture/Integer%20Programming/IP5/IP5.pdf)
- [Opt Models/Lecture/Integer Programming/IP5/IEOR4004_TSPwithMTZ.ipynb](../Lecture/Integer%20Programming/IP5/IEOR4004_TSPwithMTZ.ipynb)
- [Opt Models/Lecture/NM/IEOR4004_ShortestPathProblem.ipynb](../Lecture/NM/IEOR4004_ShortestPathProblem.ipynb)

---

## 一、TSP 核心模型

### 1) 决策变量
- `x_{ij} \in \{0,1\}`：是否从城市 `i` 走到城市 `j`。

### 2) 目标函数
最小化总路程：

$$
\min \sum_{i}\sum_{j} c_{ij}x_{ij}
$$

### 3) 度约束（每点进 1 出 1）

$$
\sum_{j}x_{ij}=1,\ \forall i
$$

$$
\sum_{i}x_{ij}=1,\ \forall j
$$

并且禁止自环：`x_{ii}=0`。

---

## 二、为什么要 MTZ
只写“每点进1出1”会出现多个小环（subtours），不是一条覆盖所有点的大环。

MTZ 通过顺序变量 `u_i` 消除子环。

### MTZ 常见形式
对 `i,j \in \{2,\dots,n\}, i\neq j`：

$$
u_j \ge u_i + 1 - (n-1)(1-x_{ij})
$$

并设置：

$$
1 \le u_i \le n-1
$$

起点通常固定顺序（例如 `u_1 = 0` 或不参与约束）。

---

## 三、IP5 notebook 对应实现要点
在 [Opt Models/Lecture/Integer Programming/IP5/IEOR4004_TSPwithMTZ.ipynb](../Lecture/Integer%20Programming/IP5/IEOR4004_TSPwithMTZ.ipynb) 中：

1. 用坐标构建距离矩阵 `Distance`。
2. 建立二进制变量 `x[i,j]`。
3. 建立连续变量 `u[i]`（其中起点 `u[0]` 固定为0）。
4. 添加进/出约束与 MTZ 约束。
5. 求解并输出顺序。
6. 额外演示了两种启发式：
   - Nearest Neighbor（最近邻）
   - Christofides

---

## 四、和最短路径（Shortest Path）的关系
- 最短路径：从 `s` 到 `t` 的一条最短路（单源-单汇）
- TSP：必须访问所有城市并回到起点（组合爆炸更严重）

最短路径（流平衡）常用写法：

$$
\min \sum_{(i,j)\in E} c_{ij}x_{ij}
$$

$$
\sum_j x_{sj}-\sum_j x_{js}=1,
\sum_j x_{tj}-\sum_j x_{jt}=-1,
\sum_j x_{ij}-\sum_j x_{ji}=0\ (i\neq s,t)
$$

---

## 五、考试速记（高频）
1. 只写度约束不够，必须处理子回路。
2. MTZ 用 `u` 变量表达访问顺序。
3. 注意 `x_{ii}=0`。
4. Dijkstra 仅适用于非负边权。
5. TSP 是 NP-hard，启发式速度快但不保证全局最优。

---

## 六、今天问答记录（整理版）

### Q1（你）：
要考试了，上课没听，带我学。

### A1（我）：
给了冲刺结构：
- 先抓最短路径 LP + Dijkstra 条件
- 再抓 TSP 度约束 + MTZ
- 最后给考试高频易错点（自环、进出约束、MTZ下标、Dijkstra负权等）

---

### Q2（你）：
LP5 里除了 MTZ 和 PDF，其他文件该归到哪里？

### A2（我）：
识别并归类为两组：
- Vaccine 案例组：`IE4004_InClass_VaccineAllocation.ipynb`、`vaccine.lp`、`vaccine.mps`
- Shipment 案例组：`Shipment_Problem.ipynb`、`shipment.lp`、`shipment_data.xlsx`
并确认这些 `.lp/.mps` 是 notebook 运行导出的模型文件。

---

### Q3（你）：
整理。

### A3（我）：
已完成目录整理：
- [Opt Models/Lecture/Linear Programming/LP5/Vaccine](../Lecture/Linear%20Programming/LP5/Vaccine)
- [Opt Models/Lecture/Linear Programming/LP5/Shipment](../Lecture/Linear%20Programming/LP5/Shipment)
并将对应文件移动到分类目录。

---

### Q4（你）：
先带我学 IP5，再写 classnotes，把问答内容都放进去。

### A4（我）：
本文件即为完整课堂笔记 + 今日问答汇总。

---

## 七、接下来怎么复习（30 分钟）
1. 先过一遍本文件第 1~5 节（10 分钟）。
2. 打开 IP5 notebook，对照约束逐行看（10 分钟）。
3. 手写一次 TSP+MTZ 数学模型（10 分钟）。

完成标准：你能在不看资料的情况下写出
- 目标函数
- 进/出约束
- MTZ 约束
- 自环约束 `x_{ii}=0`

---

## 八、课件图示与术语补充（后续追问）

### 1) `celestial*` 是什么
- `celestial` = 天体的、天空中的。
- 在课件里可理解为 `astronomical objects`（天文观测目标）。
- 含义：TSP 不只用于物流，也可用于天文成像/观测路径规划。

### 2) 图里的圆圈是什么意思
- 每个圆圈是“可接受访问区域（neighborhood）”，不是必须到圆心。
- 半径通常由业务容忍度决定（定位误差、观测范围、覆盖半径等）。
- 半径越大，路径可选空间通常越大；圆圈重叠越多，路径设计更复杂。

### 3) P8 海浪图想表达什么
- 是在说明 TSP 可用于图像路径规划（如扫描顺序、绘图轨迹优化）。
- 核心思想：把要访问的点串成一条总长度尽量短的路径。

---

## 九、复杂度与算法概念（口语版）

### 1) NP
- 给你一个候选解，能在多项式时间内快速验证对错。

### 2) NP-hard
- 至少和 NP 中最难问题一样难。
- TSP（优化版）是 NP-hard。

### 3) Heuristic（启发式）
- 快速找“不错的解”，但不保证全局最优。

---

## 十、P15 逐句释义（你发图那页）

- **How To Solve TSP Computationally?**  
   如何用计算方法求解 TSP。
- **TSP resembles a totally connected network... does not share integrality properties...**  
   TSP 看起来像网络流，但没有“LP 松弛自动整数”的好性质。
- **Heuristic approaches exist / Find good solutions / No guarantee of quality**  
   有启发式；能快找好解；但不保证最优。
- **We will approach it as a binary integer program**  
   用 0-1 整数规划建模。
- **Guarantee on global optimality**  
   精确法在求解完成时可保证全局最优。
- **Multiple ways to do this**  
   建模不止一种（如 MTZ、DFJ）。
- **Symmetric TSP... distance i->j same as j->i**  
   对称 TSP：`c_{ij}=c_{ji}`。

---

## 十一、子回路（Subtour）补充说明

### 1) 只靠度约束为什么不够
- 即使每点都“进1出1”，仍可能出现多个小圈（不是一个完整大圈）。

### 2) SEC 的作用（直觉）
- 约束任意真子集不能自成闭环。
- 等价理解：任何子集都必须和外部保持连接，不能封闭成孤岛。

### 3) “出现子回路就加约束再求解”在做什么
- 这是 cutting-plane / lazy constraints 思路：
   1. 先解基础模型
   2. 发现子回路
   3. 只补针对该子回路的约束
   4. 重复直到只剩单一大圈
- 原因：所有子回路约束一次性加完数量过大。

---

## 十二、MTZ（小学生版再记一遍）

- 给城市一个访问顺序号 `u[i]`。
- 如果走了 `i -> j`，就强制 `j` 排在 `i` 后面。
- 这样小圈会产生顺序矛盾，因此被禁止。

常用写法（与前文一致）：
- `u_j >= u_i + 1 - (n-1)(1-x_{ij})`
- `1 <= u_i <= n-1`（通常 `i=2..n`，起点单独处理）

---

## 十三、今天新增问答记录（补充）

### Q5（你）：
`celestial*` 是什么？图啥意思？

### A5（我）：
解释为天文观测目标；图展示球面上的 TSP with neighborhoods，比较不重叠与重叠实例。

### Q6（你）：
圆圈半径怎么选？和问题有什么关系？

### A6（我）：
半径来自业务容忍度/覆盖范围；半径和重叠程度会改变可行域与路径复杂度。

### Q7（你）：
P8 图想表达什么？是不是日本海浪图？

### A7（我）：
是图像路径规划应用场景，核心是“把访问点连成较短轨迹”。

### Q8（你）：
什么是 NP-hard / NP / Heuristic？

### A8（我）：
分别解释了复杂度层级与启发式“快但不保最优”的特点。

### Q9（你）：
“integrality properties” 那句看不懂，是不是就不用松弛了？

### A9（我）：
结论是：不能只靠松弛当最终答案；通常要整数规划。松弛仍可用于下界和加速。

### Q10（你）：
subtour constraint 为什么能保证？

### A10（我）：
因为它禁止任意真子集自成环；配合度约束后只剩一个全局大环。

### Q11（你）：
“If a subtour appears... add constraint and solve again” 是什么？

### A11（我）：
是迭代加割方法：发现小圈就补约束并重解，直到无子回路。

### Q12（你）：
MTZ 是啥？请小学生版。

### A12（我）：
用顺序变量控制访问先后，防止形成独立小圈。
