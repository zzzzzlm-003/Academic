# LP7: Duality

## Introduction to Duality Theory

- For a maximization LP, any feasible solution provides a lower bound on the optimal value.
- We can construct a new LP, called the **dual** problem, to find an upper bound on the optimal value of the original LP (the **primal** problem).

## Primal and Dual Problems

- The original problem is called the **primal**.
- The new problem is called the **dual**.
- If the primal is a maximization problem, the dual is a minimization problem, and vice-versa.

### Converting a Primal to a Dual (Standard Form)

For a primal in the form:
`max c^T * x`
`s.t. A*x <= b`
`x >= 0`

The dual is:
`min b^T * y`
`s.t. A^T * y >= c`
`y >= 0`

**Conversion Rules:**

| Primal (Max)                   | Dual (Min)                     |
| :----------------------------- | :----------------------------- |
| i-th constraint <=             | i-th variable y_i >= 0         |
| i-th constraint >=             | i-th variable y_i <= 0         |
| i-th constraint =              | i-th variable y_i unrestricted |
| j-th variable x_j >= 0         | j-th constraint >=             |
| j-th variable x_j <= 0         | j-th constraint <=             |
| j-th variable x_j unrestricted | j-th constraint =              |

## Duality Theorems

### 1. Weak Duality Theorem

- The value of any feasible solution to the dual problem provides an upper bound on the value of any feasible solution to the primal problem.
- If the primal is unbounded, the dual must be infeasible.
- If the dual is unbounded, the primal must be infeasible.
- It is possible for both to be infeasible.

### 2. Strong Duality Theorem

- If a primal problem has an optimal solution, its dual problem also has an optimal solution, and their optimal values are equal.

### 3. Complementary Slackness Theorem

- At optimality, if a primal constraint is not tight (has positive slack), the corresponding dual variable must be 0.
- `A_i * x < b_i  => y_i = 0`
- Conversely, if a primal variable is strictly positive, the corresponding dual constraint must be tight (slack is 0).
- `x_j > 0 => (A^T)_j * y = c_j`
- 

* 如果资源有富余（松弛 > 0），那它的价格就必须是 0。
  * 不松弛，全用完了，价格 $y_i >0$
* 如果资源价格不为零（价格 > 0），那它必须被用得精光（松弛 = 0）。

## Economic Interpretation of the Dual Problem

- The dual variables can be interpreted as the "shadow price" of the resources in the primal problem.
- The shadow price represents the marginal value of an additional unit of a resource.
- For a resource with slack (i.e., not fully used), the shadow price is 0.

---

## Discussion & Clarification

**Question:** 这个bounded的意思，是不是就，找到约束条件，相加相乘，看有没有一定大于objective value的，如果有，那就一定有上界?
(Does 'bounded' mean that you can find a combination of the constraints (by multiplying and adding them) that results in an expression that is always greater than or equal to the objective function? If such a combination exists, does that guarantee an upper bound?)

**Answer:** 是的，您的理解非常准确。这正是寻找线性规划（LP）问题上界的核心思想，也是对偶理论（Duality Theory）的直观解释。

具体来说：

1. **寻找上界**：当您处理一个最大化（Maximization）问题时，您可以通过对约束条件进行线性组合（即您所说的“相加相乘”），来构造一个新的不等式。
2. **判断条件**：如果这个新构造的不等式的左边（变量侧）在所有可行解上都**大于或等于**原始目标函数，那么这个新不等式的右边（数值侧）就构成了原始目标函数的一个**上界（Upper Bound）**。
3. **Bounded的含义**：如果能找到这样一个上界，我们就说这个LP问题是**有界的（Bounded）**。

这正是 `LP7.pdf`的第12到16页所演示的过程。通过为约束条件寻找合适的乘数（这些乘数后来成为了**对偶变量**），我们能够为原始问题（**Primal Problem**）的目标值设定一个限制，而寻找最优（最小）上界的过程，就引出了**对偶问题（Dual Problem）**。

---

**Question:** 第25页URS 是什么意思，为什么说有weak duality 还有strong吗?
(What does URS on page 25 mean? And if there's weak duality, is there also a strong one?)

**Answer:** 您提了两个很好的问题，我们逐一来看：

1. **URS 的含义**:
   `URS` 是 **Unrestricted in Sign** 的缩写，意思是“**符号不受限制**”。在线性规划中，这表示变量（在这个例子中是 `x1`）可以取正数、负数或零，而不像典型的变量那样被限制为必须大于等于零 (`≥ 0`)。

   正如您在幻灯片第23页的转换规则表中看到的，一个“unrestricted”的原始变量（Primal variable）正好对应一个“等式” (`=`) 的对偶约束（Dual constraint）。因此，在第25页的例子中，因为 `x1` 是 URS，所以它对应的第一个对偶约束是 `y1 + 2y2 + y3 = 3`，这是一个等式而不是不等式。
2. **Weak Duality 与 Strong Duality**:
   是的，您说得很对。既然有 **Weak Duality (弱对偶性)**，那也一定有 **Strong Duality (强对偶性)**。它们是描述原始问题和对偶问题之间关系的两个核心定理，在您之前让我总结的笔记里也有提及：

   * **弱对偶性 (Weak Duality)** - (幻灯片第27页)

     * 它指出：对于*任何*可行的原始解和*任何*可行的对偶解，原始问题（最大化问题）的目标值 **小于或等于** 对偶问题（最小化问题）的目标值。
     * 它之所以“弱”，是因为它只提供了一个**界限**（一个不等式关系），这个关系对于任何可行解都成立，但并不精确。
   * **强对偶性 (Strong Duality)** - (幻灯片第31页)

     * 它指出：如果原始问题有*最优*解，那么它的对偶问题也一定有*最优*解，并且两个问题的**最优解的目标值是相等的**。
     * 它之所以“强”，是因为它给出了一个非常强大的**等式**结论，但这个结论只在“最优”这个特定情况下才成立。

简单总结一下：**弱对偶性**说的是“一个不比另一个大”，而**强对偶性**说的是“在最好的情况下，它俩一样大”。

---

**Question**: primal和dual的关系，我记得有什么相关性因果性，哦不对 充分必要条件，就是这俩的无界和没有可行集互为充要条件对吧?
(Regarding the relationship between primal and dual, I remember something about correlation, causality... oh wait, necessary and sufficient conditions. Is it that unboundedness and infeasibility are necessary and sufficient conditions for each other between the two?)

**Answer**: 您对这个关系的记忆非常准，这确实是对偶理论中一个非常重要且容易混淆的知识点。您提到的“充分必要条件”很接近了，但实际关系稍微有些不同，它更多是**充分条件**，而不是充要条件。

让我为您梳理一下：

1. **单向的充分条件**：

   * 如果**原始问题是无界的 (Primal is unbounded)**，那么**对偶问题一定是不可行的 (Dual must be infeasible)**。
   * 反之亦然：如果**对偶问题是无界的 (Dual is unbounded)**，那么**原始问题也一定是不可行的 (Primal must be infeasible)**。

   这是一个**单向的推论**。为什么呢？根据弱对偶性，任何一个可行的对偶解都会给原始问题（最大化问题）提供一个上界。但如果原始问题是无界的，就意味着它没有任何上界，这与“存在一个可行的对偶解”相矛盾。所以，对偶问题必须是不可行的。
2. **为什么不是充要条件？**
   您提到的“互为充要条件”意味着反过来也必须成立，即“如果一个问题是不可行的，另一个问题必须是无界的”。但这**不一定**正确。

   这里存在第三种可能：**原始问题和对偶问题可能“双双不可行” (Both can be infeasible)**。

### 幻灯片第30页的总结表

这张表最清楚地总结了所有可能性：

|                                        | **Primal Infeasible** (原始不可行) | **Primal Bounded** (原始有界) | **Primal Unbounded** (原始无界) |
| :------------------------------------- | :--------------------------------------- | :---------------------------------- | :------------------------------------ |
| **Dual Infeasible** (对偶不可行) | **可能 (Possible)**                | **不可能 (Impossible)**       | **一定 (Must be)**              |
| **Dual Bounded** (对偶有界)      | **不可能 (Impossible)**            | **一定 (Must be)**            | **不可能 (Impossible)**         |
| **Dual Unbounded** (对偶无界)    | **一定 (Must be)**                 | **不可能 (Impossible)**       | **不可能 (Impossible)**         |

**核心结论**：

* 您记忆中的“一个无界 <=> 另一个不可行”说对了一半，正确的说法是：**“一个无界 => 另一个不可行”**。
* 关键的区别在于：“一个不可行”**不一定**能推出“另一个无界”，因为存在两者都不可行的可能性。

---

## 深入理解互补松弛性 (In-Depth: Complementary Slackness)

### 一句话总结

**在最优解中，一个约束的“富余量”（松弛）和它所对应的“影子价格”（对偶变量）的乘积永远是零。**

换句话说，两者中必须至少有一个是零：

* 如果资源有富余（松弛 > 0），那它的价格就必须是 0。
* 如果资源价格不为零（价格 > 0），那它必须被用得精光（松弛 = 0）。

### 关键概念：“松弛”是什么？ (What is "Slack"?)

“松弛” (Slack) 就是“剩余量”或“未用尽的量”。

#### 背包的比喻

假设你的背包**最多能装10公斤**的东西。这是一个**约束** (`背包重量 <= 10`)。

* **情况一：有松弛 (Has Slack)**
  你今天只放了**7公斤**的书进去。

  * 你的“剩余量”是多少？`10 - 7 = 3` 公斤。
  * 这 **3公斤** 就是“**松弛量**”。
  * 在这种情况下，我们说背包的重量约束是**不紧的 (Not Binding)**，因为它没被用满，有富余。
* **情况二：没有松弛 (Zero Slack)**
  你今天正好放了**10公斤**的东西进去。

  * 你的“剩余量”是多少？`10 - 10 = 0` 公斤。
  * “**松弛量**”为 **0**。
  * 在这种情况下，我们说背包的重量约束是**紧的 (Binding / Tight)**，因为它被“顶格”用满了，一点富余都没有。

#### 回到线性规划问题

现在我们把这个概念应用到家具公司的例子里：

* **约束**: 公司最多有**100小时**的木工工时。(`木工总工时 <= 100`)
* **有松弛**: 如果你的生产计划（生产x1张桌子，x2把椅子）一共只用了 **80小时**的木工工时。那么**松弛量就是 `100 - 80 = 20` 小时**。这个约束就是“不紧的”。
* **没有松弛**: 如果你的生产计划正好用完了**100小时**的木工工时。那么**松弛量就是 `100 - 100 = 0`**。这个约束就是“紧的” (Binding/Tight)。

---

### 数学应用：如何检验最优性

互补松弛性最强大的应用在于**检验最优性**，它把经济学直觉变成了可以严格计算的步骤。我们用幻灯片第34页的例子来走一遍数学流程：

**已知信息**:

* **原始问题**:
  * `C1: x1 + x2 <= 6`  (我们叫它“资源1”)
  * `C2: 0.5x1 + x2 <= 4`  (我们叫它“资源2”)
* **对偶问题**:
  * `D1: y1 + 0.5y2 >= 3` (对应 `x1`的“盈利”约束)
  * `D2: y1 + y2 >= 2` (对应 `x2`的“盈利”约束)
* **一个猜想的最优解**: `x1 = 6`, `x2 = 0`

#### 数学验证步骤

**第一步：检查原始问题的约束是“紧的”还是“松的”**

1. **代入C1 (资源1)**: `x1 + x2 = 6 + 0 = 6`。

   * 约束是 `<= 6`，结果正好是 `6`。
   * **结论**: 约束 C1 是**紧的 (Binding)**，松弛量为0。资源1被用光，它的影子价格 `y1` *可以*大于0。
2. **代入C2 (资源2)**: `0.5*x1 + x2 = 0.5*6 + 0 = 3`。

   * 约束是 `<= 4`，结果是 `3`。`3 < 4`。
   * **结论**: 约束 C2 是**松的 (Slack)**，松弛量为 `4 - 3 = 1`。
   * **数学规则推论**: **`y2` 必须等于 0**。(因为资源2有剩余，不值钱)

**第二步：检查原始问题的变量是“正的”还是“零”**

1. **看变量 `x1`**: `x1 = 6`，它是正数 (`> 0`)。

   * **数学规则推论**: `x1` 对应的对偶约束 D1 `y1 + 0.5y2 >= 3` **必须取等号**。(因为我们决定生产 `x1`，说明它能带来的利润正好被资源的价值完全覆盖)
2. **看变量 `x2`**: `x2 = 0`。

   * 规则没说此时对应的约束D2该怎么样。

**第三步：解方程，得出结论**

现在我们把上面用数学规则得到的“推论”组合起来：

* 从第一步我们知道: `y2 = 0`
* 从第二步我们知道: `y1 + 0.5y2 = 3`

把 `y2 = 0` 代入第二个式子，我们得到 `y1 + 0 = 3`，所以 **`y1 = 3`**。

我们现在有了一个完整的对偶解 `y1 = 3, y2 = 0`。最后一步是检查这个解是否满足所有对偶约束（尤其是我们之前忽略的D2）。

* 代入D2: `y1 + y2 = 3 + 0 = 3`。`3 >= 2`。**约束成立！**

**最终结论**: 因为我们根据 `x = (6, 0)` 和互补松弛数学规则，成功地找到了一个完全满足所有对偶约束的对偶解 `y = (3, 0)`，所以我们证明了 `x = (6, 0)` **确实是最优解**。
