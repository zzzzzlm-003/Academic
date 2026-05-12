# Integer Programming (IP) Methods and Applications
## IEOR E4004: Optimization Models and Methods
**Instructor:** Yaren Bilge Kaya, PhD  
**Institution:** Columbia University, Department of Industrial Engineering and Operations Research

---

## Table of Contents
1. [Introduction to Solving Integer Programs](#introduction-to-solving-integer-programs)
2. [Branch and Bound Algorithm](#branch-and-bound-algorithm)
3. [Cutting Planes Algorithm](#cutting-planes-algorithm)
4. [Implementation Using Gurobi](#implementation-of-algorithms-using-gurobi)
5. [Traveling Salesperson Problem (TSP)](#traveling-salesperson-problem)

---

## Introduction to Solving Integer Programs

### Overview
Integer programs (IPs) are optimization problems where some or all decision variables must take integer values. This lecture covers two fundamental algorithmic approaches:
- **Branch and Bound Algorithm**
- **Cutting Planes Algorithm**

Additionally, we explore how to implement these algorithms using commercial solvers like Gurobi.

---

## Branch and Bound Algorithm

### Concept and Motivation
The Branch and Bound (B&B) algorithm is a systematic method for solving integer programs. The basic idea is to:
1. Relax the integrality constraints to obtain a linear program (LP)
2. Recursively partition the solution space
3. Use bounds to eliminate branches that cannot contain the optimal solution

### Step-by-Step Process

#### Step 1: Linear Relaxation
Remove the integrality constraints from the original integer program to obtain the LP relaxation. This LP provides an **upper bound** on the optimal integer solution for maximization problems.

#### Step 2: Check Solution Integrality
Solve the LP relaxation:
- If the optimal solution is integer-valued, we have found the optimal IP solution
- If the optimal solution contains fractional values, proceed to Step 3

#### Step 3: Branching
Select a fractional variable and create two sub-problems:
- **Sub-problem 1:** Add constraint that the variable is rounded down
- **Sub-problem 2:** Add constraint that the variable is rounded up

This partitions the solution space into smaller regions, each solved recursively.

#### Step 4: Pruning/Fathoming
A node in the branch-and-bound tree can be pruned if:
- The LP relaxation is infeasible
- The LP relaxation yields a solution worse than the current best known integer solution
- The LP relaxation is integer-valued (a feasible integer solution is found)

**Key Insight:** By maintaining the best feasible integer solution found so far, we can eliminate many branches early, significantly reducing computation.

### Algorithm Properties

**Pros:**
- Usually performs well in practice
- Guarantees finding the optimal solution (if the problem is bounded)

**Cons:**
- No theoretical guarantee on computational time to termination
- Can be slow for large-scale problems with many binary/integer variables

### Mixed-Integer Programming (MIP)
For mixed-integer programs where only some variables require integrality:
- Branch only on the integer-constrained variables
- LP relaxations of sub-problems may still contain fractional integer variables
- Same pruning strategies apply

---

## Cutting Planes Algorithm

### Motivation: Beyond LP Relaxations

#### Problem Formulations Vary in Strength
The same integer program can be formulated in multiple ways:
- All formulations yield identical integer solutions
- However, their LP relaxations can differ significantly
- Different LP relaxations provide different bounds (some tighter than others)

**Key Observation:** A tighter LP relaxation bound brings us closer to the true integer optimum, requiring less branching.

### Illustrative Example: LP vs. IP Solutions

Consider a simple IP:
$$\max \, c^T x$$
$$\text{subject to: } Ax \leq b$$
$$x \text{ integer}$$

The LP relaxation may yield a fractional solution that is far from any feasible integer solution. However, we can improve the LP by adding **cutting planes** (valid inequalities) that:
- Eliminate the fractional LP solution
- Do not eliminate any feasible integer solutions

### Finding Valid Inequalities

#### General Approach: Chvátal-Gomory (CG) Cuts

**Definition:** A **valid inequality** is a linear constraint that is satisfied by all feasible integer solutions but may be violated by the LP relaxation solution.

Consider a generic integer program:
$$\max \, c^T x$$
$$\text{subject to: } Ax \leq b$$
$$x \geq 0, \text{ } x \text{ integer}$$

**Construction of CG Cut:**

1. Select any vector $u$ with $u_i \geq 0$ for all $i$
2. Compute: $u^T A x \leq u^T b$
3. Since $x$ is integer, we can round down coefficients: $\lfloor u^T A \rfloor x \leq \lfloor u^T b \rfloor$
4. This yields a valid inequality that may cut off fractional solutions

**Why This Works:** The key is that due to integrality of $x$, we can round down the right-hand side without violating the constraint for any integer-feasible solution.

#### Example: Chvátal-Gomory Cut - Detailed Explanation

Given a system:
$$\max \, 3x_1 + 2x_2$$
$$\text{subject to: } 10x_1 + 6x_2 \leq 45$$
$$x_1, x_2 \geq 0, \text{ integer}$$

**Step 1: Solve LP Relaxation**
- Remove integrality constraints
- LP optimal: $(4.5, 0)$, with $Z_{LP} = 13.5$
- **Problem**: Not integer!

**Step 2: Choose Multiplier Vector and Combine Constraints**

This is the key step! Select $u = 0.5$ and multiply the constraint:
$$0.5 \times (10x_1 + 6x_2 \leq 45)$$
$$5x_1 + 3x_2 \leq 22.5$$

**Why This Works:**
- We're creating a new valid inequality by scaling
- Since $x_1$ and $x_2$ must be integers, $5x_1 + 3x_2$ must also be an integer
- Any integer that is $\leq 22.5$ is also $\leq \lfloor 22.5 \rfloor = 22$

**Step 3: Round Down (The Magic!)**

$$\lfloor 5 \rfloor x_1 + \lfloor 3 \rfloor x_2 \leq \lfloor 22.5 \rfloor$$
$$5x_1 + 3x_2 \leq 22$$

This is the **Chvátal-Gomory (CG) Cut**!

**Step 4: Verify Its Power**

- LP optimal $(4.5, 0)$: $5(4.5) + 3(0) = 22.5 \not\leq 22$ ✗ **Cut off!**
- Integer solution $(4, 0)$: $5(4) + 3(0) = 20 \leq 22$ ✓ **Preserved!**

This demonstrates the CG cut's key property:
- **Eliminates fractional LP solutions**
- **Preserves all integer-feasible solutions**

#### Why CG Cuts Work: The Mathematical Properties

**Key Property 1: Why Can We Round Down?**

If $x$ is an integer vector and $\lfloor A \rfloor$ is the coefficient matrix with entries rounded down:
$$\lfloor A \rfloor x = \text{integer vector} \times \text{integer vector} = \text{integer}$$

So if an integer solution satisfies $Ax \leq b$, then:
$$\lfloor A \rfloor x \leq \lfloor Ax \rfloor \leq \lfloor b \rfloor$$

**Key Property 2: Why Not Remove Integer Solutions?**

For any integer feasible solution $x^*$ satisfying the original constraints:
1. Start with: $Ax^* \leq b$
2. Multiply by $u \geq 0$: $u^T Ax^* \leq u^T b$
3. Because $x^*$ is integer: $u^T Ax^*$ is an integer
4. Therefore: $\lfloor u^T A \rfloor x^* \leq \lfloor u^T b \rfloor$ (the CG cut is satisfied!)

**Key Property 3: Why Eliminate Fractional Solutions?**

The LP optimal $(4.5, 0)$ satisfies original constraints but:
- Computing: $5(4.5) + 3(0) = 22.5$
- But CG cut requires: $\leq 22$
- Since $22.5 > 22$, the LP optimal is **cut off** ✗

**In Practice:** 
- Iteratively apply multiple CG cuts to progressively eliminate fractional space
- Each cut tightens the LP bounds: Original LP bound $13.5$ → New LP bound $12$ → etc.
- Eventually LP optimal becomes (approximately) integer
- The choice of $u$ significantly affects cut quality
- Finding good multipliers requires heuristics (Gomory's method, MIR cuts, etc.)

### Ralph Gomory and the History of Cutting Planes

Ralph Gomory pioneered the cutting plane approach in the late 1950s—a groundbreaking contribution to integer programming theory. While cutting planes were originally the primary solution method, modern solvers typically combine them with branch-and-bound in a **branch-and-cut** framework.

### Cutting Plane Approach in a Nutshell

**Algorithm:**
1. Solve the LP relaxation of the current problem
2. If the optimal solution is integer-valued:
   - **Found:** The optimal IP solution
3. If the optimal solution is fractional:
   - Generate a **valid inequality** (cutting plane) that:
     - Cuts off the current fractional solution
     - Does not eliminate any integer-feasible solution
   - Add this constraint to the problem
   - Return to Step 1 (resolve the augmented LP)

**Repeat:** Continue until an integer solution is found

### Branch-and-Cut Algorithm

Modern solvers use a hybrid approach called **branch-and-cut**:
- Begin with branch-and-bound
- During the branching process, dynamically add cutting planes to improve bounds
- This combination is more efficient than either approach alone

---

## Implementation of Algorithms Using Gurobi

### Example: A Clothing Company Problem

**Problem Description:**
A clothing company manufactures three types of clothing: shirts, shorts, and pants. The company must rent machinery for each type:
- Shirt machinery: \$200/week
- Shorts machinery: \$150/week
- Pants machinery: \$100/week

**Available Resources (per week):**
- Labor: 150 hours
- Cloth: 160 square yards

**Product Information:**

| Product | Labor (hours) | Cloth (sq yd) | Variable Cost | Selling Price |
|---------|---------------|---------------|---------------|---------------|
| Shirt   | 3             | 4             | \$6           | \$12          |
| Shorts  | 2             | 3             | \$4           | \$8           |
| Pants   | 6             | 4             | \$5           | \$15          |

### Decision Variables

$$x_1 = \text{number of shirts produced per week}$$
$$x_2 = \text{number of shorts produced per week}$$
$$x_3 = \text{number of pants produced per week}$$

$$y_j = \begin{cases} 
1 & \text{if any garment type } j \text{ is manufactured} \\
0 & \text{otherwise}
\end{cases} \text{ for } j = 1, 2, 3$$

**Interpretation:** If $x_j > 0$, then $y_j = 1$. If $x_j = 0$, then $y_j = 0$.

### Objective Function

Maximize weekly profit:
$$\text{Weekly Profit} = \text{(Sales Revenue)} - \text{(Variable Costs)} - \text{(Machinery Rental)}$$

$$Z = (12 - 6)x_1 + (8 - 4)x_2 + (15 - 5)x_3 - 200y_1 - 150y_2 - 100y_3$$
$$Z = 6x_1 + 4x_2 + 10x_3 - 200y_1 - 150y_2 - 100y_3$$

### Complete Integer Programming Formulation

$$\max \quad 6x_1 + 4x_2 + 10x_3 - 200y_1 - 150y_2 - 100y_3$$

**Subject to:**
$$3x_1 + 2x_2 + 6x_3 \leq 150 \quad \text{(labor constraint)}$$
$$4x_1 + 3x_2 + 4x_3 \leq 160 \quad \text{(cloth constraint)}$$
$$x_j \leq M \cdot y_j \quad \text{for } j = 1, 2, 3 \quad \text{(machinery availability)}$$
$$x_j \geq 0, \text{ integer } \quad \text{for } j = 1, 2, 3$$
$$y_j \in \{0, 1\} \quad \text{for } j = 1, 2, 3$$

**Note:** $M$ is a sufficiently large constant that ensures the big-M constraint is non-binding when the machinery is available.

---

## Traveling Salesperson Problem

### Problem Definition

**The Traveling Salesperson Problem (TSP):**
A salesperson must design a route that:
- Visits each of $n$ cities exactly once
- Returns to the starting location
- Minimizes total travel distance (or time/cost)

### Applications of TSP

While the name refers to a salesperson, TSP has numerous practical applications:
- **Routing and Logistics:** Vehicle routing, delivery optimization
- **Manufacturing:** Integrated circuit design (wire routing)
- **Astronomy:** Optimal sequencing of telescope observations
- **Art:** Computational art and design problems

### Computational Complexity

#### Combinatorial Explosion
The number of possible unique tours grows factorially with the number of cities:

| Number of Cities | Number of Unique Tours |
|------------------|------------------------|
| 3                | 1                      |
| 4                | 3                      |
| 5                | 12                     |
| 6                | 60                     |
| 8                | 20,160                 |
| 10               | 1,814,400              |
| 12               | 239,500,800            |
| 20               | 1.2 × 10^18            |

This rapid growth explains why exact algorithms become prohibitive for large instances.

### Historical Context: P&G Contest (1962)

Procter & Gamble offered a \$10,000 contest (equivalent to \$80,000 today) to find the shortest route visiting 33 locations across the United States starting and ending in Chicago, Illinois.

This classic problem highlights the practical importance of TSP.

### Computational Properties

- **Complexity:** TSP is **NP-hard** (Non-deterministic Polynomial-time hard), meaning no known polynomial-time algorithm exists for its exact solution
- **Exact Solutions:** Modern solvers can optimally solve instances with over 30,000 "cities" (including microchip layout problems)
- **Approximate Solutions:** Very large instances (millions of cities) can be solved to within 1% of optimality using heuristics

### Heuristic vs. Exact Approaches

**Heuristic Approaches:**
- Find reasonably good solutions quickly
- No guarantee of solution quality
- Practical for very large problems

**Exact Integer Programming Approach:**
- Guarantees global optimality
- Multiple formulations possible
- We consider a symmetric TSP (distance from $i$ to $j$ equals distance from $j$ to $i$)

---

### Formulating the TSP as an Integer Program (Symmetric Case)

#### Binary Variable Definition

$$x_{ij} = \begin{cases} 
1 & \text{if the arc from city } i \text{ to city } j \text{ is traversed, } i \neq j \\
0 & \text{otherwise}
\end{cases}$$

where $d_{ij}$ is the distance from city $i$ to city $j$ ($i \neq j$).

#### Basic TSP Formulation

$$\min \quad \sum_{i \neq j} d_{ij} x_{ij}$$

**Subject to:**
$$\sum_{j \neq i} x_{ij} = 1 \quad \forall i \quad \text{(exactly one outgoing arc per city)}$$
$$\sum_{i \neq j} x_{ij} = 1 \quad \forall j \quad \text{(exactly one incoming arc per city)}$$
$$x_{ij} \in \{0, 1\} \quad \forall i, j$$

**Note:** The above formulation is incomplete. There is something still missing!

---

### The Subtours Problem

#### Issue: Subtour Constraints

The basic formulation above has a critical flaw: it may permit **subtours**.

**Definition:** A subtour is a tour that connects only a strict subset of cities, satisfying the degree constraints but not visiting all cities.

**Example:** Suppose cities are {1, 2, 3, 4, 5}. A subtour might be:
- Subtour 1: 1 → 2 → 3 → 1
- Subtour 2: 4 → 5 → 4

Both subtours satisfy the degree constraints but don't form a valid complete tour.

### Subtour-Breaking Constraints

#### Key Observations

Let $S$ be any proper subset of nodes with $|S| \geq 3$ (e.g., $S = \{2, 3, 4, 7, 9\}$):

1. A subtour consisting entirely of nodes in $S$ contains exactly $|S|$ arcs
2. A valid tour through the complete network can contain **at most** $|S| - 1$ arcs with both endpoints in $S$

#### Subtour Elimination Constraint

To forbid any subtour on subset $S$, add:
$$\sum_{i \in S, j \in S, i \neq j} x_{ij} \leq |S| - 1$$

This constraint sums all possible arcs within subset $S$ (not just the active ones) and ensures that no subtour can form within $S$.

#### Complete TSP Formulation

$$\min \quad \sum_{i \neq j} d_{ij} x_{ij}$$

**Subject to:**
$$\sum_{j \neq i} x_{ij} = 1 \quad \forall i$$
$$\sum_{i \neq j} x_{ij} = 1 \quad \forall j$$
$$\sum_{i \in S, j \in S, i \neq j} x_{ij} \leq |S| - 1 \quad \forall S \subseteq N, |S| \geq 3$$
$$x_{ij} \in \{0, 1\} \quad \forall i, j$$

---

### The Subtour Constraint Explosion Problem

#### Scalability Issue

For small networks (6-10 cities), adding all subtour constraints is computationally feasible. However, the number of possible subsets grows exponentially:

$$\text{Number of proper subsets} = 2^n - 2$$

For $n = 20$ cities: over 1 million subsets!  
For $n = 30$ cities: over 1 billion subsets!

**Conclusion:** It is prohibitive to explicitly add all subtour constraints for realistically-sized problems.

#### Solution Strategies

**Two practical approaches:**

1. **Partial Constraint Approach:**
   - Include only some critical subtour constraints
   - Solve the resulting IP
   - If subtours appear, add new constraints and resolve

2. **Dynamic/Lazy Constraint Generation:**
   - Formulate IP with only degree constraints (and possibly a few subtour constraints)
   - Solve the IP
   - Check if solution contains subtours
   - If yes, add the violated subtour constraint and resolve
   - Repeat until a valid tour is found

This iterative approach avoids the need to pre-specify all constraints while still guaranteeing optimality.

---

## Alternative Formulations for TSP

While the subtour-based formulation above is one approach, other formulations exist:

### Miller-Tucker-Zemlin (MTZ) Formulation

The MTZ formulation avoids the exponential growth of subtour constraints by introducing **continuous variables** $u_i$ representing the "order" in which city $i$ is visited:

$$u_i - u_j + (n-1)x_{ij} \leq n - 2 \quad \forall i, j$$

**Advantage:** Only $O(n^2)$ additional constraints instead of exponentially many  
**Disadvantage:** Often provides weaker LP bounds

### Comparison of Approaches

Different formulations offer trade-offs:
- **Subtour-based:** Strong bounds but exponential constraints
- **MTZ-based:** Polynomial constraints but weaker bounds
- **Hybrid (branch-and-cut):** Add subtour constraints dynamically during branch-and-bound

---

## Key Takeaways

### Branch and Bound
- Systematic way to solve IPs by relaxing integrality and recursively partitioning
- Practical performance but no theoretical worst-case guarantees
- Fundamental technique used in modern solvers

### Cutting Planes
- Add valid inequalities to strengthen the LP relaxation
- Chvátal-Gomory cuts provide a general framework
- Modern solvers use branch-and-cut for efficiency

### TSP and Formulation Strength
- Multiple formulations of the same problem can have very different LP relaxations
- A tighter formulation requires less branching
- Trade-offs exist between formulation size and bound quality

### Practical Solution
- Commercial solvers like Gurobi implement sophisticated versions of these algorithms
- Understanding the underlying methods helps in:
  - Formulating problems effectively
  - Tuning solver parameters
  - Recognizing when a problem is inherently difficult

---

## References and Resources

- **TSP Game:** https://algorithms.discrete.ma.tum.de/graph-games/tsp-game/index_en.html
  - Interactive tool to visualize TSP heuristics (choose Germany map, 25 cities)

- **Ralph Gomory's Contributions:** https://www.youtube.com/watch?v=OesVp4Hlqps
  - Historical perspective on the development of cutting plane methods

---

## Appendix: Discussion Notes (Q&A Remarks)

### 1) Why do we branch as $x_f \le 5$ and $x_f \ge 6$?
- If the LP solution has a fractional variable, e.g., $x_f=5.6$, then integer feasibility requires either $x_f \le \lfloor 5.6 \rfloor = 5$ or $x_f \ge \lceil 5.6 \rceil = 6$.
- These two branches partition all integer possibilities for that variable with no overlap and no omission.

### 2) Why can a branch be pruned even if LP optimum is fractional?
- For maximization, each node LP optimum is an **upper bound** on any integer solution in that node.
- If node bound $\le$ current best integer objective (incumbent), that node cannot improve the incumbent and is safely pruned.

### 3) Where does the “current best integer solution” come from?
- It does **not** come only from the root LP.
- It can come from:
  1. A node whose LP optimum is already integer,
  2. Solver heuristics (feasible MIP solution found early),
  3. Any explored branch that yields a feasible integer solution.

### 4) Are we “back to LP again” after starting from IP?
- Yes, intentionally.
- Branch-and-bound solves many LP relaxations:
  - root LP relaxation,
  - node LP relaxations after branching,
  - plus cuts and re-optimization.
- Goal remains IP optimality; LPs are the bounding/search engine.

### 5) CG cuts: what does “multiply by a vector $u$” mean?
- We take a nonnegative linear combination of constraints with multipliers $u$.
- This produces: $u^T A x \le u^T b$.
- With integer variables, we derive a valid cut by rounding down (in standard CG/Gomory derivations):
  $$\lfloor u^T A \rfloor x \le \lfloor u^T b \rfloor$$
- The cut should remove the current fractional LP solution while keeping all integer-feasible solutions.

### 6) How is $u$ chosen in practice?
- In classroom demos: often hand-picked to illustrate the idea.
- In solvers: generated automatically from LP basis/tableau and separation routines (Gomory, MIR, flow-cover, etc.).
- Strong cuts come from good $u$; weak $u$ may produce little improvement.

### 7) Big-M in this model: why needed?
- Linking constraints: $x_i \le M_i y_i$ with $y_i\in\{0,1\}$.
  - If $y_i=0$ (machine not rented): forces $x_i=0$.
  - If $y_i=1$ (machine rented): allows production up to $M_i$.
- Choose $M_i$ tight (not huge) for better performance.
- A practical choice:
  $$M_i=\min\left(\left\lfloor\frac{L}{a_i}\right\rfloor,\left\lfloor\frac{C}{b_i}\right\rfloor\right)$$

### 8) What is root relaxation?
- The LP relaxation at the **root node** of the branch-and-bound tree.
- It provides the initial global bound and helps indicate how hard the MIP may be.
- Modern solvers also generate cuts and heuristic solutions at root before deeper branching.

---

*Notes compiled from IEOR E4004 lecture by Yaren Bilge Kaya, PhD*
