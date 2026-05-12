## IEOR E4004 – Assignment 1 Luomeng Zhou (UNI:lz3064)

Please discuss the following statements and briefly explain your opinion on them in a clear and concise way.

---

### Question 1

#### Statement 1: If a linear programming problem has a feasible solution, it is guaranteed to have an optimal solution.

I think this statement is not correct. Having a feasible solution alone does not guarantee the existence of an optimal solution, because we also need the objective function to be bounded and actually attained on the feasible region. Intuitively, the constraints may define a nonempty feasible set, but the objective value may keep improving without bound or only approach its best value at some limit point that is never reached, so no true optimum exists.

---

#### Statement 2: If the feasible region of a linear programming problem is unbounded, it means the objective function can be improved indefinitely.

I also think this statement is wrong. An unbounded feasible region does not automatically imply that the objective function can be improved indefinitely. For instance, the feasible set might extend to infinity in the first quadrant, but the objective could be “pulling” the solution toward one of the axes, so the best value is still finite and achieved somewhere, instead of improving without limit.

---

#### Statement 3: If an LP is feasible, then it has an optimal solution that is a corner point.

At first glance this statement looks correct, because we often say that “an optimal solution occurs at a corner point.” However, if we are not even guaranteed to have an optimal solution at all (as in Statement 1), then this statement is actually stronger and therefore cannot always be true. A more accurate version would be: if an LP has an optimal solution, then there exists at least one optimal solution that is a corner point.

---

#### Statement 4: The graphical solution method can only be used to solve linear programs with two decision variables.

I basically agree with this statement. The graphical method relies on drawing the feasible region in a two-dimensional coordinate system and moving level sets of the objective function on the picture, so it is only practical for problems with two decision variables. When there are more variables, we usually need to use the simplex method or other numerical algorithms instead of solving it purely by drawing graphs.

---

## Question 2

uni:yq2439,lh3345,ml5338

After further reflection and discussion, my refined answers are:

1. **Statement 1**The statement is **not always true**. Feasibility only means that the feasible region is nonempty; it does **not** rule out the possibility that the objective can be improved indefinitely. For example, consider$ \max x \quad \text{s.t. } x \ge 0. $The feasible region is nonempty and unbounded, but the objective is also unbounded above, so there is no optimal solution. A more accurate version would be: *If an LP has a nonempty feasible region and the objective is bounded over that region, then it has at least one optimal solution.*
2. **Statement 2**This statement is **false**. An unbounded feasible region only means that the set of feasible points is not bounded in some direction; it does **not** guarantee that the objective can be made arbitrarily large (for maximization) or arbitrarily small (for minimization). For example, consider$ \min x + y \quad \text{s.t. } x \ge 0,\; y \ge 0,\; x + y \ge 1. $The feasible region is unbounded (it extends infinitely in the first quadrant), but the minimum value of the objective is $1$, attained at points such as $(1,0)$ and $(0,1)$. Here the objective is bounded and there is an optimal solution despite the unbounded feasible region.
3. **Statement 3**As written, the statement is **not always true**. Feasibility alone does not ensure the existence of an optimal solution (the objective might be unbounded), so we cannot claim that there *is* an optimal corner point just from feasibility. The fundamental theorem of linear programming says: *If an LP has an optimal solution and its feasible region is a polyhedron, then at least one optimal solution can be found at an extreme point (corner point) of the feasible region.* Thus, the missing condition is the existence of an optimal solution (and the usual polyhedral structure), not just feasibility.
4. **Statement 4**
   In practice, this statement is **essentially true**. The classical graphical method is only practical for LPs with two decision variables, because we can easily visualize the feasible region and objective in a 2D plot. For more variables, direct visualization becomes difficult or impossible, and we switch to algebraic or algorithmic methods such as the simplex method, interior-point methods, or other solvers. (In theory, one might attempt 3D plots for three variables, but this is rarely used beyond very simple illustrative examples.)

## Question 3

### (1) Description of decision variables, objective, and constraints

- **Decision variables**For each supplier $j \in \{1,2,3\}$ and each resource $k$ in$\{\text{Toilet Paper}, \text{Liquid Soap}, \text{Detergent}, \text{Cloths}, \text{Toothpaste}, \text{Toothbrushes}, \text{Sanitary Pads}, \text{Shampoo}\}$,let $x_{jk} \ge 0$ denote the amount (in appropriate units) of resource $k$ purchased from supplier $j$.
- **Objective function**Minimize the total monthly purchasing cost:
  $ \min \sum_{j=1}^3 \sum_{k} c_{jk} x_{jk} $
  where $c_{jk}$ is the unit cost of resource $k$ from supplier $j$, given in Table 1.
- **Constraints**

  - **Supply capacity constraints**: For each supplier $j$ and resource $k$,
    $ 0 \le x_{jk} \le s_{jk} $
    where $s_{jk}$ is the maximum monthly supply capacity from Table 2.
  - **Minimum demand constraints**: For each resource $k$,
    $ \sum_{j=1}^3 x_{jk} \ge d_k $
    where $d_k$ is the minimum required quantity from Table 3.
  - **Budget constraint**: The total monthly spending cannot exceed \$2000,
    $ \sum_{j=1}^3 \sum_{k} c_{jk} x_{jk} \le 2000 $

### (2) Algebraic LP formulation

- **Decision variables**: $x_{jk} \ge 0$ for all $j \in S, k \in R$.
- **Objective**:$ \min \; Z = \sum_{j \in S} \sum_{k \in R} c_{jk} x_{jk} $
- **Constraints**:
  $ \sum_{j \in S} x_{jk} \ge d_k, \quad \forall k \in R \qquad \text{(minimum demand)} $

  $ 0 \le x_{jk} \le s_{jk}, \quad \forall j \in S, k \in R \qquad \text{(supply capacity)} $

  $ \sum_{j \in S} \sum_{k \in R} c_{jk} x_{jk} \le 2000 \qquad \text{(budget)} $

### (3) Gurobi implementation (Python)

Below is a complete Gurobi model in Python that implements this LP using the data from Tables 1–3.

```python
from gurobipy import Model, GRB, quicksum

# Sets
suppliers = ["S1", "S2", "S3"]
resources = [
    "toilet_paper",
    "liquid_soap",
    "detergent",
    "cloths",
    "toothpaste",
    "toothbrushes",
    "pads",
    "shampoo",
]

# Cost per unit c_{jk} (from Table 1)
cost = {
    ("S1", "toilet_paper"): 0.80,
    ("S1", "liquid_soap"): 6.40,
    ("S1", "detergent"): 6.80,
    ("S1", "cloths"): 10.00,
    ("S1", "toothpaste"): 2.60,
    ("S1", "toothbrushes"): 0.80,
    ("S1", "pads"): 0.20,
    ("S1", "shampoo"): 2.30,

    ("S2", "toilet_paper"): 0.95,
    ("S2", "liquid_soap"): 3.98,
    ("S2", "detergent"): 4.60,
    ("S2", "cloths"): 11.00,
    ("S2", "toothpaste"): 3.00,
    ("S2", "toothbrushes"): 0.85,
    ("S2", "pads"): 0.18,
    ("S2", "shampoo"): 1.20,

    ("S3", "toilet_paper"): 0.84,
    ("S3", "liquid_soap"): 5.50,
    ("S3", "detergent"): 7.50,
    ("S3", "cloths"): 10.50,
    ("S3", "toothpaste"): 2.80,
    ("S3", "toothbrushes"): 0.82,
    ("S3", "pads"): 0.15,
    ("S3", "shampoo"): 3.00,
}

# Maximum supply capacity s_{jk} (from Table 2)
capacity = {
    ("S1", "toilet_paper"): 150,
    ("S1", "liquid_soap"): 25,
    ("S1", "detergent"): 20,
    ("S1", "cloths"): 10,
    ("S1", "toothpaste"): 50,
    ("S1", "toothbrushes"): 50,
    ("S1", "pads"): 150,
    ("S1", "shampoo"): 20,

    ("S2", "toilet_paper"): 100,
    ("S2", "liquid_soap"): 15,
    ("S2", "detergent"): 10,
    ("S2", "cloths"): 10,
    ("S2", "toothpaste"): 60,
    ("S2", "toothbrushes"): 60,
    ("S2", "pads"): 100,
    ("S2", "shampoo"): 20,

    ("S3", "toilet_paper"): 70,
    ("S3", "liquid_soap"): 30,
    ("S3", "detergent"): 15,
    ("S3", "cloths"): 15,
    ("S3", "toothpaste"): 30,
    ("S3", "toothbrushes"): 30,
    ("S3", "pads"): 100,
    ("S3", "shampoo"): 30,
}

# Minimum required quantities d_k (from Table 3)
demand = {
    "toilet_paper": 200,
    "liquid_soap": 40,
    "detergent": 30,
    "cloths": 20,
    "toothpaste": 100,
    "toothbrushes": 100,
    "pads": 300,
    "shampoo": 40,
}

# Create model
m = Model("shelter_procurement")

# Decision variables: x[j, r] >= 0
x = m.addVars(suppliers, resources, name="x", lb=0.0)

# Set upper bounds according to capacity
for (j, r), cap in capacity.items():
    x[j, r].ub = cap

# Objective: minimize total cost
m.setObjective(
    quicksum(cost[j, r] * x[j, r] for j in suppliers for r in resources),
    GRB.MINIMIZE,
)

# Demand constraints: sum_j x_{jk} >= d_k
for r in resources:
    m.addConstr(
        quicksum(x[j, r] for j in suppliers) >= demand[r],
        name=f"demand_{r}",
    )

# Budget constraint: total cost <= 2000
m.addConstr(
    quicksum(cost[j, r] * x[j, r] for j in suppliers for r in resources) <= 2000,
    name="budget",
)

# Optimize
m.optimize()

# Print solution
if m.status == GRB.OPTIMAL:
    print(f"Optimal total cost: {m.ObjVal:.2f}")
    for j in suppliers:
        for r in resources:
            if x[j, r].X > 1e-6:
                print(f"{j}, {r}: {x[j, r].X:.2f}")
else:
    print("No optimal solution found.")
```

### (4) Solver output and solution

Running the above model in Gurobi yields the following console output:

```text
Optimal total cost: 1224.80
S1, toilet_paper: 150.00
S1, detergent: 20.00
S1, cloths: 10.00
S1, toothpaste: 50.00
S1, toothbrushes: 50.00
S1, pads: 100.00
S1, shampoo: 20.00
S2, liquid_soap: 15.00
S2, detergent: 10.00
S2, toothpaste: 20.00
S2, toothbrushes: 20.00
S2, pads: 100.00
S2, shampoo: 20.00
S3, toilet_paper: 50.00
S3, liquid_soap: 25.00
S3, cloths: 10.00
S3, toothpaste: 30.00
S3, toothbrushes: 30.00
S3, pads: 100.00
```

In words, the minimum total cost is **\$1224.80**, achieved by purchasing:

- From supplier 1: 150 toilet paper rolls, 20 liters of detergent, 10 m² of cloths, 50 toothpaste tubes, 50 toothbrushes, 100 packs of pads, and 20 liters of shampoo.
- From supplier 2: 15 liters of liquid soap, 10 liters of detergent, 20 toothpaste tubes, 20 toothbrushes, 100 packs of pads, and 20 liters of shampoo.
- From supplier 3: 50 toilet paper rolls, 25 liters of liquid soap, 10 m² of cloths, 30 toothpaste tubes, 30 toothbrushes, and 100 packs of pads.

### 4. Explain the Solution

By comparing the unit price of each item across the three suppliers, we first choose the cheapest supplier and purchase up to its maximum supply capacity. If this still does not meet the minimum requirement for that item, we then buy the remaining required quantity from the second-cheapest supplier (and continue similarly if needed).
