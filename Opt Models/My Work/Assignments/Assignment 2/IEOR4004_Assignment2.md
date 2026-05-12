# IEOR 4004 Assignment 2 
Luomeng Zhou  (uni:lz3064)

---

## Question 1

### Part 1: Formulate and solve a linear program to minimize the dispatcher labor costs

**Problem Summary:** Three airports in New York City operate 24 hours a day, 7 days a week. The total number of air traffic controllers needed during each 4-hour interval is given in Table 1. Controllers work either 8-hour or 12-hour shifts. 8-hour shifts start every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00). 12-hour shifts start only at 00:00, 08:00, 12:00, or 20:00. Cost: $93/hour (8h shifts), $100.8/hour (12h shifts).

**Table 1 – Demand**

| Time Interval | Controllers Needed |
|---------------|---------------------|
| 12 a.m. to 4 a.m. | 8 |
| 4 a.m. to 8 a.m.  | 10 |
| 8 a.m. to 12 p.m. | 16 |
| 12 p.m. to 4 p.m. | 21 |
| 4 p.m. to 8 p.m.  | 18 |
| 8 p.m. to 12 a.m. | 12 |

**Decision Variables:** `x_{8,t}` = employees starting 8-hour shift at time t; `x_{12,t}` = employees starting 12-hour shift at time t. All >= 0.

**Objective:** `min Z = 744 * sum x_{8,t} + 1209.6 * sum x_{12,t}` (744 = 8×93, 1209.6 = 12×100.8)

**Constraints (demand per period):**
- T1 00–04: `x_{8,00} + x_{8,20} + x_{12,00} + x_{12,20} >= 8`
- T2 04–08: `x_{8,00} + x_{8,04} + x_{12,00} + x_{12,20} >= 10`
- T3 08–12: `x_{8,04} + x_{8,08} + x_{12,00} + x_{12,08} >= 16`
- T4 12–16: `x_{8,08} + x_{8,12} + x_{12,08} + x_{12,12} >= 21`
- T5 16–20: `x_{8,12} + x_{8,16} + x_{12,08} + x_{12,12} >= 18`
- T6 20–24: `x_{8,16} + x_{8,20} + x_{12,12} + x_{12,20} >= 12`

**Optimal Solution (Gurobi):**
- 8h shifts: 8@00, 2@04, 14@08, 6@12, 11@16, 0@20
- 12h shifts: 0@00, 0@08, 1@12, 0@20
- **Optimal daily cost:** $31,713.60

```
8-hour shift starting at 00:00: 8.00 employees
8-hour shift starting at 04:00: 2.00 employees
8-hour shift starting at 08:00: 14.00 employees
8-hour shift starting at 12:00: 6.00 employees
8-hour shift starting at 16:00: 11.00 employees
8-hour shift starting at 20:00: 0.00 employees
12-hour shift starting at 00:00: 0.00 employees
12-hour shift starting at 08:00: 0.00 employees
12-hour shift starting at 12:00: 1.00 employees
12-hour shift starting at 20:00: 0.00 employees
Minimum total cost: $31713.60
```

---

### Part 2: At most one-third of controllers can work 12-hour shifts

Add constraint: `sum x_{12,t} <= (1/3) * (sum x_{8,t} + sum x_{12,t})`. With the same optimal solution as Part 1, 12h staff = 1, total = 42, so 1/42 < 1/3. The constraint is non-binding; optimal solution and cost remain the same: **$31,713.60**.

---

### Part 3: Staff the three airports separately

**Model changes:** (1) Index variables and constraints by airport: `x_{8,t}^{(i)}`, `x_{12,t}^{(i)}` for airport i = 1, 2, 3; (2) Each airport has its own demand constraints per time period; (3) Objective = sum of costs across all three airports (still minimize total cost).

**Additional data needed:** Table 1 gives aggregate demand. To staff separately, we need **demand per airport per time period** `d_{i,t}` (e.g., Airport A needs 3 in 00–04, Airport B needs 2, Airport C needs 3, summing to 8). Without this split, the separate-staffing model cannot be implemented.

---

## Question 2

### Primal Problem

```
max Z = 10*x_1 + 14*x_2 + 20*x_3
s.t. 2*x_1 + 3*x_2 + 4*x_3 <= 220
     4*x_1 + 2*x_2 - x_3   <= 385
     x_1 + 4*x_3           <= 160
     x_1, x_2, x_3 >= 0
```

### Dual Problem

```
min W = 220*y_1 + 385*y_2 + 160*y_3
s.t. 2*y_1 + 4*y_2 + y_3   >= 10
     3*y_1 + 2*y_2         >= 14
     4*y_1 - y_2 + 4*y_3   >= 20
     y_1, y_2, y_3 >= 0
```

---

## Question 3

### Part 1: Sensible cutting patterns (leftover < 3 ft)

| # | Pattern | 3-ft | 4-ft | 5-ft | Used (ft) | Waste (ft) |
|---|---------|------|------|------|-----------|------------|
| 1 | 3×3 | 3 | 0 | 0 | 9 | 1 |
| 2 | 2×4 | 0 | 2 | 0 | 8 | 2 |
| 3 | 2×5 | 0 | 0 | 2 | 10 | 0 |
| 4 | 2×3+1×4 | 2 | 1 | 0 | 10 | 0 |
| 5 | 1×3+1×5 | 1 | 0 | 1 | 8 | 2 |
| 6 | 1×4+1×5 | 0 | 1 | 1 | 9 | 1 |

Note: 1×3+1×4 gives leftover 3 ft, excluded by “leftover < 3 ft”.

### Part 2: Integer linear program

**Variables:** `x_j` = number of 10-ft boards cut with pattern j, j = 1,...,6.

**Objective:** `min sum_{j=1}^{6} x_j`

**Constraints:** For each length (3-ft, 4-ft, 5-ft), total yield >= demand:
- 3-ft: `3*x_1 + 2*x_4 + x_5 >= 90`
- 4-ft: `2*x_2 + x_4 + x_6 >= 60`
- 5-ft: `2*x_3 + x_5 + x_6 >= 60`
- `x_j` in Z_+ (nonnegative integers)

**Optimal solution (Gurobi):** Minimum **83** boards. Pattern 2: 8, Pattern 3: 30, Pattern 4: 45; others 0.

```
Minimum number of boards: 83
  Pattern 2 (2×4): 8
  Pattern 3 (2×5): 30
  Pattern 4 (2×3+1×4): 45
```

### Part 3: Minimize scrap (same board count)

**Algebraic model:** Let N* = 83 be the minimum boards from Part 2. We solve a second ILP: `min sum_j w_j * x_j` s.t. `sum_j x_j = N*`, `sum_j a_{jk} * x_j >= d_k` for each length k, and x_j in Z_+, where w_j = waste (ft) for pattern j, a_{jk} = yield of length k from pattern j, d_k = demand for length k. This keeps the board count at N* while minimizing total scrap.

**Optimal solution:** Minimum scrap **14 ft**. Pattern 2: 7, Pattern 3: 30, Pattern 4: 46.

```
Minimum scrap (ft): 14
  Pattern 2 (2×4): 7
  Pattern 3 (2×5): 30
  Pattern 4 (2×3+1×4): 46
```

---

## Appendix: Code

### staff_scheduling.py (Question 1)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gurobipy import GRB, Model

demand = {
    "00:00-04:00": 8,
    "04:00-08:00": 10,
    "08:00-12:00": 16,
    "12:00-16:00": 21,
    "16:00-20:00": 18,
    "20:00-24:00": 12,
}

cost_8h_per_hour = 93
cost_12h_per_hour = 100.8
cost_8h_shift = cost_8h_per_hour * 8
cost_12h_shift = cost_12h_per_hour * 12

start_times_8h = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
start_times_12h = ["00:00", "08:00", "12:00", "20:00"]

def solve_staffing_problem(apply_12h_shift_constraint=False):
    model = Model("Staffing_Problem")
    x_8h = model.addVars(start_times_8h, name="X_8h", vtype=GRB.CONTINUOUS)
    x_12h = model.addVars(start_times_12h, name="X_12h", vtype=GRB.CONTINUOUS)

    model.setObjective(
        sum(x_8h[t] * cost_8h_shift for t in start_times_8h) +
        sum(x_12h[t] * cost_12h_shift for t in start_times_12h),
        GRB.MINIMIZE
    )

    model.addConstr(x_8h["00:00"] + x_8h["20:00"] + x_12h["00:00"] + x_12h["20:00"] >= demand["00:00-04:00"], "Demand_T1")
    model.addConstr(x_8h["00:00"] + x_8h["04:00"] + x_12h["00:00"] + x_12h["20:00"] >= demand["04:00-08:00"], "Demand_T2")
    model.addConstr(x_8h["04:00"] + x_8h["08:00"] + x_12h["00:00"] + x_12h["08:00"] >= demand["08:00-12:00"], "Demand_T3")
    model.addConstr(x_8h["08:00"] + x_8h["12:00"] + x_12h["08:00"] + x_12h["12:00"] >= demand["12:00-16:00"], "Demand_T4")
    model.addConstr(x_8h["12:00"] + x_8h["16:00"] + x_12h["08:00"] + x_12h["12:00"] >= demand["16:00-20:00"], "Demand_T5")
    model.addConstr(x_8h["16:00"] + x_8h["20:00"] + x_12h["12:00"] + x_12h["20:00"] >= demand["20:00-24:00"], "Demand_T6")

    if apply_12h_shift_constraint:
        total_12h_staff = sum(x_12h[t] for t in start_times_12h)
        total_staff = sum(x_8h[t] for t in start_times_8h) + total_12h_staff
        model.addConstr(total_12h_staff <= (1/3) * total_staff, "12h_Shift_Limit")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        for t in start_times_8h:
            print(f"8-hour shift starting at {t}: {x_8h[t].X:.2f} employees")
        for t in start_times_12h:
            print(f"12-hour shift starting at {t}: {x_12h[t].X:.2f} employees")
        print(f"Minimum total cost: ${model.ObjVal:.2f}")

if __name__ == "__main__":
    solve_staffing_problem(apply_12h_shift_constraint=False)
    solve_staffing_problem(apply_12h_shift_constraint=True)
```

### lumberyard_cutting.py (Question 3)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IEOR 4004 Assignment 2 - Question 3: Lumberyard Cutting"""

from gurobipy import GRB, Model, quicksum

patterns = [
    {"name": "3×3", "yield": (3, 0, 0), "waste": 1},
    {"name": "2×4", "yield": (0, 2, 0), "waste": 2},
    {"name": "2×5", "yield": (0, 0, 2), "waste": 0},
    {"name": "2×3+1×4", "yield": (2, 1, 0), "waste": 0},
    {"name": "1×3+1×5", "yield": (1, 0, 1), "waste": 2},
    {"name": "1×4+1×5", "yield": (0, 1, 1), "waste": 1},
]
demand = {0: 90, 1: 60, 2: 60}

def solve_min_boards():
    m = Model("lumberyard_min_boards")
    n = len(patterns)
    x = m.addVars(n, name="x", vtype=GRB.INTEGER, lb=0)
    m.setObjective(quicksum(x[j] for j in range(n)), GRB.MINIMIZE)
    for k in range(3):
        m.addConstr(quicksum(patterns[j]["yield"][k] * x[j] for j in range(n)) >= demand[k])
    m.optimize()
    return m, x

def solve_min_waste(min_boards):
    m = Model("lumberyard_min_waste")
    n = len(patterns)
    x = m.addVars(n, name="x", vtype=GRB.INTEGER, lb=0)
    m.setObjective(quicksum(patterns[j]["waste"] * x[j] for j in range(n)), GRB.MINIMIZE)
    m.addConstr(quicksum(x[j] for j in range(n)) == min_boards)
    for k in range(3):
        m.addConstr(quicksum(patterns[j]["yield"][k] * x[j] for j in range(n)) >= demand[k])
    m.optimize()
    return m, x

if __name__ == "__main__":
    model1, x1 = solve_min_boards()
    if model1.status == GRB.OPTIMAL:
        min_boards = int(model1.ObjVal)
        print(f"Minimum number of boards: {min_boards}")
        for j in range(len(patterns)):
            if x1[j].X > 0:
                print(f"  Pattern {j+1} ({patterns[j]['name']}): {x1[j].X:.0f}")
        model2, x2 = solve_min_waste(min_boards)
        if model2.status == GRB.OPTIMAL:
            print(f"Minimum scrap (ft): {model2.ObjVal:.0f}")
            for j in range(len(patterns)):
                if x2[j].X > 0:
                    print(f"  Pattern {j+1} ({patterns[j]['name']}): {x2[j].X:.0f}")
```
