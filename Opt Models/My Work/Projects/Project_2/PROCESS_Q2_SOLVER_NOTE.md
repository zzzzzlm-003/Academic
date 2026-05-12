# Q2 Solver Note (Gurobi license issue)

## What happened
When solving Q2 with `gurobipy`, the run failed with:

`GurobiError: Model too large for size-limited license`

This is **not** a modeling mistake. It is a **license limitation** of the restricted/size-limited Gurobi license, which caps the number of variables/constraints that can be solved.

## What we did instead
We kept the **same lecture-aligned integer programming formulation**:

- Binary decision variable \(x_{i,j,d}\): 1 if on date \(d\), team \(i\) plays **home** vs team \(j\)
- Linear equality constraints (e)–(h) exactly as in the PDF prompt
- Objective set to 0 (feasibility problem)

But we switched the solver backend to an **open-source MILP solver**:

- **PuLP + CBC**

So the schedule produced is still a feasible solution to the **same IP**; only the solver changed due to the Gurobi license size restriction.

