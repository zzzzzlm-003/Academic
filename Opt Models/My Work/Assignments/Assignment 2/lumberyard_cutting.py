#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEOR 4004 Assignment 2 - Question 3: Lumberyard Cutting Problem
10-ft boards cut into 3-ft, 4-ft, 5-ft boards. Leftover < 3 ft.
Demand: 90 x 3-ft, 60 x 4-ft, 60 x 5-ft.
"""

from gurobipy import GRB, Model, quicksum

# Sensible cutting patterns (leftover < 3 ft)
# (3ft, 4ft, 5ft) yield per pattern, waste (ft)
patterns = [
    {"name": "3×3", "yield": (3, 0, 0), "waste": 1},
    {"name": "2×4", "yield": (0, 2, 0), "waste": 2},
    {"name": "2×5", "yield": (0, 0, 2), "waste": 0},
    {"name": "2×3+1×4", "yield": (2, 1, 0), "waste": 0},
    {"name": "1×3+1×5", "yield": (1, 0, 1), "waste": 2},
    {"name": "1×4+1×5", "yield": (0, 1, 1), "waste": 1},
]

demand = {0: 90, 1: 60, 2: 60}  # 3ft, 4ft, 5ft


def solve_min_boards():
    """Part 2: Minimize number of 10-ft boards."""
    m = Model("lumberyard_min_boards")
    n = len(patterns)
    x = m.addVars(n, name="x", vtype=GRB.INTEGER, lb=0)

    m.setObjective(quicksum(x[j] for j in range(n)), GRB.MINIMIZE)

    for k in range(3):  # 3ft, 4ft, 5ft
        m.addConstr(
            quicksum(patterns[j]["yield"][k] * x[j] for j in range(n)) >= demand[k],
            name=f"demand_{k}",
        )

    m.optimize()
    return m, x


def solve_min_waste(min_boards):
    """Part 3: Minimize scrap, subject to using min_boards."""
    m = Model("lumberyard_min_waste")
    n = len(patterns)
    x = m.addVars(n, name="x", vtype=GRB.INTEGER, lb=0)

    m.setObjective(
        quicksum(patterns[j]["waste"] * x[j] for j in range(n)),
        GRB.MINIMIZE,
    )

    m.addConstr(quicksum(x[j] for j in range(n)) == min_boards, name="board_count")
    for k in range(3):
        m.addConstr(
            quicksum(patterns[j]["yield"][k] * x[j] for j in range(n)) >= demand[k],
            name=f"demand_{k}",
        )

    m.optimize()
    return m, x


if __name__ == "__main__":
    print("=== Part 2: Minimize number of 10-ft boards ===\n")
    model1, x1 = solve_min_boards()
    if model1.status == GRB.OPTIMAL:
        print(f"Minimum number of boards: {model1.ObjVal:.0f}\n")
        for j in range(len(patterns)):
            if x1[j].X > 0:
                print(f"  Pattern {j+1} ({patterns[j]['name']}): {x1[j].X:.0f}")
        min_boards = int(model1.ObjVal)
    else:
        print("No optimal solution.")
        min_boards = None

    if min_boards is not None:
        print("\n=== Part 3: Minimize scrap (keeping min boards) ===\n")
        model2, x2 = solve_min_waste(min_boards)
        if model2.status == GRB.OPTIMAL:
            print(f"Minimum scrap (ft): {model2.ObjVal:.0f}\n")
            for j in range(len(patterns)):
                if x2[j].X > 0:
                    print(f"  Pattern {j+1} ({patterns[j]['name']}): {x2[j].X:.0f}")
