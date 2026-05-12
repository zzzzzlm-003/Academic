"""
Q3 — Traffic Congestion QP
Reads q3_congestion.lp and solves with Gurobi, prints flows + cost.
"""
import gurobipy as gp
from gurobipy import GRB
import os

LP_FILE = os.path.join(os.path.dirname(__file__), "q3_congestion.lp")

m = gp.read(LP_FILE)
m.optimize()

if m.Status == GRB.OPTIMAL:
    print("\n=== Optimal flows ===")
    for v in m.getVars():
        print(f"  {v.VarName} = {v.X:.4f}")
    print(f"\nTotal congestion cost = {m.ObjVal:.4f}")
else:
    print(f"Solver did not return OPTIMAL (status={m.Status}).")
