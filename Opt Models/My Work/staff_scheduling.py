#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gurobipy import GRB, Model

# --- 数据定义 ---
# 需求人数
demand = {
    "00:00-04:00": 8,
    "04:00-08:00": 10,
    "08:00-12:00": 16,
    "12:00-16:00": 21,
    "16:00-20:00": 18,
    "20:00-24:00": 12,
}

# 班次成本
cost_8h_per_hour = 93
cost_12h_per_hour = 100.8

cost_8h_shift = cost_8h_per_hour * 8
cost_12h_shift = cost_12h_per_hour * 12

# 班次启动时间
# 8 小时班可以从任意整点开始，这里简化为每4小时一个班次，与需求时间段对齐
start_times_8h = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
# 12 小时班根据作业要求
start_times_12h = ["00:00", "08:00", "12:00", "20:00"]

# 时间段索引和对应的开始时间
time_periods = {
    "00:00-04:00": {"start": "00:00", "end": "04:00"},
    "04:00-08:00": {"start": "04:00", "end": "08:00"},
    "08:00-12:00": {"start": "08:00", "end": "12:00"},
    "12:00-16:00": {"start": "12:00", "end": "16:00"},
    "16:00-20:00": {"start": "16:00", "end": "20:00"},
    "20:00-24:00": {"start": "20:00", "end": "00:00"}, # End 00:00 for next day
}

def solve_staffing_problem(apply_12h_shift_constraint=False):
    # 创建模型
    model = Model("Staffing_Problem")

    # --- 决策变量 ---
    # 8 小时班次的员工数量
    x_8h = model.addVars(start_times_8h, name="X_8h", vtype=GRB.CONTINUOUS)
    # 12 小时班次的员工数量
    x_12h = model.addVars(start_times_12h, name="X_12h", vtype=GRB.CONTINUOUS)

    # --- 目标函数 ---
    # 最小化总成本
    model.setObjective(
        sum(x_8h[t] * cost_8h_shift for t in start_times_8h) +
        sum(x_12h[t] * cost_12h_shift for t in start_times_12h),
        GRB.MINIMIZE
    )

    # --- 约束条件 ---
    # 需求约束（基于班次覆盖矩阵验证）
    # T1 00:00-04:00: 8h@00, 8h@20, 12h@00, 12h@20
    model.addConstr(
        x_8h["00:00"] + x_8h["20:00"] + x_12h["00:00"] + x_12h["20:00"] >= demand["00:00-04:00"],
        "Demand_T1"
    )
    # T2 04:00-08:00: 8h@00, 8h@04, 12h@00, 12h@20
    model.addConstr(
        x_8h["00:00"] + x_8h["04:00"] + x_12h["00:00"] + x_12h["20:00"] >= demand["04:00-08:00"],
        "Demand_T2"
    )
    # T3 08:00-12:00: 8h@04, 8h@08, 12h@00, 12h@08
    model.addConstr(
        x_8h["04:00"] + x_8h["08:00"] + x_12h["00:00"] + x_12h["08:00"] >= demand["08:00-12:00"],
        "Demand_T3"
    )
    # T4 12:00-16:00: 8h@08, 8h@12, 12h@08, 12h@12
    model.addConstr(
        x_8h["08:00"] + x_8h["12:00"] + x_12h["08:00"] + x_12h["12:00"] >= demand["12:00-16:00"],
        "Demand_T4"
    )
    # T5 16:00-20:00: 8h@12, 8h@16, 12h@08, 12h@12
    model.addConstr(
        x_8h["12:00"] + x_8h["16:00"] + x_12h["08:00"] + x_12h["12:00"] >= demand["16:00-20:00"],
        "Demand_T5"
    )
    # T6 20:00-24:00: 8h@16, 8h@20, 12h@12, 12h@20
    model.addConstr(
        x_8h["16:00"] + x_8h["20:00"] + x_12h["12:00"] + x_12h["20:00"] >= demand["20:00-24:00"],
        "Demand_T6"
    )

    # 非负约束已通过 vtype=GRB.CONTINUOUS 隐含定义，Gurobi 默认变量下界为 0

    # --- 问题 2 约束 ---
    if apply_12h_shift_constraint:
        total_12h_staff = sum(x_12h[t] for t in start_times_12h)
        total_staff = sum(x_8h[t] for t in start_times_8h) + total_12h_staff
        model.addConstr(total_12h_staff <= (1/3) * total_staff, "12h_Shift_Limit")

    # 优化模型
    model.optimize()

    # --- 结果展示 ---
    if model.status == GRB.OPTIMAL:
        print("\nOptimal solution found:")
        for t in start_times_8h:
            print(f"8-hour shift starting at {t}: {x_8h[t].X:.2f} employees")
        for t in start_times_12h:
            print(f"12-hour shift starting at {t}: {x_12h[t].X:.2f} employees")
        print(f"Minimum total cost: ${model.ObjVal:.2f}")
    else:
        print("\nNo optimal solution found.")

if __name__ == "__main__":
    print("--- Solving Problem 1 (without 12-hour shift limit) ---")
    solve_staffing_problem(apply_12h_shift_constraint=False)

    print("\n--- Solving Problem 2 (with 12-hour shift limit) ---")
    solve_staffing_problem(apply_12h_shift_constraint=True)
