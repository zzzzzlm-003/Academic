from gurobipy import Model, GRB, quicksum


def main():
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

    # Decision variables: x[j, r] >= 0 with upper bounds from capacity
    x = m.addVars(suppliers, resources, name="x", lb=0.0)
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
        print(f"No optimal solution found. Status code: {m.status}")


if __name__ == "__main__":
    main()

