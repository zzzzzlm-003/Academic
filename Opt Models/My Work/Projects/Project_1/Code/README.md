# Code

Jupyter notebook implementing the optimization models.

## Files

| File | Size | Purpose |
|------|------|---------|
| `Project1_Full.ipynb` | 93 KB | Complete implementation with both models |

## Contents

### Model 1: Idealistic Baseline
- **Objective**: Minimize total funding to eliminate child care deserts
- **Assumptions**: Unlimited capacity, no geographic constraints
- **Outcome**: Cost lower bound
- **Status**: OPTIMAL (0.07 seconds)
- **Optimal Cost**: $215,083,191

### Model 2: Realistic Constraints
- **Additional constraints**: Minimum 0.06-mile facility spacing
- **Piecewise costs**: Expansion tiers (Tier 1, 2, 3)
- **Outcome**: More accurate real-world estimate

## Running the Notebook

```bash
jupyter notebook Project1_Full.ipynb
```

1. Run cells 1-24 for Model 1
2. Run cells 25+ for Model 2
3. Results extracted automatically

## Dependencies

- Python 3.8+
- Gurobi (commercial solver, academic license required)
- pandas, numpy, matplotlib
