# IEOR 4004 Project I: Child Care Capacity Planning

## Project Overview
Mixed-integer optimization model for eliminating child care deserts in New York City.

## 📁 File Structure

```
IEOR4004_ProjectI_Datasets/
├── Data/                              📊 Raw data
│   ├── population_nyc.csv            — Child population by age and zip
│   ├── child_care_regulated_nyc.csv  — Licensed facilities & current capacity
│   ├── avg_individual_income_nyc.csv — Average household income by zip
│   ├── employment_rate_nyc.csv       — Labor force participation rate
│   └── potential_locations_nyc.csv   — Candidate sites for new facilities
│
├── Code/                              💻 Implementation
│   └── Project1_Full.ipynb           — Complete notebook (Model 1 & Model 2)
│
├── Reports/                           📄 Documentation
│   ├── Model1_Report.tex             — LaTeX report (Overleaf-ready)
│   ├── IEOR4004_ProjectI_Description.pdf  — Project assignment
│   ├── Optimization_Project1_Model1.pdf   — Compiled Model 1 results
│   └── Opt_project_2nd_model_(1).pdf      — Model 2 analysis
│
└── README.md                          📋 This file
```

## 🚀 Quick Start

1. **Run the notebook** to generate optimization results:
   - Open `Project1_Full.ipynb` in Jupyter
   - Execute all cells to solve Model 1

2. **Compile the report**:
   - Open `Model1_Report.tex` in Overleaf or local LaTeX compiler
   - All numeric results are pre-filled

3. **Key Results** (Model 1):
   - Optimal total cost: $215,083,191
   - Status: OPTIMAL (proven)
   - Runtime: 0.07 seconds

## 📋 Model Summary

**Objective**: Minimize total funding to eliminate all child care deserts while meeting NYC's 0–5 age policy

**Constraints**:
- Desert elimination (total slots)
- 0–5 age requirement (≥2/3 of population)
- Expansion capacity limits
- New facility allocation

**Decision Variables**:
- Expansion slots (continuous)
- New facilities by type (integer)
- 0–5 equipment configuration (continuous)

## 📊 Data Quality
- 180 zip codes with child population
- 162 zips classified as deserts pre-optimization
- All constraints satisfied at optimum

---

**Last Updated**: March 25, 2026
**Status**: Complete
