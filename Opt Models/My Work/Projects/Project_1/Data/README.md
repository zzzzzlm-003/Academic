# Data Files

Input datasets for the child care capacity optimization models.

## Files

| File | Size | Description |
|------|------|-------------|
| `population_nyc.csv` | 19 KB | Child population (ages 0-5, 6-12) by zip code |
| `child_care_regulated_nyc.csv` | 837 KB | All licensed facilities with capacity and location |
| `avg_individual_income_nyc.csv` | 4.8 KB | Average household income by zip |
| `employment_rate_nyc.csv` | 4.9 KB | Labor force participation rate by zip |
| `potential_locations_nyc.csv` | 1.4 MB | Candidate sites for new facilities |

## Data Quality Notes

- 180 zip codes with positive child population
- 162 zips classified as "child care deserts" before optimization
- All data sourced from NYC Department of Health and Census Bureau
- Coordinates in latitude/longitude format

## Processing

Used in `Code/Project1_Full.ipynb` for:
1. Desert classification
2. Demand threshold calculation
3. Facility aggregation by zip code
4. Optimization constraint generation
