\ Model vaccine_distribution
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.5 x_A + 0.7 x_B + x_C + 0.8 x_D
Subject To
 total_vaccines: x_A + x_B + x_C + x_D = 5000
 min_vaccines_A: x_A >= 500
 capacity_A: x_A <= 2300
 min_vaccines_B: x_B >= 500
 capacity_B: x_B <= 1700
 min_vaccines_C: x_C >= 500
 capacity_C: x_C <= 1200
 min_vaccines_D: x_D >= 500
 capacity_D: x_D <= 900
Bounds
End
