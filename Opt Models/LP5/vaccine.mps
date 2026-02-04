NAME vaccine_distribution
OBJSENSE MAX
ROWS
 N  OBJ
 E  total_vaccines
 G  min_vaccines_A
 L  capacity_A
 G  min_vaccines_B
 L  capacity_B
 G  min_vaccines_C
 L  capacity_C
 G  min_vaccines_D
 L  capacity_D
COLUMNS
    x_A       OBJ       0.5
    x_A       total_vaccines  1
    x_A       min_vaccines_A  1
    x_A       capacity_A  1
    x_B       OBJ       0.7
    x_B       total_vaccines  1
    x_B       min_vaccines_B  1
    x_B       capacity_B  1
    x_C       OBJ       1
    x_C       total_vaccines  1
    x_C       min_vaccines_C  1
    x_C       capacity_C  1
    x_D       OBJ       0.8
    x_D       total_vaccines  1
    x_D       min_vaccines_D  1
    x_D       capacity_D  1
RHS
    RHS1      total_vaccines  5000
    RHS1      min_vaccines_A  500
    RHS1      capacity_A  2300
    RHS1      min_vaccines_B  500
    RHS1      capacity_B  1700
    RHS1      min_vaccines_C  500
    RHS1      capacity_C  1200
    RHS1      min_vaccines_D  500
    RHS1      capacity_D  900
BOUNDS
ENDATA
