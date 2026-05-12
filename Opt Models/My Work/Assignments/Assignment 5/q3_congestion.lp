\ Q3: Minimum-congestion flow QP, IEOR4004 Assignment 5
\ Variables x_ij = flow on edge (i,j); objective = sum R_ij * x_ij^2
\ Note: in LP/QP file format, the [ ... ] / 2 form is the standard quadratic block;
\ since our objective is sum R*x^2 (no cross terms), the matrix Q is diagonal with
\ entries 2*R_ij so that (1/2) x' Q x = sum R_ij * x_ij^2.

Minimize
 obj: [ 4 x12 ^2 + 10 x13 ^2 + 2 x23 ^2 + 6 x24 ^2 + 4 x34 ^2
      + 8 x35 ^2 + 4 x45 ^2 + 12 x46 ^2 + 6 x56 ^2 ] / 2

Subject To
 node1: x12 + x13 = 150
 node2: x12 - x23 - x24 = 0
 node3: x13 + x23 - x34 - x35 = 0
 node4: x24 + x34 - x45 - x46 = 0
 node5: x35 + x45 - x56 = 0
 node6: x46 + x56 = 150

Bounds
 x12 >= 0
 x13 >= 0
 x23 >= 0
 x24 >= 0
 x34 >= 0
 x35 >= 0
 x45 >= 0
 x46 >= 0
 x56 >= 0

End
