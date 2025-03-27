# Error Distribution dh
| Unit (mm) | Train RMSE dh| Test RMSE dh | Train Max dh | Test Max dh | Interval 95% |
|---|---|---|---|---|---|
| Linear loglog model | 0.28 | 0.30 | 1.72 | 1.60 | 0.00~0.68 |
| Quadratic loglog model | 0.26 | 0.27 | 1.77 | 1.73 | 0.00~0.64 |
| Neural Network model | 0.16 | 0.17 | 1.06 | 1.24 | 0.00~0.36 |
| Gaussian Process model | 0.07 | 0.23 | 0.43 | 1.59 | 0.00~0.19 |

# Error Distribution dw
| Unit (mm) | Train RMSE dw| Test RMSE dw | Train Max dw | Test Max dw | Interval 95% |
|---|---|---|---|---|---|
| Linear loglog model | 0.56 | 0.52 | 3.71 | 2.01 | 0.01~1.21 |
| Quadratic loglog model | 0.56 | 0.54 | 3.41 | 2.46  | 0.01~1.21 |
| Neural Network model | 0.41 | 0.34 | 3.55 | 1.93 | 0.01~0.86 |
| Gaussian Process model | 0.28 | 0.39 | 1.85 | 1.76 | 0.00~0.65 |

Log-log time: <0.01 sec
NN time: 185.52 sec
GP time: 1249.72 sec