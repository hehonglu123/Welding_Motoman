# Error Statistics

## Error Statistics dh
| Unit (mm) | Train RMSE dh| Test RMSE dh | Train Max dh | Test Max dh | Interval 95% |
|---|---|---|---|---|---|
| Linear loglog model | 0.28 | 0.30 | 1.72 | 1.60 | 0.00~0.68 |
| Quadratic loglog model | 0.26 | 0.27 | 1.77 | 1.73 | 0.00~0.64 |
| Neural Network model | 0.16 | 0.17 | 1.06 | 1.24 | 0.00~0.36 |
| Gaussian Process model | 0.07 | 0.23 | 0.43 | 1.59 | 0.00~0.19 |

## Error Statistics dw
| Unit (mm) | Train RMSE dw| Test RMSE dw | Train Max dw | Test Max dw | Interval 95% |
|---|---|---|---|---|---|
| Linear loglog model | 0.56 | 0.52 | 3.71 | 2.01 | 0.01~1.21 |
| Quadratic loglog model | 0.56 | 0.54 | 3.41 | 2.46  | 0.01~1.21 |
| Neural Network model | 0.41 | 0.34 | 3.55 | 1.93 | 0.01~0.86 |
| Gaussian Process model | 0.28 | 0.39 | 1.85 | 1.76 | 0.00~0.65 |

# NN Ablation

## Error Statistics dh
| Unit (mm) | Train RMSE dh| Test RMSE dh | Train Max dh | Test Max dh | Interval 95% |
|---|---|---|---|---|---|
| NN (v $\omega$) | 0.27 | 0.29 | 1.56 | 1.58 | 0.00~0.64 |
| NN (v $\omega$ $h_t$) | 0.18 | 0.19 | 1.04 | 1.22 | 0.00~0.40 |
| NN (v $\omega$ $h_l$) | 0.21 | 0.26 | 1.11 | 1.77 | 0.00~0.53 |
| NN (v $\omega$ $h_t$ $h_l$) | 0.17 | 0.20 | 1.03 | 1.28 | 0.00~0.41 |

## Error Statistics dw
| Unit (mm) | Train RMSE dw| Test RMSE dw | Train Max dw | Test Max dw | Interval 95% |
|---|---|---|---|---|---|
| NN (v $\omega$) | 0.52 | 0.44 | 4.14 | 2.21 | 0.01~1.07 |
| NN (v $\omega$ $h_t$) | 0.44 | 0.35 | 3.74 | 1.83 | 0.01~0.96 |
| NN (v $\omega$ $h_l$) | 0.45 | 0.44 | 3.61 | 2.51 | 0.01~0.93 |
| NN (v $\omega$ $h_t$ $h_l$) | 0.40 | 0.42 | 3.12 | 2.20 | 0.01~0.87 |

Log-log time: <0.01 sec
NN time: 185.52 sec
GP time: 1249.72 sec 