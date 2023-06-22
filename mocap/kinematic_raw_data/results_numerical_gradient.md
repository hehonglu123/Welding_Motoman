# Result 0504 Zero

## Calibrate Stretch

|    | P2          | P3          | P4          | P5          | P6          | P7          |
|----|-------------|-------------|-------------|-------------|-------------|-------------|
|    | [148.886, 0.0, 0.0]             | [-0.533, 0.0, 760.728]          | [1082.312, 1.112, 199.4]       | [0.0, -1.112, 0.349]           | [0.0, 1.152, -0.329]           | [245.123, 7.712, -36.64]       |
|    | [150.271, -1.548, -0.74] | [0.517, -0.84, 759.274]  | [1081.66, 2.694, 200.49]  | [2.491, 1.297, -0.464]  | [-2.245, -1.826, 0.87]  | [246.118, 6.491, -37.619] |

|    | H1          | H2          | H3          | H4          | H5          | H6          |
|----|-------------|-------------|-------------|-------------|-------------|-------------|
|    | [0.0, 0.0, 1.0] | [0.0, 1.0, -0.0]      | [0.0, -1.0, 0.0]     | [-1.0, -0.001, 0.0]    | [0.001, -1.0, 0.001]   | [-1.0, -0.001, 0.001]  |
|  | [0.005, -0.009, 0.999]  | [-0.061, 0.995, -0.076] | [0.062, -0.993, 0.097] | [-0.999, -0.016, -0.031] | [0.011, -0.999, 0.024]   | [-1, -0.004, -0.003]    |

|                    | Mean Position Error Vec           | Mean Abs Position Error Vec           | Mean Position Error | Std Position Error | Std Position Error Norm | Mean Orientation Error Vec     | Mean Orientation Error | Std Orientation Error | Std Orientation Error Norm |
|--------------------|-----------------------------------|---------------------------------------|---------------------|--------------------|------------------------|-------------------------------|------------------------|-----------------------|---------------------------|
| **Training**       | [0.00002, 0.00001, -0.00004]      | [0.02470, 0.02353, 0.05179]          | 0.06601             | [0.02881, 0.02762, 0.05874]          | 0.02619                | [0.00000, -0.00000, -0.00000]    | 0.00316                | [0.00235, 0.00151, 0.00190] | 0.00120                   |
| **Testing**        | [-0.01338, 0.00541, 0.01706]      | [0.01953, 0.01682, 0.02521]          | 0.04075             | [0.01601, 0.02223, 0.02778]          | 0.01899                | [-0.00039, -0.00051, -0.00038] | 0.00219                | [0.00119, 0.00155, 0.00074] | 0.00036                   |

|                            | Pose Zero                         | Pose Stretch                      | Pose Inward                       | Pose Right                        | Pose Left                         |
|----------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
| **Mean Position Error**    | 1.91099                           | 0.05229                           | 106.63089                         | 44.31922                          | 62.44648                          |
| **Std Position Error**     | 0.04207                           | 0.02676                           | 0.02546                           | 0.02825                           | 0.01705                           |
| **Mean Orientation Error** | 0.05210                           | 0.00273                           | 1.37860                           | 2.58527                           | 2.23750                           |
| **Std Orientation Error**  | 0.00190                           | 0.00099                           | 0.00167                           | 0.00215                           | 0.00131                           |

## Dataset 0516
### Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.2198|0.1166|0.7097|
|Optimize PH|0.0936|0.0262|0.1927|
### Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.5145|0.2141|1.0707|
|Zero PH|0.5907|0.3425|1.9491|
|One PH|0.5911|0.4140|2.0903|
|Nearest PH|0.2030|0.1083|0.7922|
|Linear Interp PH|0.1932|0.0992|0.7150|
|Cubic Interp PH|0.1960|0.1036|0.7114|
|RBF Interp PH|0.1963|0.1024|0.7205|
### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Rotate PH|0.0682|0.0305|0.2058|
|Zero PH|0.0631|0.0291|0.1962|
|One PH|0.1191|0.0496|0.2497|
|Nearest PH|0.0637|0.0258|0.1624|
|Linear Interp PH|0.0635|0.0255|0.1664|
|Cubic Interp PH|0.0635|0.0256|0.1653|
|RBF Interp PH|0.0635|0.0255|0.1657|

(using torch extension)
### Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.6956|0.2610|1.4805|       
|Zero PH|0.6551|0.3926|2.2572|
|One PH|0.8437|0.4736|3.3480|
|Nearest PH|0.3728|0.1944|1.1399|      
|Linear Interp PH|0.3600|0.1887|1.0196|
|Cubic Interp PH|0.3638|0.1930|1.0241| 
|RBF Interp PH|0.3658|0.1951|1.0410|   

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|      
|-|-|-|-|
|Rotate PH|0.0682|0.0305|0.2058|       
|Zero PH|0.0631|0.0291|0.1962|
|One PH|0.1191|0.0496|0.2497|
|Nearest PH|0.0637|0.0258|0.1624|      
|Linear Interp PH|0.0635|0.0255|0.1664|
|Cubic Interp PH|0.0635|0.0256|0.1653| 
|RBF Interp PH|0.0635|0.0255|0.1657|

**P Variation**
||Mean (mm)|Std (mm)|
|-|-|-|
|P1|0.0775|0.0339|
|P2|1.2778|0.0346|
|P3|0.8880|0.0385|
|P4|1.1334|0.0265|
|P5|0.9242|0.0204|
|P6|0.9094|0.0170|

**H Variation**
||Mean (mm)|Std (mm)|
|-|-|-|
|H1|0.0303|0.0276|
|H2|0.0290|0.0182|
|H3|0.0315|0.0201|
|H4|0.0197|0.0001|
|H5|0.0370|0.0001|
|H6|0.0483|0.0001|

## Dataset 0516 (torch optimize)

### Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.6956|0.2610|1.4805|       
|Zero PH|1.1442|0.7430|3.4536|
|One PH|1.8899|1.4644|5.8126|
|Nearest PH|0.3995|0.2200|1.3079|      
|Linear Interp PH|0.3730|0.1853|1.0585|
|Cubic Interp PH|0.3835|0.1935|1.1256| 
|RBF Interp PH|0.3835|0.1865|1.0679|   

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|      
|-|-|-|-|
|Rotate PH|0.0682|0.0305|0.2058|       
|Zero PH|0.0675|0.0340|0.2252|
|One PH|0.1991|0.1437|0.5322|
|Nearest PH|0.0695|0.0364|0.2257|      
|Linear Interp PH|0.0677|0.0332|0.1935|
|Cubic Interp PH|0.0683|0.0336|0.1915| 
|RBF Interp PH|0.0679|0.0330|0.1947|   

### Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.2956|0.1522|0.8904|
|Optimize PH|0.1339|0.0728|0.5452| 

**P Variation**
||Mean (mm)|Std (mm)|
|-|-|-|
|P1|0.1035|0.0564|   
|P2|0.6957|0.0456|   
|P3|1.3199|0.0611|   
|P4|1.0957|0.0796|   
|P5|0.5768|0.0790|   
|P6|0.5249|0.0659|   

**H Variation**
||Mean (deg)|Std (deg)|
|-|-|-|
|H1|0.0481|0.0395|
|H2|0.0539|0.0505|
|H3|0.0658|0.0287|
|H4|0.0623|0.0002|
|H5|0.0018|0.0002|
|H6|0.0609|0.0002|

## 0613 (before/after 1 week)

Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.0376|0.0000|0.0376|
|Optimize PH|0.1038|0.0889|0.6007|

Testing Data (Position) (before)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.4175|0.1819|1.1863|
|Zero PH|0.4823|0.3019|1.4235|
|One PH|0.3914|0.1784|0.9271|
|Nearest PH|0.4408|0.3360|1.9513|
|Linear Interp PH|0.4137|0.2980|1.5877|
|Cubic Interp PH|0.4296|0.3154|1.7364|
|RBF Interp PH|0.4258|0.3112|1.6834|

Testing Data (Position) (before,torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.5924|0.2349|1.5129|
|Zero PH|0.7702|0.4450|2.7884|
|One PH|1.0353|0.6902|3.0205|
|Nearest PH|0.9260|0.7248|4.8042|
|Linear Interp PH|0.8810|0.6453|4.1698|
|Cubic Interp PH|0.9063|0.6832|4.3914|
|RBF Interp PH|0.9009|0.6751|4.3475|

Testing Data (Position) (after 1 week, before calib)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|1.3012|0.2026|1.8259|
|Zero PH|1.7500|0.2436|2.6042|
|One PH|1.4300|0.2566|2.0515|
|Nearest PH|1.6342|0.2835|2.9202|
|Linear Interp PH|1.6171|0.2508|2.9010|
|Cubic Interp PH|1.6274|0.2630|2.9098|
|RBF Interp PH|1.6274|0.2590|2.8778|

Testing Data (Position) (after 1 week, before calib, torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|1.8892|0.2349|2.7754|
|Zero PH|2.4756|0.3225|4.2251|
|One PH|2.0352|0.7781|3.4674|
|Nearest PH|2.4998|0.6391|6.6383|
|Linear Interp PH|2.4641|0.5670|6.0772|
|Cubic Interp PH|2.4848|0.5951|6.2547|
|RBF Interp PH|2.4838|0.5856|6.2117|

Testing Data (Position) (after 1 week, after calib)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.7685|0.0956|1.0443|
|Zero PH|1.1428|0.2204|1.9760|
|One PH|0.9655|0.2169|1.4482|
|Nearest PH|1.1009|0.3024|2.4021|
|Linear Interp PH|1.0833|0.2706|2.2642|
|Cubic Interp PH|1.0934|0.2839|2.2715|
|RBF Interp PH|1.0919|0.2798|2.2442|

Testing Data (Position) (after 1 week, after calib, torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|1.3045|0.1885|2.3016|
|Zero PH|1.8487|0.3287|3.6938|
|One PH|1.6416|0.6828|3.1236|
|Nearest PH|1.9174|0.7218|6.3543|
|Linear Interp PH|1.8810|0.6522|5.7697|
|Cubic Interp PH|1.9020|0.6805|5.9607|
|RBF Interp PH|1.9002|0.6713|5.9170|

## 0620 

Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.2167|0.1146|0.7157|
|Optimize PH|0.0770|0.0508|0.3594|

Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.2156|0.1458|0.7073|
|Zero PH|0.2352|0.1380|0.7154|
|One PH|0.5444|0.3289|1.3808|
|Nearest PH|0.2761|0.2044|1.3278|
|Linear Interp PH|0.2530|0.1859|1.2027|
|Cubic Interp PH|0.2595|0.1894|1.2968|
|RBF Interp PH|0.2564|0.1828|1.2375|

Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Rotate PH|0.0760|0.0397|0.2583|
|Zero PH|0.0824|0.0440|0.2667|
|One PH|0.1637|0.0915|0.4145|
|Nearest PH|0.0986|0.0526|0.2675|
|Linear Interp PH|0.0955|0.0492|0.2552|
|Cubic Interp PH|0.0970|0.0504|0.2603|
|RBF Interp PH|0.0966|0.0493|0.2593|

Testing Data (Position) (torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.3807|0.2312|1.2319|
|Zero PH|0.4088|0.2282|1.2880|
|One PH|1.2381|0.7869|3.4310|
|Nearest PH|0.6765|0.5126|2.8029|
|Linear Interp PH|0.6361|0.4622|2.6026|
|Cubic Interp PH|0.6510|0.4751|2.7714|
|RBF Interp PH|0.6475|0.4595|2.6637|

Testing Data (Orientation) (torch)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Rotate PH|0.0760|0.0397|0.2583|
|Zero PH|0.0824|0.0440|0.2667|
|One PH|0.1637|0.0915|0.4145|
|Nearest PH|0.0986|0.0526|0.2675|
|Linear Interp PH|0.0955|0.0492|0.2552|
|Cubic Interp PH|0.0970|0.0504|0.2603|
|RBF Interp PH|0.0966|0.0493|0.2593|

## 0621

Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.2363|0.1289|0.8810|
|Optimize PH|0.0910|0.0480|0.3778|

Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.5761|0.2485|1.1654|
|Zero PH|0.6521|0.2810|1.2294|
|One PH|0.7860|0.5712|2.3508|
|Nearest PH|0.3973|0.1837|0.9046|
|Linear Interp PH|0.3913|0.1739|0.8530|
|Cubic Interp PH|0.3954|0.1784|0.8621|
|RBF Interp PH|0.3949|0.1793|0.8675|

Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Rotate PH|0.0671|0.0328|0.2381|
|Zero PH|0.0661|0.0308|0.2281|
|One PH|0.1131|0.0644|0.3152|
|Nearest PH|0.0684|0.0344|0.2320|
|Linear Interp PH|0.0681|0.0342|0.2368|
|Cubic Interp PH|0.0683|0.0343|0.2372|
|RBF Interp PH|0.0683|0.0342|0.2371|

Testing Data (Position) (torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Rotate PH|0.7075|0.3339|1.6284|
|Zero PH|0.7802|0.3709|1.7224|
|One PH|1.0171|0.6622|2.8564|
|Nearest PH|0.5342|0.2424|1.2982|
|Linear Interp PH|0.5272|0.2339|1.2782|
|Cubic Interp PH|0.5314|0.2367|1.2986|
|RBF Interp PH|0.5318|0.2363|1.3002|

Testing Data (Orientation) (torch)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Rotate PH|0.0671|0.0328|0.2381|
|Zero PH|0.0661|0.0308|0.2281|
|One PH|0.1131|0.0644|0.3152|
|Nearest PH|0.0684|0.0344|0.2320|
|Linear Interp PH|0.0681|0.0342|0.2368|
|Cubic Interp PH|0.0683|0.0343|0.2372|
|RBF Interp PH|0.0683|0.0342|0.2371|

