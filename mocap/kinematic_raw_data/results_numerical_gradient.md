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
|CPA PH|0.5145|0.2141|1.0707|
|Zero PH|0.5907|0.3425|1.9491|
|One PH|0.5911|0.4140|2.0903|
|Nearest PH|0.2030|0.1083|0.7922|
|Linear Interp PH|0.1932|0.0992|0.7150|
|Cubic Interp PH|0.1960|0.1036|0.7114|
|RBF Interp PH|0.1963|0.1024|0.7205|
### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|CPA PH|0.0682|0.0305|0.2058|
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
|CPA PH|0.6956|0.2610|1.4805|       
|Zero PH|0.6551|0.3926|2.2572|
|One PH|0.8437|0.4736|3.3480|
|Nearest PH|0.3728|0.1944|1.1399|      
|Linear Interp PH|0.3600|0.1887|1.0196|
|Cubic Interp PH|0.3638|0.1930|1.0241| 
|RBF Interp PH|0.3658|0.1951|1.0410|   

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|      
|-|-|-|-|
|CPA PH|0.0682|0.0305|0.2058|       
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
|CPA PH|0.6956|0.2610|1.4805|       
|Zero PH|1.1442|0.7430|3.4536|
|One PH|1.8899|1.4644|5.8126|
|Nearest PH|0.3995|0.2200|1.3079|      
|Linear Interp PH|0.3730|0.1853|1.0585|
|Cubic Interp PH|0.3835|0.1935|1.1256| 
|RBF Interp PH|0.3835|0.1865|1.0679|   

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|      
|-|-|-|-|
|CPA PH|0.0682|0.0305|0.2058|       
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

## Dataset 0725 R1

### Training Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.1858|0.1175|0.6884|
|Optimize PH|0.0574|0.0275|0.2197|

### Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|0.3754|0.1830|0.8618|
|Zero PH|0.9031|0.5562|2.8936|
|One PH|1.5340|1.5940|5.5823|
|Nearest PH|0.2744|0.1487|0.8633|
|Linear Interp PH|0.2620|0.1374|1.2580|
|Cubic Interp PH|0.2644|0.1386|1.1490|
|RBF Interp PH|0.2631|0.1366|0.8348|
|FBF Interp PH|0.2616|0.1355|0.8474|

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Origin PH|0.0794|0.0342|0.1762|
|CPA PH|0.0754|0.0333|0.1672|
|Zero PH|0.0777|0.0271|0.1616|
|One PH|0.4302|0.2314|0.8992|
|Nearest PH|0.0755|0.0329|0.1627|
|Linear Interp PH|0.0752|0.0331|0.1635|
|Cubic Interp PH|0.0754|0.0330|0.1632|
|RBF Interp PH|0.0754|0.0330|0.1629|
|FBF Interp PH|0.0750|0.0331|0.1642|

### Testing Data (Position) (Torch Extension)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Origin PH|1.7297|0.2110|2.1997|
|CPA PH|0.5069|0.2629|1.2805|
|Zero PH|0.9776|0.6259|3.3324|
|One PH|3.3194|2.6456|9.2988|
|Nearest PH|0.3867|0.2046|1.1036|
|Linear Interp PH|0.3730|0.1972|1.3728|
|Cubic Interp PH|0.3751|0.1981|1.2608|
|RBF Interp PH|0.3743|0.1950|1.0415|
|FBF Interp PH|0.3746|0.2088|1.3479|

### Testing Data (Orientation) (Torch Extension)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Origin PH|0.0794|0.0342|0.1762|
|CPA PH|0.0754|0.0333|0.1672|
|Zero PH|0.0777|0.0271|0.1616|
|One PH|0.4302|0.2314|0.8992|
|Nearest PH|0.0755|0.0329|0.1627|
|Linear Interp PH|0.0752|0.0331|0.1635|
|Cubic Interp PH|0.0754|0.0330|0.1632|
|RBF Interp PH|0.0754|0.0330|0.1629|
|FBF Interp PH|0.0750|0.0331|0.1642|

**P Variation**
||Mean (mm)|Std (mm)|
|-|-|-|
|P1|0.0508|0.0276|
|P2|0.9274|0.0272|
|P3|1.0286|0.0309|
|P4|0.7963|0.0356|
|P5|0.3715|0.0438|
|P6|0.3135|0.0418|
|P7|0.0874|0.0368|

**H Variation**
||Mean (deg)|Std (deg)|
|-|-|-|
|H1|0.0408|0.0437|
|H2|0.0190|0.0243|
|H3|0.0246|0.0210|
|H4|0.0341|0.0054|
|H5|0.0258|0.0044|
|H6|0.0711|0.0052|

## Dataset 0801 R1

### Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.2436|0.1387|0.8530|
|Optimize PH|0.0431|0.0243|0.1006|

### Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Origin PH|1.2898|0.2067|1.7211|
|CPA PH|0.4429|0.2017|0.9927|
|Zero PH|2.4483|2.0558|9.8207|
|One PH|0.9192|0.3739|1.6083|
|Nearest PH|0.4779|0.5330|2.8421|
|Linear Interp PH|0.3658|0.3052|1.8840|
|Cubic Interp PH|0.4361|0.4052|2.1234|
|RBF Interp PH|0.4335|0.4018|1.9572|
|FBF Interp PH|0.2721|0.1277|0.7398|

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Origin PH|0.0951|0.0366|0.2010|
|CPA PH|0.0828|0.0340|0.1733|
|Zero PH|0.1178|0.0699|0.3712|
|One PH|0.2543|0.2088|0.7638|
|Nearest PH|0.0836|0.0354|0.1922|
|Linear Interp PH|0.0816|0.0334|0.1787|
|Cubic Interp PH|0.0826|0.0341|0.1796|
|RBF Interp PH|0.0823|0.0339|0.1788|
|FBF Interp PH|0.0800|0.0333|0.1700|

### Testing Data (Position) (Torch Extension)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Origin PH|1.3294|0.2262|2.0625|
|CPA PH|0.5521|0.2884|1.3661|
|Zero PH|2.6720|2.3717|11.2400|
|One PH|1.8871|1.2568|6.2322|
|Nearest PH|0.6093|0.5627|3.1933|
|Linear Interp PH|0.4829|0.3244|2.0342|
|Cubic Interp PH|0.5581|0.4175|2.3760|
|RBF Interp PH|0.5588|0.4091|2.1923|
|FBF Interp PH|0.3701|0.2108|1.3251|

### Testing Data (Orientation) (Torch Extension)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Origin PH|0.0951|0.0366|0.2010|
|CPA PH|0.0828|0.0340|0.1733|
|Zero PH|0.1178|0.0699|0.3712|
|One PH|0.2543|0.2088|0.7638|
|Nearest PH|0.0836|0.0354|0.1922|
|Linear Interp PH|0.0816|0.0334|0.1787|
|Cubic Interp PH|0.0826|0.0341|0.1796|
|RBF Interp PH|0.0823|0.0339|0.1788|
|FBF Interp PH|0.0800|0.0333|0.1700|

**P Variation**
||Mean (mm)|Std (mm)|
|-|-|-|
|P1|0.0643|0.0367|
|P2|1.2486|0.0351|
|P3|1.1469|0.0391|
|P4|1.3984|0.0344|
|P5|0.3812|0.0536|
|P6|0.2795|0.0485|
|P7|0.0897|0.0352|

**H Variation**
||Mean (deg)|Std (deg)|
|-|-|-|
|H1|0.0885|0.0922|
|H2|0.0366|0.0544|
|H3|0.0519|0.0467|
|H4|0.0440|0.0118|
|H5|0.0270|0.0114|
|H6|0.0796|0.0130|

## Dataset 0801 R2

### Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.5452|0.2807|1.5240|
|Optimize PH|0.0542|0.0216|0.0920|

### Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Origin PH|1.1163|0.6329|2.7039|
|CPA PH|1.3006|0.5393|2.8695|
|Zero PH|1.7403|0.9989|3.9494|
|One PH|3.9402|3.3258|12.8049|
|Nearest PH|0.5307|0.2757|1.5652|
|Linear Interp PH|0.4823|0.2006|1.0977|
|Cubic Interp PH|1.3684|3.4239|47.0713|
|RBF Interp PH|0.5396|0.3040|3.2480|
|FBF Interp PH|0.4906|0.2093|1.1799|

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Origin PH|0.2203|0.1177|0.5741|
|CPA PH|0.2106|0.1113|0.5435|
|Zero PH|0.2425|0.1366|0.6504|
|One PH|1.4328|1.3235|4.2891|
|Nearest PH|0.2193|0.1168|0.5993|
|Linear Interp PH|0.2179|0.1146|0.5797|
|Cubic Interp PH|0.2850|0.3124|4.8388|
|RBF Interp PH|0.2221|0.1129|0.5862|
|FBF Interp PH|0.2222|0.1155|0.5596|

**P Variation**
||Mean (mm)|Std (mm)|
|-|-|-|
|P1|0.2350|0.1141|
|P2|0.3449|0.1751|
|P3|1.2592|0.1292|
|P4|0.9638|0.1677|
|P5|0.6204|0.1396|
|P6|0.4489|0.1315|
|P7|0.3468|0.1196|

**H Variation**
||Mean (deg)|Std (deg)|
|-|-|-|
|H1|0.1095|0.0925|
|H2|0.0723|0.0458|
|H3|0.1123|0.0665|
|H4|0.1254|0.0404|
|H5|0.1529|0.0161|
|H6|0.1003|0.0448|

## Dataset 0804 R2

### Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.5210|0.2964|1.5786|
|Optimize PH|0.0401|0.0318|0.1017|

### Testing Data (Position)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|Origin PH|1.0777|0.8088|3.0710|
|CPA PH|1.4370|0.6138|3.1874|
|Zero PH|1.7265|0.7319|3.1812|
|One PH|5.5030|4.0046|17.2480|
|Nearest PH|0.4714|0.3050|1.2889|
|Linear Interp PH|0.4205|0.2563|1.1982|
|Cubic Interp PH|1.4995|5.4042|68.9520|
|RBF Interp PH|0.4774|0.4882|5.5839|
|FBF Interp PH|0.3756|0.1341|0.8656|

### Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|Origin PH|0.1141|0.0408|0.2390|
|CPA PH|0.1126|0.0392|0.2089|
|Zero PH|0.1100|0.0473|0.2282|
|One PH|2.6914|1.8872|8.0547|
|Nearest PH|0.0980|0.0378|0.1987|
|Linear Interp PH|0.0954|0.0365|0.1923|
|Cubic Interp PH|0.1528|0.3122|4.0555|
|RBF Interp PH|0.0992|0.0410|0.3538|
|FBF Interp PH|0.0928|0.0383|0.1918|

**P Variation**
||Mean (mm)|Std (mm)|
|-|-|-|
|P1|0.2219|0.1106|
|P2|0.3431|0.1675|
|P3|1.2365|0.1301|
|P4|0.9877|0.1598|
|P5|0.6171|0.1341|
|P6|0.4525|0.1259|
|P7|0.3448|0.1149|

**H Variation**
||Mean (deg)|Std (deg)|
|-|-|-|
|H1|0.0616|0.0391|
|H2|0.0237|0.0167|
|H3|0.0668|0.0202|
|H4|0.1319|0.0091|
|H5|0.1455|0.0021|
|H6|0.1168|0.0088|

## 0613 (before/after 1 week)

Training Data
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|One PH|0.0376|0.0000|0.0376|
|Optimize PH|0.1038|0.0889|0.6007|

Testing Data (Position) (before)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|0.4175|0.1819|1.1863|
|Zero PH|0.4823|0.3019|1.4235|
|One PH|0.3914|0.1784|0.9271|
|Nearest PH|0.4408|0.3360|1.9513|
|Linear Interp PH|0.4137|0.2980|1.5877|
|Cubic Interp PH|0.4296|0.3154|1.7364|
|RBF Interp PH|0.4258|0.3112|1.6834|

Testing Data (Position) (before,torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|0.5924|0.2349|1.5129|
|Zero PH|0.7702|0.4450|2.7884|
|One PH|1.0353|0.6902|3.0205|
|Nearest PH|0.9260|0.7248|4.8042|
|Linear Interp PH|0.8810|0.6453|4.1698|
|Cubic Interp PH|0.9063|0.6832|4.3914|
|RBF Interp PH|0.9009|0.6751|4.3475|

Testing Data (Position) (after 1 week, before calib)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|1.3012|0.2026|1.8259|
|Zero PH|1.7500|0.2436|2.6042|
|One PH|1.4300|0.2566|2.0515|
|Nearest PH|1.6342|0.2835|2.9202|
|Linear Interp PH|1.6171|0.2508|2.9010|
|Cubic Interp PH|1.6274|0.2630|2.9098|
|RBF Interp PH|1.6274|0.2590|2.8778|

Testing Data (Position) (after 1 week, before calib, torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|1.8892|0.2349|2.7754|
|Zero PH|2.4756|0.3225|4.2251|
|One PH|2.0352|0.7781|3.4674|
|Nearest PH|2.4998|0.6391|6.6383|
|Linear Interp PH|2.4641|0.5670|6.0772|
|Cubic Interp PH|2.4848|0.5951|6.2547|
|RBF Interp PH|2.4838|0.5856|6.2117|

Testing Data (Position) (after 1 week, after calib)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|0.7685|0.0956|1.0443|
|Zero PH|1.1428|0.2204|1.9760|
|One PH|0.9655|0.2169|1.4482|
|Nearest PH|1.1009|0.3024|2.4021|
|Linear Interp PH|1.0833|0.2706|2.2642|
|Cubic Interp PH|1.0934|0.2839|2.2715|
|RBF Interp PH|1.0919|0.2798|2.2442|

Testing Data (Position) (after 1 week, after calib, torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|1.3045|0.1885|2.3016|
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
|CPA PH|0.2156|0.1458|0.7073|
|Zero PH|0.2352|0.1380|0.7154|
|One PH|0.5444|0.3289|1.3808|
|Nearest PH|0.2761|0.2044|1.3278|
|Linear Interp PH|0.2530|0.1859|1.2027|
|Cubic Interp PH|0.2595|0.1894|1.2968|
|RBF Interp PH|0.2564|0.1828|1.2375|

Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|CPA PH|0.0760|0.0397|0.2583|
|Zero PH|0.0824|0.0440|0.2667|
|One PH|0.1637|0.0915|0.4145|
|Nearest PH|0.0986|0.0526|0.2675|
|Linear Interp PH|0.0955|0.0492|0.2552|
|Cubic Interp PH|0.0970|0.0504|0.2603|
|RBF Interp PH|0.0966|0.0493|0.2593|

Testing Data (Position) (torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|0.3807|0.2312|1.2319|
|Zero PH|0.4088|0.2282|1.2880|
|One PH|1.2381|0.7869|3.4310|
|Nearest PH|0.6765|0.5126|2.8029|
|Linear Interp PH|0.6361|0.4622|2.6026|
|Cubic Interp PH|0.6510|0.4751|2.7714|
|RBF Interp PH|0.6475|0.4595|2.6637|

Testing Data (Orientation) (torch)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|CPA PH|0.0760|0.0397|0.2583|
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
|CPA PH|0.5761|0.2485|1.1654|
|Zero PH|0.6521|0.2810|1.2294|
|One PH|0.7860|0.5712|2.3508|
|Nearest PH|0.3973|0.1837|0.9046|
|Linear Interp PH|0.3913|0.1739|0.8530|
|Cubic Interp PH|0.3954|0.1784|0.8621|
|RBF Interp PH|0.3949|0.1793|0.8675|

Testing Data (Orientation)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|CPA PH|0.0671|0.0328|0.2381|
|Zero PH|0.0661|0.0308|0.2281|
|One PH|0.1131|0.0644|0.3152|
|Nearest PH|0.0684|0.0344|0.2320|
|Linear Interp PH|0.0681|0.0342|0.2368|
|Cubic Interp PH|0.0683|0.0343|0.2372|
|RBF Interp PH|0.0683|0.0342|0.2371|

Testing Data (Position) (torch)
||Mean (mm)|Std (mm)|Max (mm)|
|-|-|-|-|
|CPA PH|0.7075|0.3339|1.6284|
|Zero PH|0.7802|0.3709|1.7224|
|One PH|1.0171|0.6622|2.8564|
|Nearest PH|0.5342|0.2424|1.2982|
|Linear Interp PH|0.5272|0.2339|1.2782|
|Cubic Interp PH|0.5314|0.2367|1.2986|
|RBF Interp PH|0.5318|0.2363|1.3002|

Testing Data (Orientation) (torch)
||Mean (deg)|Std (deg)|Max (deg)|
|-|-|-|-|
|CPA PH|0.0671|0.0328|0.2381|
|Zero PH|0.0661|0.0308|0.2281|
|One PH|0.1131|0.0644|0.3152|
|Nearest PH|0.0684|0.0344|0.2320|
|Linear Interp PH|0.0681|0.0342|0.2368|
|Cubic Interp PH|0.0683|0.0343|0.2372|
|RBF Interp PH|0.0683|0.0342|0.2371|

