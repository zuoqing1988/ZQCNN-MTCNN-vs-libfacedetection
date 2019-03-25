# ZQCNN-MTCNN-vs-libfacedetection

## ZQCNN:下载时间2019-03-25 09:00

依赖库:

windows：mkl, opencv3.4.2

arm-linux: openblas, opencv3.4.2

## libfacedetection:下载时间2019-03-25 09:00

# (一) WINDOWS下对比 (测试机器为E5-1650V4 3.6GHz)

ZQCNN用.sln打开

libfacedetection用cmake-gui配置，勾选avx2和DEMO，不勾选int8和neon，配置之后用vs打开，注意在facedetection、facedetection_shared、fdt_demo三个项目的项目属性->C++->代码生成里开启AVX2和快速浮点。

**(1)对比keliamoniz1.jpg**

| 输入尺寸     | ZQCNN单线程|  libfacedetection单线程 | 备注       |
| ------       | ---------  | -----------------       | --------   |
| 640x480      |  27ms      | 75ms                    | nIter=1000 |
| 320x240      |  9.0ms     | 18ms                    | nIter=1000 |
| 160x120      |  2.8ms     | 4.0ms                   | nIter=1000 |
| 128x96       |  2.0ms     | 2.6ms                   | nIter=10000|

**(2)对比4.jpg**

| 输入尺寸     | ZQCNN单线程 | ZQCNN找到个数 |  libfacedetection单线程 | libfacedetection找到个数 | 备注       |
| ------       | ---------   | ------------- | -----------------       | ------------------------ | -------    |
| 640x480      |  30ms       | 4             | 75ms                    | 4                        | nIter=1000 |
| 320x240      |  10ms       | 4             | 18ms                    | 4                        | nIter=1000 |
| 160x120      |  4.7ms      | 4             | 4.0ms                   | 4                        | nIter=1000 |
| 128x96       |  3.5ms      | 4             | 2.6ms                   | 3                        | nIter=10000|

**(3)对比其他图片**

| 图片名      | 输入尺寸     | ZQCNN单线程 | ZQCNN找到个数 |  libfacedetection单线程 | libfacedetection找到个数 | 备注      |
| ------      | ------       | ---------   | ------------- | -----------------       | ------------------------ | --------  |
| 1.jpg       | 1920x1080    |  165ms      | 6             | 522ms                   | 10 (2虚检)               | nIter=100 |
| 2.jpg       | 3164x1440    |  354ms      | 44 (2虚检)    | 1200ms                  | 42 (很多框不准)          | nIter=10  |
| test.jpg    | 1920x1080    |  200ms      | 60            | 526ms                   | 50                       | nIter=100 |
| test2.jpg   | 2048x1217    |  240ms      | 82            | 656ms                   | 50                       | nIter=50  |
| test11.jpg  | 5520x3680    |  1800ms     | 844           | 5230ms                  | 50                       | nIter=10  |
| test12.jpg  | 1280x720     |  85ms       | 14            | 238ms                   | 14                       | nIter=100 |

# (二) ARM-LINUX下对比 (测试机器为firefly rk3399)

ZQCNN: 先编译OpenBLAS, OpenCV3.4.2, 然后编译ZQCNN, 使用命令cmake .. -DSIMD_ARCH_TYPE=arm64 -DBLAS_TYPE=openblas_zq_gemm 具体参见[ZQCNN项目的README](https://github.com/zuoqing1988/ZQCNN)

libfacedetection: 命令cmake .. -DENABLE_NEON=ON -DCMAKE_BUILD_TYPE=RELEASE


**(1)对比keliamoniz1.jpg**

| 输入尺寸     | ZQCNN单A72   | ZQCNN单A53   |  libfacedetection单A72 |  libfacedetection单A53   | 备注       |
| ------       | ---------    | -----------  |   ------------         |  ------------------      | --------   |
| 640x480      |  129ms       | 327ms        |    341ms               |  628ms                   | nIter=100  |
| 320x240      |  37.7ms      | 92.8ms       |    84ms                |  152ms                   | nIter=100  |
| 160x120      |  11.5ms      | 29.2ms       |    19.5ms              |  35.8ms                  | nIter=500  |
| 128x96       |  8.77ms      | 22.2ms       |    13.0ms              |  23.9ms                  | nIter=1000 |

**(2)对比4.jpg**

| 输入尺寸     | ZQCNN单A72   | ZQCNN单A53  | ZQCNN找到个数 |  libfacedetection单A72 |  libfacedetection单A53  | libfacedetection找到个数  | 备注       |
| ------       | ---------    | ----------- | -----------   |   ------------         |  ------------------     | ------------------------- | --------   |
| 640x480      |  133ms       | 340ms       |       4       |  315ms                 |    598ms                | 4                         | nIter=100  |
| 320x240      |  42.5ms      | 108ms       |       4       |  77.5ms                |    144ms                | 4                         | nIter=100  |
| 160x120      |  19.7ms      | 50.9ms      |       4       |  18.0ms                |    33.7ms               | 4                         | nIter=500  |
| 128x96       |  13.5ms      | 35.8ms      |       4       |  12.0ms                |    22.6ms               | 3                         | nIter=1000 |

**(3)对比其他图片**

| 图片名      | 输入尺寸     | ZQCNN单A72 | ZQCNN单A53 | ZQCNN找到个数 |  libfacedetection单A72 |  libfacedetection单A53  | libfacedetection找到个数 | 备注     |
| ------      | ------       | ---------  | ---------- | ------------- |   ------------         |  ------------------     | ------------------------ | -------  |
| 1.jpg       | 1920x1080    |  845ms     | 2230ms     | 6             |   2300ms               |  4265ms                 | 10 (2虚检)               | nIter=10 |
| test.jpg    | 1920x1080    |  940ms     | 2490ms     | 60            |   2307ms               |  4266ms                 | 50                       | nIter=10 |
| test2.jpg   | 2048x1217    |  1210ms    | 3150ms     | 82            |   2800ms               |  5185ms                 | 50                       | nIter=10 |
| test12.jpg  | 1280x720     |  412ms     | 1110ms     | 14            |   1034ms               |  1896ms                 | 14                       | nIter=10 |
