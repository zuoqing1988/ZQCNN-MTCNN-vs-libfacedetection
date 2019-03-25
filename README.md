# ZQCNN-MTCNN-vs-libfacedetection

## ZQCNN:下载时间2019-03-25 09:00

依赖库:

windows：mkl, opencv3.4.2

arm-linux: openblas, opencv3.4.2

## libfacedetection:下载时间2019-03-25 09:00

# (一) WINDOWS下对比 (测试机器为E5-1650V4 3.6GHz)

ZQCNN用.sln打开

libfacedetection用cmake-gui配置，勾选avx2和DEMO，不勾选int8和neon，配置之后用vs打开，注意在facedetection、facedetection_shared、fdt_demo三个项目的项目属性->C++->代码生成里开启AVX2和快速浮点。

**对比keliamoniz1.jpg**

| 输入尺寸     | ZQCNN单线程  |  libfacedetection单线程 | 备注       |
| ------       | ---------    | -----------------       | --------   |
| 640x480      |  约27ms      | 约75ms                  | nIter=1000 |
| 320x240      |  约9.0ms     | 约18ms                  | nIter=1000 |
| 160x120      |  约2.8ms     | 约4.0ms                 | nIter=1000 |
| 128x96       |  约2.0ms     | 约2.6ms                 | nIter=10000|

**对比4.jpg**

| 输入尺寸     | ZQCNN单线程 | ZQCNN找到个数 |  libfacedetection单线程 | libfacedetection找到个数 | 备注       |
| ------       | ---------   | ------------- | -----------------       | ------------------------ | -------    |
| 640x480      |  约30ms     | 4             | 约75ms                  | 4                        | nIter=1000 |
| 320x240      |  约10ms     | 4             | 约18ms                  | 4                        | nIter=1000 |
| 160x120      |  约4.7ms    | 4             | 约4.0ms                 | 4                        | nIter=1000 |
| 128x96       |  约3.5ms    | 4             | 约2.6ms                 | 3                        | nIter=10000|

**对比其他图片**

| 图片名      | 输入尺寸     | ZQCNN单线程 | ZQCNN找到个数 |  libfacedetection单线程 | libfacedetection找到个数 | 备注      |
| ------      | ------       | ---------   | ------------- | -----------------       | ------------------------ | --------  |
| 1.jpg       | 1920x1080    |  约165ms    | 6             | 约522ms                 | 10 (2虚检)               | nIter=100 |
| 2.jpg       | 3164x1440    |  约354ms    | 44 (2虚检)    | 约1200ms                | 42 (很多框不准)          | nIter=10  |
| test.jpg    | 1920x1080    |  约200ms    | 60            | 约526ms                 | 50                       | nIter=100 |
| test2.jpg   | 2048x1217    |  约240ms    | 82            | 约656ms                 | 50                       | nIter=50  |
| test11.jpg  | 5520x3680    |  约1800ms   | 844           | 约5230ms                | 50                       | nIter=10  |
| test12.jpg  | 1280x720     |  约85ms     | 14            | 约238ms                 | 14                       | nIter=100 |