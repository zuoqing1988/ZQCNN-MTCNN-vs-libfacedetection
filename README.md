# ZQCNN-MTCNN-vs-libfacedetection

## ZQCNN:下载时间2019-03-25 09:00

依赖库:

windows：mkl, opencv3.4.2

arm-linux: openblas, opencv3.4.2

## libfacedetection:下载时间2019-03-25 09:00

# WINDOWS下对比

ZQCNN用.sln打开

libfacedetection用cmake-gui配置，勾选avx2和DEMO，不勾选int8和neon，配置之后用vs打开，注意在facedetection、facedetection_shared、fdt_demo三个项目的项目属性->C++->代码生成里开启AVX2和快速浮点。

**对比keliamoniz1.jpg**

| 输入尺寸     | ZQCNN单线程  |  libfacedetection单线程 | 备注                         |
| ------       | ---------    | -----------------       | --------                     |
| 640x480      |              | 约75ms                  | E5-1650V4 3.6GHz, nIter=1000 |
| 320x240      |              | 约18ms                  | E5-1650V4 3.6GHz, nIter=1000 |
| 160x120      |              | 约4.0ms                 | E5-1650V4 3.6GHz, nIter=1000 |
| 128x96       |              | 约2.6ms                 | E5-1650V4 3.6GHz, nIter=10000|


