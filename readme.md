##  **一、配置文件信息：**
 
 1. config文件中的defaults.py存放默认配置信息
 2. run文件夹中的monoflex.yaml文件（主要存放和网络结构相关的命令）
 3. engine文件夹下的 default_argument_parser函数（将文件作脚本运行时，通过命令行配置一些参数）
 ~~4. config文件夹下的paths_catalog文件（主要存放数据的结构形式）~~ 
 5. ImageSets文件夹：包含若干txt文件，每个文件中指明，训练或者测试的图片的名称

其中，1，2，3中存在部分相同的配置信息，当对同属性进行配置时，**优先级为③>②>①**


## **二、对自有文件进行测试**

**测试前预准备：**

1.创建如下图所示结构的文件夹，将测试图像和每张图像的标定信息分别存放于test文件夹下的calib和image_2文件夹下（图像文件名称与标定文件名称相同且一一对应）

2.在Imagesets文件夹中创建test.txt文件，在文件中存放所有待检测的图像的名称（不要后缀，每个文件名单独占用一行）

3.将testing文件夹的目录路径赋予config文件夹下的defaults.py文件夹下的_C.IFERENCE.ROOT变量

4. 检查my_engine文件夹下的default_argument_parser参数设置

5. 运行visual_result.py

```python
CUDA_VISIBLE_DEVICES=0 python tools/visual_result.py --config runs/monoflex.yaml   --eval --vis
```