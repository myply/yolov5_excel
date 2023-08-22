## 矩阵乘模拟yolov5

#### 简介

模拟了一个yolov5x6网络的完整推理过程，使用im2col和矩阵乘，最后结果和真实结果进行比较，误差在1e-5左右。运行方法，python gen_result.py。运行时间约1.5h，主要simConv2d.py函数中27-33行中将输入转化矩阵部分非常耗时。

#### 结构目录说明

* data/目录下保存有进行inference所用的原始图片和部分参数配置
* tensor/目录下保存在真实运行情况下每层的输入和输出，为了节约空间，只保存了第一个模块的输layer0Convintensor.pt和最后一个模块的四个输出
* best.pt模拟使用的训练完毕的一个yolov5x6的模型权重文件
* gen.py用来辅助生成最终模拟文件。
* gen_result.py为最终生成的模拟文件,python gen_result.py即可运行，运行完毕后会输出模拟结果和原网络结果的误差。
* simConv2d.py进行im2col和矩阵乘的操作

## yolov5模型生成excel表格

#### 简介

如题所示，可以读取yolov5的模型，将其中每一层的维度等数据保存在excel表格中。使用方法，将genxls.py文件放置在yolov5项目下(与train.py等文件同级)，并保证统计目录下有需要生成的yolov5的.pt的权重文件，python genxls.py即可。如下三个参数可修改，分别代表需要生成的权重，输入图片的长和宽。最后生成的excel保存在excel/文件夹中。权重可选项目有yolov5n.pt,yolov5s.pt,yolov5m.pt,yolov5l.pt,yolov5x.pt,yolov5x6.pt,best.pt。

```python
model_name="best.pt"
image_h=960
image_w=1280
```

#### 测试

1. 使用best.pt生成excel时，所生成的表格与之前的yolov5矩阵乘模拟的结果是对应的。

2. 由于yolov5的版本更替，并且具体的模型参数受.yaml配置文件的影响，生成的yolov5s的表格与之前的表格并不对应，例如最开始模型没有使用Focus单元而是Conv单元，中间结构的单元数也不一致。为了统一，使用github上最新的几个权重文件进行生成。