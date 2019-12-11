# Requirements
```
* pytorch >=1.1.0
* python == 3.6.*
* pytopicrank
```
# 2019-12-11待优化记录
## 待完成
```
* 梳理并重新写一下Log的打印位置
* 编写模型保存函数
* 编写eval函数
* 编写rouge测试函数
```
## 待优化
```
* 在生成阶段，原计划是生成到EOS token后停止生成,但是由于是一个minibatch的数据，这样好像无法因为某一条数据生成EOS而停止
* 数据处理的过程中，未将SOS和EOS加入到标题的首尾。
* 模型用的是CPU，还没有将CUDA用起来
* 去除padding的影响，但是由于target的padding有两类，一类是常规的index的padding, 一类是type的padding,由于type是0，1标注，因此对于type是否能够消除padding的影响存疑。
* 数据中topic phrase的生成。目前使用的是github某一开源代码：pytopicrank
```
