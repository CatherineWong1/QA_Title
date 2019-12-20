# Requirements
```
* pytorch >=1.1.0
* python == 3.6.*
* pytopicrank
```
# Version1待优化记录
## 待完成
```
* 编写eval函数
* 编写rouge测试函数
```
## 待优化
```
* 在生成阶段，原计划是生成到EOS token后停止生成,但是由于是一个minibatch的数据，这样好像无法因为某一条数据生成EOS而停止
* 数据处理的过程中，未将SOS和EOS加入到标题的首尾。
* 去除padding的影响，但是由于target的padding有两类，一类是常规的index的padding, 一类是type的padding,由于type是0，1标注，因此对于type是否能够消除padding的影响存疑。
* 数据中topic phrase的生成。目前使用的是github某一开源代码：pytopicrank
* 没有使用Beam Search的方法来生成。后续可以加上。
```
Notes:
* 在看Pytorch的官方Tutorial（NMT和Chatbot）的时候，发现了一种teacher forcing的机制。它有一个明显的缺点就是在inference阶段，因为没有ground truth作为输入，会得到不太好的perfomance，但是依然有很多人去用。
* 关于Teacher Forcing，ACL2019的Best Paper《Bridging the Gap between Training and Inference for Neural Machine Translation》中有提到解决方案。
* 关于padding的影响，如果使用了mask, 可以使用mask将attention中的pad token置0.从而在Attention这块消除pad的影响。

# Version1优化记录
## 已完成
```
* 增加了一个Epoch结束后，打印loss；删除了一个Batch Data中出现的Log
* 增加了一个Epoch结束后，保存模型的代码段。同时由于要保存Embedding的state_dict(), 修改了Model中Encoder和Decoder这块的init函数。
* 增加了load之前训练好的model，继续训练的代码段
* 增加了eval()函数，同时修改了train_iters()这个函数。目前eval()函数还未加上
* 模型可在GPU上进行训练。
```
