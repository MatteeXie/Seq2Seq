## seq2seq原理
> 1. seq2seq = encoder + decoder
- encoder对句子进行理解
- decoder对句子进行处理，对结果进行组装
- 输入长度为M的序列，使用Encoder编码，编码结果交给decoder，decoder进行结果生成
    - Decoder通过循环，把当前的输出作为下一步的输入
    - 直到当前输出的是结束符，循环结束


> 2. 预测数字案例
- 流程：
    - 数字转化成序列，准备数据集，DataLoader
    - 完成编码器
    - 完成解码器
    - 完成整个seq2seq结构
    - 训练
    - 评估
    <br/><br/>

- 准备数据集：
    - 在样本的target中，需要实现EOS和SOS表示句子的开始和结束
    - transform中需要实现添加EOS的操作
    <br/><br/>
- 实现编码器  
encoder使用EMbedding + GRU的结构，使用最后一个时间步的hidden_state作为句子的编码结果
    - 注意
    1. Embedding和gru的参数，把batch放在前面
    2. 输出结果的形状
    3. LSTM和GRU按时间步计算，速度很慢。pytorch实现了`` nn.utils.rnn.pack_padded_sequence ``对padding后的句子进行打包能加快会哦的结果；同时实现了`` nn.utils.rnn.pad_packed_sequence``对打包的内容进行解包
    4. `` nn.utils.rnn.pack_padded_sequence``需要对batch中的内容按句子长度**降序排序**

    <br/><br/>
- 实现解码器
    - 每次输出是一个分类问题， 选择概率最大的词进行输出
    - decoder输出的形状是：[batch_size, max_len, vocab_size]
    - 损失函数: 交叉熵损失
    - 
