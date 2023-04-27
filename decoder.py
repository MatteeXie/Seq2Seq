'''
实现编码器
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = len(config.num_sequence),
                                      embedding_dim = config.embedding_dim,
                                      padding_idx = config.num_sequence.PAD)
        
        self.gru = nn.GRU(input_size = config.embedding_dim,
                          hidden_size = config.hidden_size,
                          num_layers = config.num_layer,
                          batch_first = True)
        
        self.fc = nn.Linear(config.hidden_size, len(config.num_sequence))


    def forward(self, target, encoder_hidden):
        # 1、获取encoder的输出，作为decoder第一次的hidden_state

        decoder_hidden = encoder_hidden
        batch_size = target.size(0)

        # 2、准备decoder的哥时间步得输入，[batch_size，1] SOS作为输入

        decoder_input = torch.LongTensor(torch.ones([batch_size,1], dtype=torch.int64)*config.num_sequence.SOS)

        # 3、在第一个时间步上计算得到第一个时间步的输出，hidden_state
        # 4、把前一个时间步的输出进行计算得到第一个最后的输出结果
        # 5、把前一次的hidden_state作为当前hidden_state的输入，把前一次的输出作为当前时间步得输入
        # 6、循环4-5
        for i in range(config.max_len+2):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input,decoder_hidden)
            value, index = torch.topk(decoder_output_t,1)
            decoder_input = index
            



    def forward_step(self, decoder_input, decoder_hidden):
        '''
        decoder_input:[batch_size, 1]
        decoder_hidden:[1,batch_size,hidden_size]
        '''
        decoder_input_embeded = self.embedding(decoder_input)   #[batch_size, 1, embedding_dim]

        out, decoder_hidden = self.gru(decoder_input_embeded,decoder_hidden)
        # out:[batch_size,1,hidden_size]
        # decoder_hidden:[1,batch_size,hidden_size]

        out = out.squeeze(1) #[batch_size, hidden_size]
        out = self.fc(out)  #[batch_size, vocab_size]
        output = F.log_softmax(out, dim=-1)
        print("out_put:", output.size())

        return output,decoder_hidden

        
