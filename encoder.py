'''
编码器
'''
import torch.nn as nn
import config
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = len(config.num_sequence),
                                      embedding_dim = config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD
                                      )
        
        self.gru = nn.GRU(input_size = config.embedding_dim,
                          num_layers = config.num_layer,
                          hidden_size = config.hidden_size,
                          batch_first = True
                          )
        

    
    def forward(self, input, input_length):
        embeded = self.embedding(input) #input:[batch_size,max_len]   embeded:[batch_size,max_len,embedding_dim]
       
        embeded = pack_padded_sequence(embeded, input_length, batch_first=True) #打包

        out, hidden= self.gru(embeded)

        # 解包
        out, out_length = pad_packed_sequence(out, batch_first=True, padding_value=config.num_sequence.PAD)

        # hidden:[1*1, batch_size, hidden_size]
        # out:[batch_size, seq_len, hidden_size]
        return out, hidden, out_length

if __name__ == "__main__":
    from dataset import train_dataloader
    encoder = Encoder()
    print(encoder)
    for input, target, input_length, target_length in train_dataloader:
        out, hidden, out_length = encoder(input, input_length)
        print(out.size())
        print(hidden.size())
        print(out_length)
        break
