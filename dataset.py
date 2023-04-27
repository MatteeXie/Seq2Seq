# 准备数据集，Dataset， DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import config

class NumDataset(Dataset):
    def __init__(self):
        self.data = np.random.randint(0,1e8,size=[500000])

    def __getitem__(self, index):
        input = list(str(self.data[index]))
        label = input + ["0"]
        input_length = len(input)
        label_length = len(label)
        return input, label, input_length, label_length
    
    def __len__(self):
        return self.data.shape[0]
    
def collate_fn(batch):
    batch = sorted(batch, key=lambda x:x[3], reverse=True)#降序排序
    input, target, input_length, target_length = list(zip(*batch))
    input = torch.LongTensor([config.num_sequence.transform(i,config.max_len) for i in input])
    target = torch.LongTensor([config.num_sequence.transform(i,config.max_len+1) for i in target])
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)
    return input, target, input_length, target_length






train_dataloader = DataLoader(NumDataset(), batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)



if __name__ == "__main__":
    for input, target, input_length, target_length in train_dataloader:
        print("input:",input)
        print("target:",target)
        print("input_length:",input_length)
        print("target_length:",target_length)
        break