class Num_sequence:
    PAD_TAG = "PAD"
    PAD = 0
    UNK_TAG = "UNK"
    UNK = 1
    SOS_TAG = "SOS"
    SOS = 2
    EOS_TAG = "EOS"
    EOS = 3


    def __init__(self):
        self.dict = {self.PAD_TAG:self.PAD, self.UNK_TAG:self.UNK, self.SOS_TAG:self.SOS, self.EOS_TAG:self.EOS}
        for i in range(10):
            self.dict[str(i)] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len, add_eos=False):
        # 把sentence转化为序列
        '''
        add_eos=True: 输出句子长度为max_len+1
        add_eos=False: 输出句子长度为max_len
        '''
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        sentence_len = len(sentence)
        if add_eos:
            sentence = sentence + [self.EOS_TAG]
        if sentence_len < max_len:
            sentence = sentence + (max_len - sentence_len)*[self.PAD_TAG]
        
        result = [self.dict.get(i,self.UNK) for i in sentence]
        return result
    
    def inverse_transform(self, incidence):
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in incidence]
    
    def __len__(self):
        return len(self.dict)



if __name__ == "__main__":
    num_sequence = Num_sequence()