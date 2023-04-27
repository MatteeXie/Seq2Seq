from dataset import train_dataloader
from encoder import Encoder
from decoder import Decoder



encoder = Encoder()
decoder = Decoder()

print(encoder)
print(decoder)
for input, target, input_length, target_length in train_dataloader:
        out, encoder_hidden, _ = encoder(input, input_length)
        decoder(target, encoder_hidden)
        break
