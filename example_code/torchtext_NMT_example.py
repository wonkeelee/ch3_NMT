from torchtext import data, datasets
import mosestokenizer
import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
from typing import Tuple


tokenizer_en = mosestokenizer.MosesTokenizer('en')
tokenizer_de = mosestokenizer.MosesTokenizer('de')

BOS = '<s>'
EOS = '</s>'
PAD = '<pad>'

src = data.Field(sequential=True,
                 use_vocab=True,
                 pad_token=PAD,
                 tokenize=tokenizer_en,
                 lower=True,
                 include_lengths=True,

                ) #unk=0, pad=1

tgt = data.Field(sequential=True,
                 use_vocab=True,
                 pad_token=PAD,
                 tokenize=tokenizer_de,
                 lower=True,
                 init_token=BOS,
                 eos_token=EOS,
                 include_lengths=True,

                 ) #unk=0, pad=1, <s>=2, </s>=3

prefix_f = './escape.en-de.tok.100k'

parallel_dataset = datasets.TranslationDataset(path=prefix_f, exts=('.en', '.de'), fields=[('src', src), ('tgt', tgt)])

src.build_vocab(parallel_dataset, min_freq=5, max_size=15000)
tgt.build_vocab(parallel_dataset, min_freq=5, max_size=15000)

train, valid = parallel_dataset.split(split_ratio=0.97)

train_iter, valid_iter = data.BucketIterator.splits((train, valid), batch_size=32,
                                                    sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)),
                                                    device='cuda')

class Encoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, pad_idx: int):
        super().__init__()

        self.dim = hidden_dim
        #self.input_dim = input_dim
        #self.emb_dim = emb_dim
        #self.enc_hid_dim = enc_hid_dim
        #self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(hidden_dim, hidden_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded) # output = (batch, len, num_direction * hidden), last_hidden = (layer * directions, batch, hidden)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) # last bidrectional hidden

        return outputs, hidden # enc_out, dec_init_hidden

class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return nn.Softmax(attention, dim=1)



# print(parallel_dataset)