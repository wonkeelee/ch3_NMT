from torchtext.data import Field, BucketIterator, interleave_keys
from torchtext.datasets import TranslationDataset
from torchtext.data import Example
import mosestokenizer
import torch

from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


tokenizer_en = mosestokenizer.MosesTokenizer('en')
tokenizer_fr = mosestokenizer.MosesTokenizer('fr')


#src = Field(sequential=True,
#            use_vocab=True,
#            pad_token=PAD,
#            #tokenize=tokenizer_en,
#            lower=True,
#            batch_first=True)


BOS = '<s>'
EOS = '</s>'
PAD = '<pad>'

src = Field(sequential=True,
            use_vocab=True,
            pad_token=PAD,
            tokenize=tokenizer_en,
            lower=True,
            batch_first=True)

tgt = Field(sequential=True,
            use_vocab=True,
            pad_token=PAD,
            tokenize=tokenizer_fr,
            lower=True,
            init_token=BOS,
            eos_token=EOS,
            batch_first=True)

# prefix_f = 'data/escape.en-de.tok.50k'
prefix_f = 'data/data'
parallel_dataset = TranslationDataset(path=prefix_f, exts=('.en', '.fr'), fields=[('src', src), ('tgt', tgt)])

src.build_vocab(parallel_dataset, max_size=15000)
tgt.build_vocab(parallel_dataset, max_size=15000)

train, valid = parallel_dataset.split(split_ratio=0.97)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 24

train_iterator, valid_iterator = BucketIterator.splits((train, valid), batch_size=BATCH_SIZE,
                                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.tgt)),
                                                    device=device)



class Encoder(nn.Module):
    def __init__(self, hidden_dim: int, src_ntoken: int, dropout: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.src_ntoken = src_ntoken

        self.embedding = nn.Embedding(src_ntoken, hidden_dim, padding_idx=src.vocab.stoi['<pad>'])
        self.rnn = nn.GRU(hidden_dim, hidden_dim, bidirectional = True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src)) # [B, L, D]

        # outputs: [B, L, D*2], hidden: [2, B, D]
        # Note: if bidirectional=False then [B, L, D], [1, B, D]
        outputs, hidden = self.rnn(embedded)

        last_hidden = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) # [B, D]
        hidden = torch.tanh(last_hidden).unsqueeze(0) # last bidirectional hidden

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        attn_in = (enc_hid_dim * 2) + dec_hid_dim # bidirectional hidden + dec_hidden
        self.linear = nn.Linear(attn_in, attn_dim)
        self.merge = nn.Linear(attn_dim, 1)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[1]
        repeated_decoder_hidden = decoder_hidden.repeat(1, src_len, 1) # [B, src_len, D]

        energy = torch.tanh(self.linear(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2))) # enc의 각 step의 hidden + decoder의 hidden 의 결과값 # [B, src_len, D*3] --> [B, src_len, D]

        score = self.merge(energy).squeeze(-1) # [B, src_len] 각 src 단어에 대한 점수
        normalized_score = F.softmax(score, dim=1)
        return  normalized_score

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, dec_ntoken: int, dropout: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention = Attention(enc_hid_dim=hidden_dim, dec_hid_dim=hidden_dim, attn_dim=hidden_dim) # attn module
        self.dec_ntoken = dec_ntoken # vocab_size

        self.embedding = nn.Embedding(dec_ntoken, hidden_dim, padding_idx=tgt.vocab.stoi['<pad>'])
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_dim*3, dec_ntoken)
        self.sm = nn.LogSoftmax(dim=-1)

    def _context_rep(self, dec_out: Tensor, enc_outs: Tensor) -> Tensor:

        scores = self.attention(dec_out, enc_outs) # [B, L]
        scores = scores.unsqueeze(1) # [B, 1, src_len] -> weight value (softmax)

        # scores: (batch, 1, src_len),  ecn_outs: (Batch, src_len, dim)
        context_vector = torch.bmm(scores, enc_outs) # weighted average -> (batch, 1, dec_dim)
        return context_vector

    def forward(self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor]:

        dec_outs = []

        embedded = self.dropout(self.embedding(input)) # [B, L, D]
        # decoder_hidden = decoder_hidden.unsqueeze(0)
        for emb_t in embedded.split(1, dim=1): # Batch의 각 time step (=각 단어) 에 대한 embedding 출력
            rnn_out, decoder_hidden = self.rnn(emb_t, decoder_hidden) # feed input with previous decoder hidden at each step

            context = self._context_rep(rnn_out, encoder_outputs)
            rnn_context = self.dropout(torch.cat([rnn_out, context], dim=2))
            dec_out = self.linear(rnn_context)
            dec_outs += [self.sm(dec_out)]

        if len(dec_outs) > 1:
            dec_outs = dec_outs[:-1] # trg = trg[:-1] # <E> 는 Decoder 입력으로 고려하지 않음.
            dec_outs = torch.cat(dec_outs, dim=1) # convert list to tensor #[B, L, vocab]
        else: # step-wise 로 decoding 하는 경우,
            dec_outs = dec_outs[0] # [B=1, L=1, vocab]
        return dec_outs, decoder_hidden

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:

        encoder_outputs, hidden = self.encoder(src)
        dec_out, _ = self.decoder(trg, hidden, encoder_outputs)
        return dec_out


INPUT_DIM = len(src.vocab)
OUTPUT_DIM = len(tgt.vocab)
EMB_DIM = 128
HID_DIM = 128
D_OUT = 0.1


encoder = Encoder(HID_DIM, INPUT_DIM, D_OUT)
decoder = Decoder(HID_DIM, OUTPUT_DIM, D_OUT)

model = Seq2Seq(encoder, decoder, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            param.data.uniform_(-0.1, 0.1)
            #nn.init.uniform_(param.data, mean=0, std=0.01)
        else:
            param.data.zero_()
            #nn.init.constant_(param.data, 0)


model.apply(init_weights) # 모델 파라미터 초기화
optimizer = optim.Adam(model.parameters(), lr=0.0005) # Optimizer 설정
criterion = nn.NLLLoss(ignore_index=tgt.vocab.stoi['<pad>'], reduction='mean') # LOSS 설정

def count_parameters(model: nn.Module):
    print(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


import math
import time

def train(model: nn.Module, iterator: BucketIterator,
          optimizer: optim.Optimizer, criterion: nn.Module, clip: float):

    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.tgt

        optimizer.zero_grad()

        output = model(src, tgt)
        output = output.view(-1, output.size(-1)) # flatten (batch * length, vocab_size)

        tgt = tgt.unsqueeze(-1)[:,1:,:].squeeze(-1).contiguous() # remove <S> placed at first from targets
        tgt = tgt.view(-1) # flatten target with shape = (batch * length)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

        if(((i+1) % int(len(iterator)*0.2)) == 0):
            num_complete = batch.batch_size * (i+1)
            total_size = batch.batch_size * int(len(iterator))
            ratio = num_complete/total_size * 100
            print('| Current Epoch:  {:>4d} / {:<5d} ({:2d}%) | Train Loss: {:3.3f}'.
                  format(num_complete, batch.batch_size * int(len(iterator)), round(ratio), loss.item())
                  )

    return epoch_loss / len(iterator)

def evaluate(model: nn.Module, iterator: BucketIterator,
             criterion: nn.Module):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt

            output = model(src, tgt)
            output = output.view(-1, output.size(-1)) # flatten (batch * length, vocab_size)

            tgt = tgt.unsqueeze(-1)[:,1:,:].squeeze(-1).contiguous() # remove <S> placed at first from targets
            tgt = tgt.view(-1) # flatten target with shape = (batch * length)
            loss = criterion(output, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def greedy_decoding(model: nn.Module, input, fields, maxLen=20):
    src_field = [('src', fields[0])]
    tgt_field = fields[1]

    ex = Example.fromlist([input], src_field)
    src_tensor = src.numericalize([ex.src], device)
    tgt_tensor = torch.tensor([[tgt.vocab.stoi['<s>']]], device=device)
    model.eval()

    dec_result = []

    with torch.no_grad():
        enc_out, hidden = model.encoder(src_tensor)
        for i in range(maxLen):
            dec_step, hidden = model.decoder(tgt_tensor, hidden, enc_out)
            _, top_idx = torch.topk(dec_step, 1)
            if tgt_field.vocab.itos[top_idx] == '</s>':
                break
            else:
                dec_result.append(top_idx.item())
                tgt_tensor = top_idx.view(1, 1)

    dec_result = [tgt_field.vocab.itos[w] for w in dec_result]
    return dec_result


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 20
CLIP = 0.1
isTrain = True

if isTrain:
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('='*65)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print('='*65)

    with open('model.pt', 'wb') as f:
        print('save model at ./model.pt')
        torch.save(model, f)

else:
    with open('model.pt', 'rb') as f:
        model = torch.load(f).to(device)

output = " ".join(greedy_decoding(model, 'She is beautiful .', fields=(src, tgt)))
print(output)