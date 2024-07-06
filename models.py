import torch
import torch.nn as nn
from torch.nn import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, block_size, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # here n_embd = C, which is dimension of key vector cf : transformers paper
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #self.register_buffer("tril") creates a read-only buffer, meaning the underlying matrix cannot be directly modified after creation.
        self.head_size = head_size

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B, T, C)
        # compute attentions corre (affinities)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B,T,C) @ (B,T,C) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T)  (B,T,C) --> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, block_size, num_heads, n_embd, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) # it will always return (B,T, n_embd) and n_embd = (head_size*num_heads)

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, block_size, vocab_size, n_embd):
        super().__init__()
        # each token directly reads off the logits for the nexct token froma lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd ) # second arg is basically any arbitrary length vec
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(block_size, n_embd, head_size)
        self.sa_heads = MultiHeadAttention(block_size, 4, n_embd, n_embd//4) #i.e 4 heads of 8-D self attention
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T,C)
        x = self.sa_heads(x) # apply one head of self attention
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the preds
            logits, loss = self(idx_cond)
            # focus olnly on the last time step
            logits = logits[:,-1, :] # becomes (B,C)
            # apply softmax to get prob
            probs = F.softmax(logits, dim=-1) # (B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to the running seq
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    
# m = BigramLanguageModel(vocab_size)
# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

# print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))