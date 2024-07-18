import torch
import torch.nn as nn
from torch.nn import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, block_size, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # here n_embd = C, which is dimension of key vector cf : transformers paper
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #self.register_buffer("tril") creates a read-only buffer, meaning the underlying matrix cannot be directly modified after creation.
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B, T, C)
        # compute attentions corre (affinities)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B,T,C) @ (B,T,C) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T)  (B,T,C) --> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, block_size, num_heads, n_embd, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # it will always return (B,T, n_embd) and n_embd = (head_size*num_heads)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a liner layer followed by non linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / (torch.sqrt(xvar + self.eps))
        x = self.gamma * xhat + self.beta
        return x


class Block(nn.Module):
    """Transformer block followed by computation"""
    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: no of head we'd like
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(block_size, n_head, n_embd, head_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, block_size, vocab_size, n_head, n_embd, n_layer, dropout):
        super().__init__()
        # each token directly reads off the logits for the nexct token froma lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd ) # second arg is basically any arbitrary length vec
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # # self.sa_head = Head(block_size, n_embd, head_size)
        # self.sa_heads = MultiHeadAttention(block_size, 4, n_embd, n_embd//4) #i.e 4 heads of 8-D self attention
        # self.ff_head = FeedForward(n_embd)
        # use Block class instead above two line and use Block to create multihead attention
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T,C)
        # x = self.sa_heads(x) # apply one head of self attention
        # x = self.ff_head(x) 
        x = self.blocks(x)
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