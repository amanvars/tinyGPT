import torch
from utils import read_text, train_val_split
from models import BigramLanguageModel


#variables
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_embd = 32
# head_size = 32
torch.manual_seed(1337)

def train_val_split(data, split=0.9):
    # split data into train and validation
    n = int(split*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


# read the data
with open('input.txt', 'r') as file:
    text = file.read()


# here all unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from character to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_data, val_data = train_val_split(data, 0.9)

# data loading
def get_batch(split_name):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split_name == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel(block_size, vocab_size, n_embd)
model = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on trains and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"steps {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))