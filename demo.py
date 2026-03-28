import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_block import Block


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

import re

# Read FMCG corpus from file
with open('fmcg_corpus.txt', 'r', encoding='utf-8') as f:
    # Lowercase text to reduce vocabulary size
    corpus = [line.strip().lower() for line in f if line.strip()]

text = " ".join(corpus)

# Basic tokenization separated into words and punctuation
tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
words = sorted(list(set(tokens)))

vocab_size = len(words)
print("Vocabulary size:", vocab_size)

word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()} 

data = torch.tensor([word2idx[w] for w in tokens], dtype=torch.long)
print("Total words in corpus:", len(data))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

block_size = 16  # Increased from 6 for better long-term context
embedding_dim = 64 # Increased representations
n_heads = 4  
n_layers = 4 
lr = 3e-4 # Better learning rate for transformers
epochs = 3000  
batch_size = 32

def get_batch(batch_size=batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x.to(device), y.to(device)




class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx



model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

model.train()
for step in range(1, epochs + 1):
    xb, yb = get_batch() 
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}/{epochs}, loss={loss.item():.4f}")


model.eval()

print("\n" + "="*40)
print("🤖 FMCG SLM Chatbot is Ready! 🤖")
print("Type 'exit' or 'quit' to close the chat.")
print("="*40 + "\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot ended. Bye!")
            break
        
        if not user_input.strip():
            continue

        # Tokenize user input
        user_tokens = re.findall(r'\b\w+\b|[.,!?;]', user_input.lower())
        
        # Filter out unknown words to prevent KeyErrors
        context_tokens = [word2idx[t] for t in user_tokens if t in word2idx]
        
        # If the user input contains entirely unknown words
        if not context_tokens:
            print("Bot: I didn't recognize any terms from my FMCG training. Try talking about detergent, shampoo, biscuits, etc.")
            continue
            
        context = torch.tensor([context_tokens], dtype=torch.long, device=device)

        # Generate the response
        with torch.no_grad():
            out = model.generate(context, max_new_tokens=25)
        
        # Extract just the newly generated tokens (optional: you can keep context too)
        # We'll just print the whole sequence to show how it completes your thought
        output_text = " ".join([idx2word[int(i)] for i in out[0]])
        
        generated = output_text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").capitalize()
        
        # Print bot output
        print(f"Bot: {generated}\n")

    except KeyboardInterrupt:
        print("\nChatbot ended. Bye!")
        break