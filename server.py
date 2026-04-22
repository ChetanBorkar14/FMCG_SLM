from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os

from transformer_block import Block

app = Flask(__name__, static_folder=".")
CORS(app)

print("Reading FMCG corpus...")
with open('fmcg_corpus.txt', 'r', encoding='utf-8') as f:
    corpus = [line.strip().lower() for line in f if line.strip()]

text = " ".join(corpus)
tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
words = sorted(list(set(tokens)))
vocab_size = len(words)

word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()} 

data = torch.tensor([word2idx[w] for w in tokens], dtype=torch.long)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 16 
embedding_dim = 64 
n_heads = 4  
n_layers = 4 

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

def train_model():
    print(f"Training TinyGPT model on {device}... this may take a moment.")
    lr = 3e-4 
    epochs = 3000  
    batch_size = 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))  
        x = torch.stack([data[i:i+block_size] for i in ix])  
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
        return x.to(device), y.to(device)

    model.train()
    for step in range(1, epochs + 1):
        xb, yb = get_batch() 
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"Step {step}/{epochs}, loss={loss.item():.4f}")
    
    model.eval()
    print("Training Complete! Server is ready.")

# Ensure model is trained on startup
train_model()

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    req = request.json
    user_input = req.get('message', '')
    
    if not user_input.strip():
        return jsonify({'message': 'Please provide a message.'})

    user_tokens = re.findall(r'\b\w+\b|[.,!?;]', user_input.lower())
    context_tokens = [word2idx[t] for t in user_tokens if t in word2idx]
    
    if not context_tokens:
        return jsonify({'message': "I didn't recognize any terms from my FMCG training. Try talking about detergent, shampoo, biscuits, etc."})
        
    context = torch.tensor([context_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(context, max_new_tokens=25)
    
    output_text = " ".join([idx2word[int(i)] for i in out[0]])
    generated = output_text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").capitalize()
    
    return jsonify({'message': generated})

if __name__ == '__main__':
    print("Starting Flask web server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000)
