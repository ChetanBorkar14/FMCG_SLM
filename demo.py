import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_block import Block

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Read FMCG corpus from file
with open('fmcg_corpus.txt', 'r', encoding='utf-8') as f:
    # Each line represents a single transaction/basket
    baskets = [line.strip().lower().split() for line in f if line.strip()]

# Add <EOS> (End of Sequence) token to denote transaction boundaries
# This helps the model understand when a basket is "complete"
EOS_TOKEN = "<EOS>"
data_tokens = []
for basket in baskets:
    data_tokens.extend(basket)
    data_tokens.append(EOS_TOKEN)

# Build vocabulary of unique products
products = sorted(list(set(data_tokens)))
vocab_size = len(products)
print("Vocabulary size (Number of unique products):", vocab_size)

word2idx = {w: i for i, w in enumerate(products)}
idx2word = {i: w for w, i in word2idx.items()}

# Convert dataset into indices
data = torch.tensor([word2idx[w] for w in data_tokens], dtype=torch.long)
print("Total items in sequence:", len(data))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

block_size = 8  # Reduced since typical baskets aren't very long
embedding_dim = 64 
n_heads = 4  
n_layers = 4 
lr = 1e-3 # Faster learning rate for small vocab
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
        # Original generation function for predicting sequence of events
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def recommend_top_items(self, idx, k=3, exclude_items=None):
        """
        Market Basket Analysis Recommendation:
        Given current basket tokens (idx), predict the top K most likely next products.
        """
        if exclude_items is None:
            exclude_items = []
            
        with torch.no_grad():
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Focus on the prediction after the last item in the basket
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)[0]
            
            # Mask out EOS token and items already in the basket
            probs[word2idx[EOS_TOKEN]] = 0.0
            for item_idx in exclude_items:
                probs[item_idx] = 0.0
                
            # Get top K products
            top_probs, top_indices = torch.topk(probs, k)
            return top_indices.tolist()

model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("Training model...")
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

print("\n" + "="*50)
print("🛒 Market Basket Analysis Recommender Ready! 🛒")
print("Enter the items currently in your basket (space separated).")
print("Type 'exit' or 'quit' to close the chat.")
print("="*50 + "\n")

while True:
    try:
        user_input = input("Current Basket: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot ended. Bye!")
            break
        
        if not user_input.strip():
            continue

        # Tokenize user input
        user_items = user_input.lower().split()
        
        # Filter unknown products
        context_tokens = [word2idx[t] for t in user_items if t in word2idx]
        
        if not context_tokens:
            print(f"Bot: I don't recognize those products. Known products include: {', '.join(products[:5])}...\n")
            continue
            
        # Display recognized items
        recognized_basket = [idx2word[t] for t in context_tokens]
        print(f"[Recognized items: {', '.join(recognized_basket)}]")

        context = torch.tensor([context_tokens], dtype=torch.long, device=device)

        # Generate top 3 recommendations
        recommended_indices = model.recommend_top_items(context, k=3, exclude_items=context_tokens)
        
        recommendations = [idx2word[i] for i in recommended_indices]
        
        print(f"Bot: Based on your basket, you might also like to buy:")
        for idx, rec in enumerate(recommendations, 1):
            print(f"  {idx}. {rec.capitalize()}")
        print()

    except KeyboardInterrupt:
        print("\nChatbot ended. Bye!")
        break