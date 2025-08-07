import torch
import torch.nn as nn
from torch.nn import functional as F
from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

# Global variables for model and mappings
model = None
device = None
stoi = None
itos = None
encode = None
decode = None
hyperparams = None

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, hyperparams['block_size'], dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_model(model_path='gpt_model_final.pth'):
    """Load the trained GPT model"""
    global model, device, stoi, itos, encode, decode, hyperparams
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract hyperparameters and vocab mappings
    hyperparams = checkpoint['hyperparameters']
    vocab_size = checkpoint['vocab_size']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Create encode/decode functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Initialize model
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=hyperparams['n_embd'],
        block_size=hyperparams['block_size'],
        n_head=hyperparams['n_head'],
        n_layer=hyperparams['n_layer'],
        dropout=hyperparams['dropout']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully! Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return True

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text from the model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500
        
        # Get request data
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = min(data.get('max_tokens', 100), 1000)  # Cap at 1000 tokens
        
        # Encode prompt
        if prompt:
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        # Generate text
        with torch.no_grad():
            generated_tokens = model.generate(context, max_new_tokens=max_tokens)
            generated_text = decode(generated_tokens[0].tolist())
        
        # If we had a prompt, remove it from the output
        if prompt:
            generated_text = generated_text[len(prompt):]
        
        return jsonify({
            'prompt': prompt,
            'generated_text': generated_text,
            'total_text': prompt + generated_text,
            'tokens_generated': max_tokens,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'hyperparameters': hyperparams,
        'vocab_size': len(stoi) if stoi else 0,
        'device': str(device),
        'parameter_count': f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M",
        'status': 'success'
    })

@app.route('/vocab', methods=['GET'])
def get_vocab():
    """Get vocabulary information"""
    if stoi is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'vocab_size': len(stoi),
        'sample_chars': list(stoi.keys())[:20],  # First 20 characters
        'status': 'success'
    })

if __name__ == '__main__':
    # Load model on startup
    model_path = 'gpt_model_final.pth'
    if os.path.exists(model_path):
        print("Loading model...")
        if load_model(model_path):
            print("Model loaded successfully!")
        else:
            print("Failed to load model!")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        print("The server will start but model endpoints won't work until model is loaded.")
    
    # Start Flask server
    print("Starting inference server...")
    app.run(host='0.0.0.0', port=5000, debug=False)