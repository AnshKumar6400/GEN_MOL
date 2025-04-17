import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=None):
        self.smiles = df['smiles'].tolist()
        self.properties = df[['logP', 'qed', 'SAS']].values
        
        # Normalize properties
        self.scaler = StandardScaler()
        self.properties = self.scaler.fit_transform(self.properties)
        
        self.tokenizer = tokenizer
        self.max_len = max_len if max_len is not None else self._get_max_len()
        
    def _get_max_len(self):
        return max(len(smi) for smi in self.smiles) + 2  # +2 for SOS and EOS
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        props = torch.FloatTensor(self.properties[idx])
        tokens = self.tokenizer.encode(smiles, self.max_len)
        return tokens, props

class SMILESTokenizer:
    def __init__(self):
        # Basic SMILES tokens
        self.vocab = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3,
            'C': 4, 'O': 5, 'N': 6, '=': 7, '(': 8, ')': 9,
            '#': 10, '[': 11, ']': 12, '1': 13, '2': 14, 
            'H': 15, '-': 16, '3': 17, '4': 18, '5': 19,
            '6': 20, '7': 21, '8': 22, '9': 23, '0': 24,
            'c': 25, 'n': 26, 'o': 27, 's': 28, 'F': 29,
            'Cl': 30, 'Br': 31, 'I': 32, 'P': 33, 'B': 34
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_idx = self.vocab['<pad>']
        self.sos_idx = self.vocab['<sos>']
        self.eos_idx = self.vocab['<eos>']
        self.unk_idx = self.vocab['<unk>']
        
    def encode(self, smiles, max_len):
        tokens = [self.sos_idx]
        
        i = 0
        while i < len(smiles):
            if i+1 < len(smiles) and smiles[i:i+2] in self.vocab:
                tokens.append(self.vocab[smiles[i:i+2]])
                i += 2
            elif smiles[i] in self.vocab:
                tokens.append(self.vocab[smiles[i]])
                i += 1
            else:
                tokens.append(self.unk_idx)
                i += 1
        
        tokens.append(self.eos_idx)
        
        if len(tokens) < max_len:
            tokens += [self.pad_idx] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
            
        return torch.LongTensor(tokens)

class PropertyEncoder(nn.Module):
    def __init__(self, prop_dim, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(prop_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x

class BetaTCVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, 
                nhead, num_layers, pad_idx, prop_dim=3, device='cpu'):
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.prop_encoder = PropertyEncoder(prop_dim, embedding_dim)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim*2,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent space
        self.mu = nn.Linear(embedding_dim*2, latent_dim)
        self.log_var = nn.Linear(embedding_dim*2, latent_dim)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(self.device)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, src, props):
        # Embed SMILES
        src_embed = self.embedding(src)
        src_embed = self.pos_encoder(src_embed)
        
        # Embed properties
        prop_embed = self.prop_encoder(props).unsqueeze(1)
        prop_embed = prop_embed.expand(-1, src_embed.size(1), -1)
        
        # Combine
        combined = torch.cat([src_embed, prop_embed], dim=-1)
        
        # Padding mask
        src_key_padding_mask = (src == self.pad_idx)
        
        # Encode
        encoded = self.encoder(combined, src_key_padding_mask=src_key_padding_mask)
        
        # Latent
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        
        return mu, log_var
    
    def decode(self, z, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # Embed target
        tgt_embed = self.embedding(tgt)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # Decode
        decoded = self.decoder(
            tgt_embed, 
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Output
        output = self.fc_out(decoded)
        return output
    
    def forward(self, src, props, tgt=None):
        # Encode
        mu, log_var = self.encode(src, props)
        z = self.reparameterize(mu, log_var)
        
        # Prepare target
        if tgt is None:
            tgt = src
            
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
        tgt_key_padding_mask = (tgt == self.pad_idx)
        
        # Decode
        output = self.decode(z, self.embedding(src), tgt, tgt_mask, tgt_key_padding_mask)
        
        return output, mu, log_var

def loss_function(recon_x, x, mu, log_var, beta=1.0, gamma=0.1):
    # Reshape tensors to be contiguous
    recon_x = recon_x.contiguous().view(-1, recon_x.size(-1))
    x = x.contiguous().view(-1)
    
    CE = F.cross_entropy(
        recon_x,
        x,
        ignore_index=0,  # ignore padding
        reduction='mean'
    )
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    TC = (log_var.exp() - 1 - log_var).mean()
    return CE + beta * KLD + gamma * TC

def train_epoch(model, dataloader, optimizer, device, beta, gamma):
    model.train()
    total_loss = 0
    
    for src, props in dataloader:
        src = src.to(device)
        props = props.to(device)
        
        # Shift target for teacher forcing
        tgt = src[:, :-1]
        target = src[:, 1:]
        
        optimizer.zero_grad()
        
        output, mu, log_var = model(src, props, tgt)
        loss = loss_function(output, target, mu, log_var, beta, gamma)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, beta, gamma):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, props in dataloader:
            src = src.to(device)
            props = props.to(device)
            
            tgt = src[:, :-1]
            target = src[:, 1:]
            
            output, mu, log_var = model(src, props, tgt)
            loss = loss_function(output, target, mu, log_var, beta, gamma)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    df = pd.read_csv('zinc250k.csv')  # Update with your path
    
    # Initialize tokenizer
    tokenizer = SMILESTokenizer()
    
    # Create datasets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = SMILESDataset(train_df, tokenizer)
    val_dataset = SMILESDataset(val_df, tokenizer)
    
    # Dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model config
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 256
    hidden_dim = 512
    latent_dim = 128
    nhead = 8
    num_layers = 6
    prop_dim = 3
    
    # Initialize model
    model = BetaTCVAE(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        nhead=nhead,
        num_layers=num_layers,
        pad_idx=tokenizer.pad_idx,
        prop_dim=prop_dim,
        device=device
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training
    epochs = 50
    beta = 1.0
    gamma = 0.1
    
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, beta, gamma)
        val_loss = evaluate(model, val_loader, device, beta, gamma)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_zinc_vae.pth')
            print('Model saved!')
    
    print('Training complete!')

if __name__ == '__main__':
    main()