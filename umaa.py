import torch
import torch.nn as nn
from position_encoding import PositionalEncoding 
import torch.nn.functional as F
from utils import *

class Encoder(nn.Module):
    def __init__(self, input_size, window_size, embed_dim, num_heads, dropout):
        super().__init__()

        self.multihead_attention1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward1 = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim / 2)),
            nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(embed_dim)
		
        self.multihead_attention2 = nn.MultiheadAttention(int(embed_dim/2), int(num_heads/2), dropout=dropout)
        self.feedforward2 = nn.Sequential(
            nn.Linear(int(embed_dim/2), int(embed_dim/4)),
            nn.ReLU()
        )
        self.norm2 = nn.LayerNorm(int(embed_dim/2))
		
        self.multihead_attention3 = nn.MultiheadAttention(int(embed_dim/4), int(num_heads/4), dropout=dropout)
        self.norm3 = nn.LayerNorm(int(embed_dim/4))
		
    def forward(self, x):
 
        # Self-attention
        attn_output, _ = self.multihead_attention1(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        # Feedforward
        x = self.feedforward1(x)

		
        # Self-attention
        attn_output, _ = self.multihead_attention2(x, x, x)
        x = x + attn_output
        x = self.norm2(x) 
        # Feedforward
        x = self.feedforward2(x)
 
		
        # Self-attention
        attn_output, _ = self.multihead_attention3(x, x, x)
        x = x + attn_output
        x = self.norm3(x)
		
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, window_size, embed_dim, num_heads, dropout):
        super().__init__()
		
        self.multihead_attention1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward1 = nn.Sequential(
            nn.Linear(int(embed_dim/2), embed_dim),
            nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(embed_dim)
		
        self.multihead_attention2 = nn.MultiheadAttention(int(embed_dim/2), int(num_heads/2), dropout=dropout)
        self.feedforward2 = nn.Sequential(
            nn.Linear(int(embed_dim/4), int(embed_dim/2)),
            nn.ReLU()
        )
        self.norm2 = nn.LayerNorm(int(embed_dim/2))
		
        self.multihead_attention3 = nn.MultiheadAttention(int(embed_dim/4), int(num_heads/4), dropout=dropout)
        self.norm3 = nn.LayerNorm(int(embed_dim/4))
		
    def forward(self, x):

        # Self-attention
        self_attn_output, _ = self.multihead_attention3(x, x, x)
        x = x + self_attn_output
        x = self.norm3(x)
		
        # Feedforward
        x = self.feedforward2(x)		
        # Self-attention
        self_attn_output, _ = self.multihead_attention2(x, x, x)
        x = x + self_attn_output
        x = self.norm2(x)
		
        # Feedforward
        x = self.feedforward1(x)		
        # Self-attention
        self_attn_output, _ = self.multihead_attention1(x, x, x)
        x = x + self_attn_output
        x = self.norm1(x)
		
        return x


class UMAA(nn.Module):
    def __init__(self, d_size, w_size, embed_dim, num_heads, dropout):
        super().__init__()
        self.lr = 0.0001
		
        self.pos_encoder = PositionalEncoding(w_size, d_size, n=10000, device='cuda')
        self.encoder = Encoder(d_size, w_size, embed_dim, num_heads, dropout)
        self.decoder1 = Decoder(d_size, w_size, embed_dim, num_heads, dropout)
        self.decoder2 = Decoder(d_size, w_size, embed_dim, num_heads, dropout)

		
    def forward(self, x):
        l = nn.MSELoss(reduction = 'none')
        x = x + self.pos_encoder(x)
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        return w3
		
    def training_step(self, batch, n): 
        l = nn.MSELoss(reduction = 'none')
        batch = batch + self.pos_encoder(batch)
        batch = batch.view([-1, batch.shape[1]*batch.shape[2]])

        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        l1 = torch.mean(loss1)
        l2 = torch.mean(loss2)
		
        return l1, l2

    def validation_step(self, batch, n):
        loss1, loss2 = self.training_step(batch, n)
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
			
			
def training(epochs, model, train_loader, val_loader, path, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
	
    early_stopping = EarlyStopping(patience=7, verbose=False, path=path)
	
    for epoch in range(epochs):
        model_loss = 0
		
        for [batch] in train_loader:
            batch=to_device(batch,get_default_device())
            
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            model_loss += loss1.item() + loss2.item()

        early_stopping(model_loss, model)
        if early_stopping.early_stop:
            break
    return history
	
def evaluate(model, val_loader, n):

    with torch.no_grad():
        model.eval()
        outputs = [model.validation_step(to_device(batch, get_default_device()), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)
	
def testing(model, test_loader, alpha=.5, beta=.5):
    results=[]
    with torch.no_grad():
        model.eval()
        for [batch] in test_loader:
            batch = batch.view([-1, batch.shape[1]*batch.shape[2]])
            batch=to_device(batch,get_default_device())
            w1=model.decoder1(model.encoder(batch))
            w2=model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results
