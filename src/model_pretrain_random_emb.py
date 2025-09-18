import torch
import torch.nn as nn
import torch.nn.functional as F 

# Create a transformer model for masked word prediction
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, N_encoder_layers, seq_len, device, nodes, entity, relation_mask_index, mask_token_id=None, pos_emb=None):
        super(TransformerModel, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.nodes = nodes
        self.entity = entity
        self.relation_mask_index = relation_mask_index
        self.embedding = nn.Embedding(vocab_size, d_model).to(device)  # Embedding layer for vocab + mask token


        self.pos_emb = pos_emb # positional embedding
        if pos_emb == 'fixed':
            self.pos_encoding = self.create_positional_encoding(seq_len, d_model).to(device)    # Positional encoding (computed dynamically)
        elif pos_emb == 'learnable':
            self.pos_embedding = nn.Embedding(seq_len, d_model).to(device)
        
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model).to(device), 
        #     num_layers = N_encoder_layers
        # )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, activation='gelu').to(device),
            num_layers = N_encoder_layers
            )

        
        self.fc = nn.Linear(d_model, vocab_size+1).to(device)  # Output layer: Predict all vocab including [MASK]
        
        # binary classification layer 
        self.classifire = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # considering only head and tail for LP unless it should be d_model*3
            nn.ReLU(),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model//3),
            nn.ReLU(),
            nn.Linear(d_model//3, 1)
        ).to(device)

        
    def forward(self, x):  
        x = self.compute_embedding(x) + self.get_x_pos_emb(x)
        x = self.transformer(x.permute(1, 0, 2))    
        x = self.fc(x.permute(1, 0, 2))
        return x

    def classify(self, x):   
        x = self.compute_embedding(x) + self.get_x_pos_emb(x)
        x = self.transformer(x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        # x = torch.concat([x[:,0,:], x[:,1,:], x[:,2,:]], dim=1)
        # x = x[:, :3, :]              # shape: [batch, 3, hidden_dim]
        x = x[:, 0:3:2, :]              # shape: [batch, 2, hidden_dim]    # only considering head and tail for LP
        x = x.reshape(x.size(0), -1) # shape: [batch, 2 * hidden_dim]
        x = self.classifire(x)
        return x

    
    # when we use node attr we generate embedding here with trainable parameters 
    def compute_embedding(self, tokens):
        x = self.embedding(tokens) 
        return x

    def create_positional_encoding(self, seq_len, d_model):
        """Generates positional encoding dynamically"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def get_x_pos_emb(self, x):
        if self.pos_emb == 'fixed':
            x_pos_emb = self.pos_encoding[:x.shape[1], :]
        elif self.pos_emb == 'learnable':
            positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
            x_pos_emb = self.pos_embedding(positions)
        else:
            x_pos_emb = 0
        return x_pos_emb


def create_mask(data, vocab_size, mask_token_id, mask_prob=0.2, min_mask = 4):
    masked_data = data.clone()
    mask = torch.rand(data.shape) < mask_prob  
    
    for i in range(masked_data.shape[0]):
        if mask[i].sum() < min_mask:
            mask[i] = torch.zeros(data.shape[1], dtype=torch.bool)
            true_indices = torch.randperm(data.shape[1])[:min_mask]    
            mask[i, true_indices] = True
        elif mask[i].sum() == data.shape[1]:                           ########## this was previously if
            mask[i, 0] = False
    masked_data[mask] = mask_token_id  # assign number of mask_token_id to the masks
    return masked_data, mask


def get_probs(context_idx, model, mask_token, seq_len, query=None, late_softmax=False):

    len_context = len(context_idx)
    assert len_context < seq_len
    
    context_idx = context_idx + [mask_token] * (seq_len - len_context)   # extend context with mask tokens
    context_idx = torch.tensor(context_idx).reshape(1, -1).to(model.device)
    out = model(context_idx)[0, len_context, :]                          # get the first masked token
    if not late_softmax:
        out = F.softmax(out, dim=0)
    if query != None:
        out = out[query]
    if late_softmax:
        out = F.softmax(out, dim=0)
    return out

##########################

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))




