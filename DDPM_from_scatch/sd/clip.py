import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()
        
        self.token_embeddng = nn.Embeding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        
        x = self.token_embedding(tokens)
        
        x += self.position_embedding
        return x

class CLIPLayer(nn.Module):
    
    def __init__(self, n_head : int, n_embed: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(5 * n_embed, n_embed)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)
        
        residue = x
        
        ## SELF ATTENTION
        
        x = self.layernorm_1(x)
        
        x = self.attention(x, causal_mask=True) # casual_mask mean that every token cannot watch the next tokens so cannot be related to Future tokens by only the one the left.
        
        x += residue 
        
        ## FEEDFOWARD LAYER
        
        residue = x
        
        x = self.layernorm_2(x)
        
        x = self.linear_1(x)
        
        x = x * torch.signmoid(1.702 * x) # QuickGELU activation function # Author said that this works better than others.
        
        x = self.linear_2(x)
        
        x += residue
        
        return x
        

class CLIP(nn.Module):
    def __init__(self):
        # TODO: 여기서는 왜 super() 안함?
        # embedding을 이용해 word를 token으로 변환할 수 있습니ㅏㄷ.
        # 먼저 sentence의 word를 vocab을 이용해 matching되는 하나의 숫자로 변환하고(전처리가 된 sentence)
        # 각 embedding은 768크기를 가짐(Orignal Transformer는 512)
        self.embedding = CLIPEmbedding(49408, 768, 77) # Vocab_size, embedding_dim, Max_Seq_Len
        
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12) # (number of head, embedding_dim)
        ])
        
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        token = tokens.type(torch.long)
        
        # (Batch_size, Seq_Len) -> (Batch_Size, Seq_Len, Dim(768))
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        # (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output