import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
    def __init__(self):
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