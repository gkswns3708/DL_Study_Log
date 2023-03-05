import torch 
import os

import torch.nn as nn

from transformers import AutoModel,ViTModel,ViTFeatureExtractor

class TourClassifier(nn.Module):
    def __init__(self, n_classes1, n_classes2, n_classes3, text_model_name, image_model_name, args):
        super(TourClassifier, self).__init__()
        self.device = int(os.environ["LOCAL_RANK"]) if args.train else 'cuda:0'
        self.text_model = AutoModel.from_pretrained(text_model_name).to(self.device)
        self.image_model = AutoModel.from_pretrained(image_model_name).to(self.device)

        # Memory를 적게 쓰는 대신 시간이 오래걸릴 때 사용하는 테크닉이라고 함. 
        # 시간이 25% 늘어나는 대신 메모리가 60% 적게 사용됨
        # self.text_model.gradient_checkpointing_enable()  
        # self.image_model.gradient_checkpointing_enable()    

        self.drop = nn.Dropout(p=0.1)

        def get_cls(target_size):
            return nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
                nn.LayerNorm(self.text_model.config.hidden_size),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, target_size),
            )
        self.cls1 = get_cls(n_classes1)
        self.cls2 = get_cls(n_classes2)
        self.cls3 = get_cls(n_classes3)
    
    def forward(self, input_ids, attention_mask, pixel_values):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        image_output = self.image_model(pixel_values=pixel_values)
        # print(text_output.last_hidden_state.size(), image_output.last_hidden_state.size())
        concat_outputs = torch.cat([text_output.last_hidden_state, image_output.last_hidden_state], 1) 
        # config hidden size 일치해야함.
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=4).to(self.device)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(self.device)

        outputs = transformer_encoder(concat_outputs)
        outputs = outputs[:, 0]
        output = self.drop(outputs)
        
        out1 = self.cls1(output)
        out2 = self.cls2(output)
        out3 = self.cls3(output)

        return out1, out2, out3 # class(cat1,2,3)
