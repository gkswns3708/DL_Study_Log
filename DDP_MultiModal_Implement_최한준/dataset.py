import os
import cv2
import torch
import torch.nn as nn

class Custom_Dataset(nn.Module):
    def __init__(self, text, image_path, cats1, cats2, cats3, tokenizer, feature_extractor, max_len, train):
        self.text = text
        self.image_path = image_path
        self.cats1 = cats1
        self.cats2 = cats2
        self.cats3 = cats3
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor # pretrained LLM (HuggingFace KoBERT)
        self.max_len = max_len # 실험을 통해 overview의 max_len을 포함하게 해야할 듯(?)
        self.train = train
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        image_path = os.path.join('../', str(self.image_path[index][2:]))
        image = cv2.imread(image_path)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation = True, # max_len 넘으면 자를거냐 (True)
            return_attention_mask = True, # Attention_mask return 할거냐 (True)
            return_tensors='pt' # Tensor로의 변환을 위해 pt 선택.  tf, np 옵션도 존재
        )
        image_feature = self.feature_extractor(images=image, return_tensors='pt')
        if self.train :
            cat1 = self.cats1[index] # Label1 
            cat2 = self.cats2[index] # Label2
            cat3 = self.cats3[index] # Label3
            return {
                'input_ids' : encoding['input_ids'].flatten(), # 이거 왜 flatten() ?
                'attention_mask' : encoding['attention_mask'].flatten(),
                'pixel_values': image_feature['pixel_values'][0],
                'cats1': torch.tensor(cat1, dtype=torch.long),
                'cats2': torch.tensor(cat2, dtype=torch.long),
                'cats3': torch.tensor(cat3, dtype=torch.long)
            } 
        else:
            return{
                'input_ids' : encoding['input_ids'].flatten(), # 이거 왜 flatten() ?
                'attention_mask' : encoding['attention_mask'].flatten(),
                'pixel_values': image_feature['pixel_values'][0],
            }
            

        