# GoogleNet_Implement

## Dataset
Animals - 10 [Link](https://www.kaggle.com/datasets/alessiocorrado99/animals10)


## Layers
### ConvBlock
```
# ConvBlock Layer Implement
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p):
        super(ConvBlock, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size = (k, k), stride=(s,s), padding=(p,p)),
            nn.BatchNorm2d(out_channel, eps=0.001),
            nn.ReLU()
        )
    
    def forward(self, input):
        output = self.convolution(input)
        return output
```

### Inception Module
```# Inception Module Implement
class InceptionModule(nn.Module):
    def __init__(self, in_channel,  out_channel_1x1, out_channel_3x3, out_channel_reduce_3x3, out_channel_5x5, out_channel_reduce_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvBlock(in_channel, out_channel_1x1, 1, 1, 0)
        self.conv3 = ReduceConvBlock(in_channel, out_channel_reduce_3x3, out_channel_3x3, 3, 1)
        self.conv5 = ReduceConvBlock(in_channel, out_channel_reduce_5x5, out_channel_5x5, 5, 2)

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)), # (1,1)
            nn.Conv2d(in_channel, pool_proj, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU()
        )

    
    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv3(input)
        out3 = self.conv5(input)
        out4 = self.pool_proj(input)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out 
```
![GoogleNet InceptionModule](https://github.com/gkswns3708/Paper-Implement/blob/main/GoogleNet_Implment/images/Inception%20Module.png)

### AuxClassifier
```
class AuxClassifier(nn.Module):
    def __init__(self, in_channel, num_classes=1000):
        super(AuxClassifier, self).__init__()
        self.AveragePool = nn.AvgPool2d(kernel_size=(5,5), stride=(3,3)) # padding?
        self.Conv = nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=(1,1), stride=(1,1))
        self.FC1 = nn.Linear(4*4*128, 1024)
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(p=0.1) # Someone Implement with p = 0.7 TODO : Why?
        self.Classifier = nn.Linear(1024, num_classes)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.AveragePool(input)
        out = self.Conv(out)
        out = self.ReLU(out)
        out = out.reshape(batch_size, -1)
        out = self.FC1(out)
        out = self.Dropout(out)
        out = self.Classifier(out)
        # 여기서 Softmax를 하지 않는 이유는 이후 Loss 계산시에 CEL을 사용하는데
        # 해당 Loss 계산시에 LogSoftmax(일종의 SoftmaxActivation)를 취하기 때문.

        return out
```


### Model Architecture
```
class GoogleNet(nn.Module):
    def __init__(self, in_channel=3, num_class=1000, training=True): # R,G,B + ImageNet Label Class
        super(GoogleNet, self).__init__()
        self.training = training

        self.Conv1 = ConvBlock(in_channel, 64, 7, 2, 3) 
        self.MaxPool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.LocalRespNorm1 = nn.LocalResponseNorm(size=64)

        self.Conv2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0), # Reduced Convolution (1x1)
            ConvBlock(64, 192, 3, 1, 1) # Normal Convolution (3x3)
        )
        self.MaxPool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)) # padding은 ceil_mode임

        self.LocalRespNorm2 = nn.LocalResponseNorm(size=192)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)     # 64 + 128 + 32 + 32 = 256 (Next Layer Input channel size : 256)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)   # 128 + 192 + 96 + 64 = 480 (Next Layer Input channel size : 480)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)) # PyTorch에서의 padding은 ceil_mode임

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.AuxClassifier1 = AuxClassifier(512, num_class)

        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)) 

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.AuxClassifier2 = AuxClassifier(528, num_class)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_class) # 여기 왜 1024 * 7 * 7??
        )
    def _forward(self, input):

        Batch_size = input.shape[0]
        x = self.Conv1(input)
        x = self.MaxPool1(x)
        # x = self.LocalRespNorm1(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.MaxPool3(x)
        x = self.inception4a(x)
        aux1: Optional[Tensor] = None
        if self.AuxClassifier1 is not None:
            if self.training:
                aux1 = self.AuxClassifier1(x)

        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2: Optional[Tensor] = None
        if self.AuxClassifier1 is not None:
            if self.training:
                aux2 = self.AuxClassifier2(x)
        
        x = self.inception4e(x)
        x = self.MaxPool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.training == False:
            assert aux1 is None
            assert aux2 is None

        return x, aux1, aux2

    def forward(self, input):
        x, aux1, aux2 = self._forward(input)
        aux_defined = self.training 
        if aux_defined:
            return GoogLeNetOutputs(x, aux1, aux2)
        else:
            return x
    
    def val_mode(self, Training=True):
        self.training = Training
```
![GoogleNet Architecture](https://github.com/gkswns3708/Paper-Implement/blob/main/GoogleNet_Implment/images/Model%20Architecture.png)
