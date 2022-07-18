# Transformer 논문 구현

---

---

# 0. Requirements & Import

```python
pip install sentencepiece
```

```python
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import json
from tqdm import tqdm

import sentencepiece as spm
```

---

# 1. Vocab

### **0. Tokenizer - 추가 예정**

추가 예정

## 1. Vocab Load

- Tokenizer도 중요한 내용입니다만, 일단은 이후에 다시 보충하겠습니다.
- 해당 구현 코드에서는 SentencePieceProcessor를 사용했습니다.

이때 Input은 vocab의 ID(index)로 변환된 채 저장되었고, 이후 

`torch.nn.utils.rnn.pad_sequence`에 의해 가장 긴 seqence(token sequence)와 길이가 동일해질 수 있도록 padding을 추가했습니다.

(가로(sequence 길이) * 세로(sentence 갯수)인 2차원 배열을 생각하면 쉬울 듯 합니다.)

```python
# vocab loading
vocab_file = "/content/drive/MyDrive/Colab_Notebooks/Paper_Implement/NLP Basic/kowiki_32000.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

# 입력 texts
lines = [
  "겨울은 너무 기나긴 날들의 연속이였습니다.",
  "감기 조심하세요."
]

# text를 tensor로 변환
inputs = []
for line in lines:
  pieces = vocab.encode_as_pieces(line)
  ids = vocab.encode_as_ids(line)
  inputs.append(torch.tensor(ids))
  print(pieces)

# 입력 길이가 다르므로 입력 최대 길이에 맟춰 padding(0)을 추가 해 줌
inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)  # padding 추가해주는 method
# shape
print(inputs.size())
# 값
print(inputs)
```

## 2. Vocab Embedding

Transformer의 Embedding은 `Input Embedding`과 `Positional Embedding` 두 가지를 합해서 즉 

### 1. Input Embedding

먼저 단어를 얼마나 Embedding(size)할 것인지 결정을 해야합니다. 

- Hyperparmeter로써 작용하고, 직관적으로 Embedding size가 커지면, model이 커질것이고, 연산이 느려질 것입니다.
- 이후 `nn.Embedding` 객체를 생성해 전체 단어에 대한 Embedding 객체를 생성합니다.
    
    ```python
    nn_emb = nn.Embedding(n_vocab, d_hidn) # embedding 객체
    ```
    
- 이후 입력으로 넣고 싶은 Input을 `nn_emb`를 통해 embedding합니다.
    
    ```python
    input_embs = nn_emb(inputs) # input embedding
    print(input_embs.size())
    # --- 출력 결과 ---
    torch.Size([2, 13, 128])
    # 2개의 문장, 최대 길이 13개의 Token Sequence, 128개의 Embedding 차원
    ```
    

### 2. Positional Encoding

- Positional Encoding은 무엇인가?
    - 저와 동일한 생각을 가지고 작성한 글이 있어, 발췌해왔습니다.
        - 먼저 단순히 **“위치 정보”**를 표현하기에는 sin, cos과 같은 함수가 아니라, 1,2,3…같은 정수가  크기가 작은 소수로 표현하는게 좋지 않을까 하는 고민이였습니다.
        - [https://pongdangstory.tistory.com/482](https://pongdangstory.tistory.com/482)
        - [https://hongl.tistory.com/231](https://hongl.tistory.com/231) → 조금 더 자세히 설명되어 있습니다!
    - 문제점
        1. 정수값으로 위치 정보를 표현하면 해당 값이 모델에 끼치는 정보가 지나치게 커질 수 있습니다.
        2. 모델의 강건성의 관점에서 별로 좋지 못한 접근법이라고도 합니다.
        3. 또한 일정 길이의 토큰을 가지고 활용할 결우, 해당 길이를 넘는 Test set에 대해서 취약한 결과가 나올 수도 있습니다.
    - `Positional Embedding`의 조건
        1. 각각의 고유한 위치값은 다른 문장이 오더라고 “유일한” 값”을 가져야합니다.
        2. 서로 다른 길이의 시퀀스에 대해서 적용이 가능해야 합니다.
        3. 서로 다른 길이의 스퀀스의 두 index 간 거리는 일정해야 합니다.
    
- Positional Encoding 예시
    - 1. Simple Indexing
        
        ![Untitled](Images/Untitled.png)
        
        앞서 언급한 것처럼  가장 간단한 것은 Indexing일 겁니다. 하지만 Sequence의 길이가 길어지면 그에 따라 인덱스도 커지기 때문에, gradient explode와 같은 현상이 Training을 매우 불안정하게 만듭니다.
        
    - 2. Normalize simple indexing
        
        ![Untitled](images/Untitled%201.png)
        
        인덱스 자체가 매우 커지는 것이 문제라면 0부터 Sequence 크기까지 일정하게 증가하는 것도 괜찮아 보입니다.
        
        하지만 이는 서로 다른 Seqeunce의 같은 Index에 대해서 다른값이 나옵니다.
        
        - ex:
            - sequence length : 11, Index 3 : **0.3**
            - sequence length : 22, Index 3 : **0.15**
            
    - 3. Using binary
        
        인덱스 값 자체를 이진수로 표현하면 어떨까요? 크기 문제도 자유롭고, Normalize도 시켜줄 필요가 없습니다. 대신 각 Index마다 어떤 값이 들어가야 하는지, 정해줘야 합니다. 이 방법은 굉장히 discrete합니다. 이것은 Interpolation이라는 개념 없이 Sequence적인 요소를 배제한 값입니다.
        
        결과적으로 이진 벡터를 연속적인 함수로 부터 discretization 결과로서 얻는 방법이 필요합니다.
        
    - 4. Continuous binary vector - 이해 x
        
        각 시퀀스의 위치를 표현하는 $d_{model}$차원의 encoding 벡터를 이었을 때 그것이 연속적인 함수로 표현되면 좋을 것 같습니다. 또한 [0, 1]을 반복적으로 왔다갔다하는 함수가 필요한데, 이것은 Sin, Cos함수입니다. 또한 Sin, Cos 모두 Normalization 까지 준비되어 있습니다.
        
        ![Untitled](images/Untitled%202.png)
        
        1번째 다이얼은 0, 1
        
        2번째 다이얼은 0, 1, 2, 3
        
        3번째 다이얼은 0, 1, 2, 3, 4, 5, 6, 7
        
        …
        
        $N$번째 다이얼은 0, 1, 2, …, $2^N -1$
        
        …
        
        $d_{model}$번째 다이얼은 0, 1, 2, …, $d_{model} - 1$ 
        
        이렇게 표현할 수 있다면 해당 모델은 $2^{d_{model}}$가짓수의 Embedding Sequence를 표현할 수 있습니다.
        
        이를 수학적으로 표현하면 1스텝을 단위로 0, 1이 바뀌니 
        
        $$
        sin(\pi i/2^j) = sin(x_iw_j)
        $$
        
        로 표현 가능합니다.
        
        $x_i$는 시퀀스의 위치 인덱스이고, $j$는 임베딩 차원으로 $d_{model}$까지의 값을 가집니다. ($w_j$는 $j$에 해당하는 주파수를 나타냅니다.) 하지만 문제가 두가지 있습니다. 첫번째 문제는 Figure 5처럼 위 식을 따라 주기가 $\pi/2, \pi/4, \pi/8$에 대한 위치에 따른 3차원 encoding 결과를 나타낸 것으로 시퀀스 인덱스가 계속 움직이면서 다시 처음 인덱스로 돌아가는 닫힌 구조로 되어 있습니다.
        
        ![Untitled](images/Untitled%203.png)
        
        생각해보면 positional encoding 결과가 이진 벡터일 필요는 없습니다. (Normalize가 되어있으면 되지만 그 값이 0, 1일 필요는 없다는 뜻입니다). 그래서 “굳이” 주기가 $\pi/2$로부터 파생될 필요가 없고 $j$에 따라서 주파수가 감소하면 됩니다. 따라서 다음과 같이 구성하고 Transformer에서는 최저 frquency $w_0 = 1/10000$으로 $j$가 증가할 때마다 주파수가 단조적으로 감소합니다.
        
        $$
        sin(x_iw_0^{j/d_{model}})
        $$
        
         두 번째 고칠점은 인덱스 $i$의 positional encoding 벡터가 인덱스 $j$의 positonal encoding 벡터의 함수로 표현되게 하고 싶은 positional encoding translation을 반영하는 것입니다. 특히, Transformer에서는 self-attention 기법을 통해 각 위치 별 다른 모든 위치에 대한 attention score를 계산한다는 점에서 각 인덱스 별 positional encoding 벡터가 다른 인덱스 벡터의 선형변환으로 표현할 수 있게 하는 점이 중요합니다.
        
    - **5. Use both Sin and Cos - 이해 x**
        
        $$
        \text{Positional Encoding} = \left ( \begin{array}{}v^{(0)} \\ \vdots \\ v^{(seqlen - 1)} \end{array} \right)
        $$
        
        위 처럼 각 위치면 positional encoding 벡터가 쌓은 PE 행렬이 구성되고, 
        $v^i = [sin(w_0x_i), ..., sin(w_{n-1}x_i)]$라 하면 인덱스 벡터간 선형변환을 위해서 $PE(x + \nabla x) = PE(x) \cdot T(\nabla x)$를 만족하는 선형변환 $T$로 구성해야만 합니다.
        
        이것은 위치 $x$가 실질적으로 sin 함수의 각도로 들어가기 때문에 아래의 수식과 같이 cos/sin 함수의 각도에 따른 rotation 법칙을 사용하면 해결할 수 있습니다.
        
        ![Untitled](images/Untitled%204.png)
        
        먼저 각 $v^i$에 대해 sin 함수를 cos 함수로 대체한 벡터를 구성합니다. 이후에 새로운 $PE$행렬을 sin 함수와 cos함수를 번갈아가면서 다음과 같이 구성합니다.
        
        $$
        v^i = [cos(w_0x_i), sin(w_0, x_i), ..., cos(w_{n-1}x_i), sin(w_{n-1}x_i)
        $$
        
        이제 아래의 수식과 같이 선형 변환을 위한 $T$를 구성할 수 있습니다. 위의 식의 $\theta, \phi$를 각각 인덱스와 인덱스 오프셋이라 생각할 수 있고 예를 들어서 $k$번째 dial에서 $dx$만큼 이동시키려면 $w_k\cdot dx$만큼의 각도 이동이 필요합니다. 즉, cos 함수를 sin 함수와 번갈아가며 사용함으로써 각 인덱스 별 positonal encoding 벡터 간의 선형변환이 가능해지고, 이는 추후 attention layer에서 선형 변환을 통해 key, query를 구성할 때 도움이 되고, Transformer에서 사용되었던 positional encoding 행렬 구성이 완료되었음을 알 수 있습니다.
        
        ![Untitled](images/Untitled%205.png)
        
        6. Wrapup
        
        positional encoding 특성은 다음과 같습니다.
        
        1. [시퀀스 길이(보통 최대 길이 or padding을 포함한 $2^n$형태의 길이, Embedding dimension] 크기의 텐서로 구성됩니다.
        2. PE 행렬의 각 행은 sin/cos 함수에서 각 위치별로 interpolate 된 벡터입니다.
        3. PE 행렬의 행은 sin/cos 함수가 번갈아가면서 구성되며 열은 임베딩 차원이 커지면서 주파수가 감소합니다.
        4. **임의의 두 행에 대해서 선형변환이 가능한 행렬이 존재합니다.**
            
            ![Untitled](images/Untitled%206.png)
            
    - Concluson
        
        시퀀스 정보를 담은 Embedding 위치 정보를 담은 Embedding을 이어 붙이지 않고(concatenate) 더하는 이유에 대해서는 명확히 정리된 것은 없음. 하지만 더하는 연산으로 인해 공간상으로 이득이 있고,
        
    
- Code
    
    ```python
    """sinusoid position embedding"""
    def get_sinusoid_encoding_table(n_seq, d_hidn):
        def cal_angle(position, i_hidn):
            return position / np.power(10000, 2 * (i_hidn//2) / d_hidn) # d_hidn 값까지 순차적으로 들어와 position을 계산하게 됨.
        
        def get_posi_angle_vec(position):
            return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]
        
        sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    
        return sinusoid_table
    ```
    
    ```python
    n_seq = 64
    pos_encoding = get_sinusoid_encoding_table(n_seq, d_hidn)
    
    print(pos_encoding.shape) # 모양 출력
    plt.pcolormesh(pos_encoding, cmap="RdBu")
    plt.xlabel('Depth')
    plt.xlim((0, d_hidn))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
    ```
    
    ![Untitled](images/Untitled%207.png)
    

### 3. Input Embedding + Positional Encoding

1. 위에서 구한 positional Encoding 값을 이용해 positional Embedding 생성합니다. 학습되는 값이 아니므로 freeze 옵션을 True로 설정합니다.
2. Inputs와 동일한 크기를 갖는 positions 값을 구합니다.
3. Input 값 중 pad(0)을 찾습니다.
4. positions 값 중 pad 부분은 0으로 변경합니다.
5. positions 값에 대항하는 embedding 값을 구합니다.

```python
pos_encoding = torch.FloatTensor(pos_encoding)
nn_pos = nn.Embedding.from_pretrained(pos_encoding, freeze=True)

# range_nums = torch.arange(9).reshape(3,3)
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])
positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(2, inputs.size(1)).contiguous() + 1
pos_mask = inputs.eq(0) # input과 동일한 형태, 값이 0이면 True가 들어감.

positions.masked_fill_(pos_mask, 0)
pos_embs = nn_pos(positions) # position embedding

print(inputs)
# print(positions)
print(pos_embs.size())
# --- 출력 결과 ---
tensor([[ 3305,    18,  2360,   376,    57,  1914,   721,   303,  1373, 12797,  5058,     7],
        [ 1303,    51, 19304,   276, 17092,     7,     0,     0,     0,     0,     0,     0]])
# tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
#         [ 1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0,  0]])
torch.Size([2, 13, 128])
```

즉 위의 것을 다시 해석하자면 

- input
    - "겨울은 너무 기나긴 날들의 연속이였습니다.", "감기 조심하세요."라는 문장을 tokenizer를 이용해 문장을 분리하고, Vocab에서 사용할 수 있는 index로 만들었습니다.
    - 또한 shape을 [문장 중 Max_sequence의 길이, Sentence 갯수]로 설정했습니다. ( 일반적으로는 병렬연산을 위해 Max_sequence를 Dataset의 문장 길이를 보고 결정하게 됩니다.(짧은게 많으면 짧게, 긴게 많으면 길게)
- positions
    - 단순 위치를 알려주는 numpy array입니다.
- pos_embs
    - Token Embedding과 Position Embedding을 더한 결과입니다.

### 도식화

![Untitled](images/Untitled%208.png)

여기서 중간에 Segment Embedding이라는 내용이 같이 더해지게 되는데, 이것은 학습시에 길이가 짧은 문장들을 붙여서 한번에 학습을 시키게 되는데, 각각이 다른 문장임을 알려주게 됩니다.

[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, …, 1, 2, 2, 2, 2, … , 2, 0, 0, 0, 0, 0] → 마지막은 Padding

# 2. Model

- Thumbnail
    
    ![Untitled](images/Untitled%209.png)
    
    ![Untitled](images/Untitled%2010.png)
    

## 1. Multi-Head Attention

![Untitled](images/Untitled%2011.png)

- 조금 요상하게 생겼지만, 대충 전체적인 맥락만 짚자면 V, K, Q가 들어가서 Linear Transform으로 `Scaled Dot-Porduct Attention`에 맞는 형태로 변환됩니다.
- 이후 Scaled Dot Product Attention이 진행되고, 각 결과가 Concat 후 다시 Linear Transform으로 이후 연산에 맞게 변형됩니다.

### 1. Linear

앞에서 연산된 Embedding Vocab Sequence가 입력으로 들어옵니다. 해당 Vocab Sequence의 목적이 Encoding이냐 Decoding이냐에 따라 다르지만, 어쨌든 들어오게되면 Linear 연산을 통해 차원축소가 일어나며 Scaled Dot-Product Attention이 진행됩니다.

→ 즉 Linear Layer의 목적은 Scaled Dot-Product Attention을 진행하기 전 Dimension Reduction의 효과가 있습니다.

### 2. Scaled Dot-Product Attention

![Untitled](images/Untitled%2012.png)

Sclaed Dot-Product Attention은 다음과 같은 연산을 내포하고 있습니다.

- **여기서 먼저 Q, K, V가 무엇이냐면, (수정 예정)**
    - Q : Attention을 구하고자 하는 Sequence(문장) 1
    - K : Attention을 구하고자 하는 Sequence(문장) 2
    - V  Attention을 구하고자 하는 Sequence(문장) 2
    
    입니다. ? 그럼 왜 꼬라지가 저따구? 
    
    - 먼저 Encoding 같은 경우 입력된 Q, K, V가 모두 같은 문장으로부터 만들어진 값들입니다.
    - 이때 Q = K =V 이므로 이를 Self-Attention이라고 하는데, 결과적으로 Input Sentence에 대한 “스스로의 Attention”을 구하기 때문입니다.
        - 그럼 왜 스스로의 Attention을 구하냐면? 한 문장 내에서 각 단어들별(Tokenizing 된 단위 별)로 각 단어들별로 관련이 있는지를 계산하기 위함입니다.
        - 예를 들어 “나는 집으로 재빠르게 간다” 라는 문장을 보면  “나”가 “간다”와 굉장히 연관이 깊어 보입니다. 또한 “재빠르게”는 “간다”와 연관성이 깊어 보입니다. 이렇게 1개의 문장 내에서 Attention을 구할 때 위와 같이 진행됩니다.
    - Decoding의 경우는 조금 다릅니다.
        - Q는 “현재 생성되고 있는 문장”이고 K, V는 Encoding된 “Encoding 완료된” 문장입니다.
        - 즉 “현재까지 생성된 문장”과 Input Encoding(Input Query)와의 Attention을 구하기 위함입니다.
        
        **→ Decoder 부분은 아직 굉장히 미숙함을 느낍니다. 죄송합니다.**
        
    
- Self-Attention
    
    "The animal didn't cross the street because it was too tired”
    
    다음의 예시 문장을 통해 직관적으로 Attention이 무엇인지 이해해보겠습니다.
    
    사람은 위 문장에서 `it`이 무엇인지 쉽게 알 수 있습니다. 하지만 컴퓨터는 `it`이 무엇인지, 무엇과 연관성이 깊은지 알기는 쉽지 않습니다. 만약 `it`이 무엇과 연관성이 깊고 혹은 무엇인지 컴퓨터가 이해하고 있다면 해당 문장을 이해하는데 굉장히 큰 도움이 될 것입니다. 이때 이러한 관계성을 표현한 것이 바로 Self-Attention Layer의 목적입니다. 
    
    ![Untitled](images/Untitled%2013.png)
    
    시각화를 통해 살펴보면 it 은 “The Animal_”과 굉장히 유사한 것을 알 수 있습니다.
    
    그럼 해당 Self-Attention이 어떻게 계산되는지 알아보겠습니다.
    
    먼저 Input Vector가 들어오면 Self-Attention층에서는 Q, K, V 벡터를 인풋 벡터수 만큼 준비합니다. 
    
    - **Query** : 현재 처리하고자 하는 토큰 벡터
    - **Key** : 일종의 레이블로, 시퀀스 내에 있는 모든 토큰에 대한 아이덴티티
    - **Value** : 키와 연결된 실제 토큰을 나타내는 벡터입니다.
    
    ![Untitled](images/Untitled%2014.png)
    
    우리는 이제 Input Query와 잘 맞는(Attention이 높은) 값을 Key에서 찾아야 합니다. 이를 위해 Attention을 구하는 과정을 진행해야 하는데, 이 과정은 Dot product(내적)을 기반으로 Softmax를 해 확률로써 표현하게 됩니다. 이때 Gradient의 안정을 위해 Scaled($\sqrt{d_k}$)로 나누어주었습니다.
    
    ![Untitled](images/Untitled%2015.png)
    
    이제 구해진 각각의 Attention을 이용해 Value와의 곱으로 각 단어를 얼마나 사용해 계산할 지를 정해 Output을 결정하게 됩니다.
    

### 3. Multi-Head Attention

앞서 했던 Scaled Dot-Product Attention으로부터 비롯된 연산입니다.

[위의 그림](https://www.notion.so/Transformer-85f3380626914efa9fb151578d8a5c36)에서처럼 Scaled Dot-Product Attention이 병렬적으로 구성되어 있고, 각각의 Layer의 Output을 Concatenate을 통해 이어 붙인 후, Linear Layer로 Dimension Reduction을 진행하게 됩니다. 이후 Classification이든 무엇이든 간에 진행된다고 생각하면 됩니다

이때 Encoder와 Decoder에서 한가지 차이가 있는데,

![Untitled](images/Untitled%2016.png)

Decoder는 입력시에 이전 이후 입력에 대한 Attention을 구한채로 연산을 진행하게 되면, 정답을 알고 Attention을 구하게 되는것과 마찬가지라, 이 과정을 Masking을 통해 진행됩니다.

이때 해당 Attention을 구할때에는 Softmax라는 함수를 이용해 값을 구하게 됩니다. 그래서 아직 나오지 않은 Sequence를 배제하려고 하면 해당 위치에 -1e9정도의 굉장히 작은 값을 넣으면 됩니다.

![Untitled](images/Untitled%2017.png)

## 2. FeedForward Network

![Untitled](images/Untitled%2018.png)

직관적으로 2번의 Linear Transform과 1번의 Activation function이 적용된 Network입니다. 

일정 차원의 Output $z$를 조금 더 큰 차원의 Output으로 Lienar Transform을 시킨 후, Target Task에 맞는 Output으로의 Linear Transform을 이용해 조금 더 Fit하게 만듭니다.

## 3. Encoder-Decode Attention

Decoder의 Masked Self Attention 층의 출력 벡터가 Residual Block과 Layer normalization을 거친 뒤에 Encoder-Decoder Attention 과정을 거치게 됩니다. 이 층에서는 인코더의 마지막 블로겡서 출력된 키, 벨류 행렬으로 Self-Attention 매커니즘을 한번 더 수앵하게 됩니다.

![http://jalammar.github.io/images/t/transformer_decoding_2.gif](http://jalammar.github.io/images/t/transformer_decoding_2.gif)

# 3. Training Process

## 0. Requirements & Dataset

- 학습데이터: ratings_train.txt
- 평가데이터: ratings_test.txt

```python
!wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
!wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
```

```python
# vocab loading
vocab_file = "/content/drive/MyDrive/Colab_Notebooks/Paper_Implement/NLP Basic/kowiki_32000.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)
```

## 1. Model

영화 댓글 감정 분석을 위한 Classification Model

```python
""" naver movie classfication """
class MovieClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = Transformer(self.config)
        self.projection = nn.Linear(self.config.d_hidn, self.config.n_output, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_dec_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, dec_inputs)
        # (bs, d_hidn)
        dec_outputs, _ = torch.max(dec_outputs, dim=1)
        # (bs, n_output)
        logits = self.projection(dec_outputs)
        # (bs, n_output), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
```

## 2. Dataset

### Dataset

Output으로 3개의 Output을 내놓는 것에 유의

```python
""" 영화 분류 데이터셋 """
class MovieDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels = []
        self.sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                self.sentences.append([vocab.piece_to_id(p) for p in data["doc"]])
    
    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        return len(self.labels)
    
    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor([self.vocab.piece_to_id("[BOS]")]))
```

### Collate_fn

배치 단위의 데이터를 처리하기위한 collate_fn

```python
""" movie data collate_fn """
def movie_collate_fn(inputs):
    labels, enc_inputs, dec_inputs = list(zip(*inputs))

    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0) # 이 함수를 이용해 전체 Input에 대한 일정한 크기로 padding을 부착함
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0) # 이 함수를 이용해 전체 Input에 대한 일정한 크기로 padding을 부착함

    batch = [
        torch.stack(labels, dim=0),
        enc_inputs,
        dec_inputs,
    ]
    return batch
```

### DataLoader

Dataset과 Collate_fn을 이용해 Train, test loader를 만듦.

```python
batch_size = 128
train_dataset = MovieDataSet(vocab, "./ratings_train.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=movie_collate_fn)
test_dataset = MovieDataSet(vocab, "./ratings_test.json")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn)
```

## 3. Training config & Training

### config

```python
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.n_output = 2
print(config)

learning_rate = 5e-5
n_epoch = 10
```

### Training Epoch

```python
""" 모델 epoch 학습 """
def train_epoch(config, epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0] # [0]인 이유는 Classification output 중 0번이 logit임.

            loss = criterion(logits, labels)
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)
```

### Evaluate Epoch

1. Encoder input과 Decoder input을 입력으로 MovieClassification을 실행합니다.
2. 1번의 결과 중 첫 번째 값이 예측 logits 입니다.
3. logits의 최대값의 index를 구합니다.
4. 3번에게 구한 값과 labels의 값이 같은지 비교 합니다.

```python
""" 모델 epoch 평가 """
def eval_epoch(config, model, data_loader):
    matchs = []
    model.eval()

    n_word_total = 0
    n_correct_total = 0
    with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
        for i, value in enumerate(data_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0]
            _, indices = logits.max(1)

            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0
```

### Training

```
model = MovieClassification(config)
model.to(config.device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses, scores = [], []
for epoch in range(n_epoch):
    loss = train_epoch(config, epoch, model, criterion, optimizer, train_loader)
    score = eval_epoch(config, model, test_loader)

    losses.append(loss)
    scores.append(score)
```

## 4. Result

```python
# table
data = {
    "loss": losses,
    "score": scores
}
df = pd.DataFrame(data)
display(df)

# graph
plt.figure(figsize=[12, 4])
plt.plot(losses, label="loss")
plt.plot(scores, label="score")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()
```

---

# 질문

- Attention에 내적이라는 개념이 나옵니다. 내적이 무엇인가요?
    - 내적을 통해 나온 값이 높다 → 유사성이 높다
    - 내적을 통해 나온 값이 낮다 → 유사성이 낮다.
    
    내적은 수식이 
    
    $$
    A\cdot B = |A||B|cos\theta
    $$
    
    즉 A도 크고, B도 크고 $cos\theta$도 커야 $A\cdot B$가 커짐.
    
    이것을 Attention 개념에 적용하면 
    
    A도 독립적으로 크고, B도 독립적으로 크고, $cos\theta$도 커야함. 
    
    여기서 각도인 $cos\theta$에 조금 집중해서 살펴보는게 직관적인 이해가 잘 될듯 함.