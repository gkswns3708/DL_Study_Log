# BERT를 구현 중인 Repo입니다.

현재 Reference 상의 Git Code를 참고해서 저의 Envs에 맞춰서 코드를 Refactoring하는 작업 중에 있습니다.

목표는 저만의 Pretrained Model을 만들고, Task Specific한 Task를 수행하는 것입니다.

---
# Folder Structure
```
    │  .gitignore
    │  config.json
    │  config.py
    │  config_half.json
    │  data.py
    │  log.txt
    │  model.py
    │  optimization.py
    │  practice.ipynb
    │  pretrain.py
    │  README.md
    │  requirements.txt
    │  train.py
    │
    ├─Data
    │  │  kowiki.model
    │  │  kowiki.txt
    │  │  kowiki.vocab
    │  │  kowiki_bert_0.json # sample data
    │  │  kowiki_bert_1.json # sample data
    │  │  kowiki_bert_2.json # sample data
    │  │  kowiki_bert_3.json # sample data
    │  │  kowiki_bert_4.json # sample data
    │  │  vocab.py
    │
    └─save
          save_pretrain.pth

```
---
# Quick Start
```
$ python data.py         # Large Corpus를 생성함. 
$ python pretrain.py     # Large Corpus를 이용한 Pre-trainin을 실행함.
$ python train.py        # Fine-tuning Task를 수행함(네이버 영화 댓글 데이터를 이용한 Sentence Binary Classification)
```
# Requirements
https://drive.google.com/drive/folders/1lf8t9jE5LcFpg2UeMTPrD_rjjJQiC8oq?usp=sharing

위 링크에서 Vocab을 형성할 때 필요한 Pre-trained Model, Vocab, Raw Data들이 있습니다. 

해당 코드에 관한 Reference는 https://paul-hyun.github.io/vocab-with-sentencepiece/ 에 있습니다.

```
black==22.6.0
blessings==1.7
brotlipy @ file:///D:/bld/brotlipy_1648854327487/work
certifi==2022.6.15
cffi @ file:///D:/bld/cffi_1656782930891/work
charset-normalizer @ file:///home/conda/feedstock_root/build_artifacts/charset-normalizer_1655906222726/work
click==8.1.3
colorama==0.4.5
cryptography @ file:///D:/bld/cryptography_1657174166005/work
docker-pycreds==0.4.0
filelock==3.7.1
gitdb==4.0.9
GitPython==3.1.27
gpustat==0.6.0
huggingface-hub==0.8.1
idna @ file:///home/conda/feedstock_root/build_artifacts/idna_1642433548627/work
joblib==1.1.0
mypy-extensions==0.4.3
numpy @ file:///D:/bld/numpy_1657483989399/work
nvidia-ml-py3==7.352.0
packaging==21.3
pandas==1.4.3
pathspec==0.9.0
pathtools==0.1.2
Pillow @ file:///D:/bld/pillow_1657007292472/work
platformdirs==2.5.2
promise==2.3
protobuf==3.20.1
psutil==5.9.1
pycparser @ file:///home/conda/feedstock_root/build_artifacts/pycparser_1636257122734/work
pyOpenSSL @ file:///home/conda/feedstock_root/build_artifacts/pyopenssl_1643496850550/work
pyparsing==3.0.9
PySocks @ file:///D:/bld/pysocks_1648857426942/work
python-dateutil==2.8.2
pytz==2022.1
PyYAML==6.0
regex==2022.7.9
requests @ file:///home/conda/feedstock_root/build_artifacts/requests_1656534056640/work
scikit-learn==1.1.1
scipy==1.8.1
sentencepiece==0.1.96
sentry-sdk==1.7.2
setproctitle==1.2.3
shortuuid==1.0.9
six==1.16.0
sklearn==0.0
smmap==5.0.0
threadpoolctl==3.1.0
tokenizers==0.12.1
tomli==2.0.1
torch==1.12.0
torchaudio==0.12.0
torchvision==0.13.0
tqdm==4.64.0
transformers==4.20.1
typing_extensions @ file:///home/conda/feedstock_root/build_artifacts/typing_extensions_1656706066251/work
urllib3 @ file:///home/conda/feedstock_root/build_artifacts/urllib3_1657224465922/work
wandb==0.12.21
win-inet-pton @ file:///D:/bld/win_inet_pton_1648771910527/work
wincertstore==0.2
```
---
# ETC
Corpus, Sentencepiece로 학습된 Vocab, SentencePiece Model 등을 아래 Reference에서 확인할 수 있습니다.
---

# Reference
- https://github.com/paul-hyun/transformer-evolution/tree/master/bert
