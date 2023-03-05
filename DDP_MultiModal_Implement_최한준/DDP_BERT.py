import os
import torch
import argparse

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import Custom_Dataset
from model import TourClassifier
from trainer import Trainer



import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from transformers import AutoTokenizer, ViTFeatureExtractor
from transformers.optimization import get_cosine_schedule_with_warmup

from pprint import pprint
# DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from utils import str2bool

def load_train_objs(args):
    if args.train:
        df = pd.read_csv('../train.csv')

        le = preprocessing.LabelEncoder()
        le.fit(df['cat3'].values)
        df['original'] = df['cat3']
        df['cat3'] = le.transform(df['cat3'].values)
        reverse_dict = {}
        for cat3, original in zip(df['cat3'], df['original']):
            reverse_dict[cat3] = original
        le = preprocessing.LabelEncoder()
        le.fit(df['cat2'].values)
        df['cat2'] = le.transform(df['cat2'].values)
        le = preprocessing.LabelEncoder()
        le.fit(df['cat1'].values)
        df['cat1'] = le.transform(df['cat1'].values) 

        folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        df['kfold'] = -1
        for i in range(5):
            df_idx, valid_idx = list(folds.split(df.values, df['cat3']))[i]
            valid = df.iloc[valid_idx]

            df.loc[df[df.id.isin(valid.id) == True].index.to_list(), 'kfold'] = i
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

        ds = Custom_Dataset(
            text=df.overview.to_numpy(),
            image_path = df.img_path.to_numpy(),
            cats1=df.cat1.to_numpy(),
            cats2=df.cat2.to_numpy(),
            cats3=df.cat3.to_numpy(),
            tokenizer=tokenizer,
            feature_extractor = feature_extractor,
            max_len=256,
            train=args.train
        )


        model = TourClassifier(n_classes1 = 6, n_classes2 = 18, n_classes3 = 128, text_model_name = "klue/roberta-small",image_model_name = "google/vit-base-patch16-224", device=int(os.environ["LOCAL_RANK"])).to(int(os.environ["LOCAL_RANK"]))
        optimizer = optim.AdamW(model.parameters(), lr=3e-5)
        total_steps = len(ds) * args.total_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps*0.1),
            num_training_steps=total_steps
        )
    else:
        df = pd.read_csv('../test.csv')
        arr = [ 
            '5일장', 'ATV', 'MTB', '강', '게스트하우스', '계곡', '고궁', '고택', '골프', '공연장',
            '공예,공방', '공원', '관광단지', '국립공원', '군립공원', '기념관', '기념탑/기념비/전망대',
            '기암괴석', '기타', '기타행사', '농.산.어촌 체험', '다리/대교', '대중콘서트', '대형서점',
            '도립공원', '도서관', '동굴', '동상', '등대', '래프팅', '면세점', '모텔', '문', '문화관광축제',
            '문화원', '문화전수시설', '뮤지컬', '미술관/화랑', '민물낚시', '민박', '민속마을', '바/까페',
            '바다낚시', '박람회', '박물관', '발전소', '백화점', '번지점프', '복합 레포츠', '분수', '빙벽등반',
            '사격장', '사찰', '산', '상설시장', '생가', '서비스드레지던스', '서양식', '섬', '성',
            '수련시설', '수목원', '수상레포츠', '수영', '스노쿨링/스킨스쿠버다이빙', '스카이다이빙', '스케이트',
            '스키(보드) 렌탈샵', '스키/스노보드', '승마', '식음료', '썰매장', '안보관광', '야영장,오토캠핑장',
            '약수터', '연극', '영화관', '온천/욕장/스파', '외국문화원', '요트', '윈드서핑/제트스키',
            '유람선/잠수함관광', '유명건물', '유스호스텔', '유원지', '유적지/사적지', '이색거리', '이색찜질방',
            '이색체험', '인라인(실내 인라인 포함)', '일반축제', '일식', '자동차경주', '자연생태관광지',
            '자연휴양림', '자전거하이킹', '전문상가', '전시관', '전통공연', '종교성지', '중식', '채식전문점',
            '카약/카누', '카지노', '카트', '컨벤션', '컨벤션센터', '콘도미니엄', '클래식음악회', '클럽',
            '터널', '테마공원', '트래킹', '특산물판매점', '패밀리레스토랑', '펜션', '폭포', '학교', '한식',
            '한옥스테이', '항구/포구', '해수욕장', '해안절경', '헬스투어', '헹글라이딩/패러글라이딩', '호수',
            '홈스테이', '희귀동.식물'
        ]
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

        ds = Custom_Dataset(
            text=df.overview.to_numpy(),
            image_path = df.img_path.to_numpy(),
            cats1=None,
            cats2=None,
            cats3=None,
            tokenizer=tokenizer,
            feature_extractor = feature_extractor,
            max_len=256,
            train=args.train
        )
        model = TourClassifier(n_classes1 = 6, n_classes2 = 18, n_classes3 = 128, text_model_name = "klue/roberta-small",image_model_name = "google/vit-base-patch16-224", args=args).to('cuda:0')
        optimizer = optim.AdamW(model.parameters(), lr=3e-5)
        total_steps = len(ds) * args.total_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps*0.1),
            num_training_steps=total_steps
        )

    return ds, model, optimizer, scheduler, arr

def prepare_dataloader(dataset : Dataset, batch_size : int, args):
    if args.train:
        return DataLoader(
            dataset, 
            batch_size=batch_size, # 여기서 GPU갯수의 배수가 되도록 해야함.
            pin_memory=True,
            shuffle=False, # DistributedSampler가 알아서 해줌.
            sampler=DistributedSampler(dataset)
        )
    else:
        return DataLoader(
            dataset, 
            batch_size=batch_size, # 여기서 GPU갯수의 배수가 되도록 해야함.
            pin_memory=True,
            shuffle=False, # DistributedSampler가 알아서 해줌.
        )


def main(args):
    if args.train:    
        init_process_group(backend="nccl")
        dataset, model, optimizer, scheduler, reverse_dict = load_train_objs(args)
        dataloader = prepare_dataloader(dataset, args.batch_size, args)
        trainer = Trainer(model, dataloader, optimizer, scheduler, args)
        trainer.train(args.total_epochs)
        destroy_process_group()
    else:
        dataset, model, optimizer, scheduler, reverse_dict = load_train_objs(args)
        dataloader = prepare_dataloader(dataset, args.batch_size, args)
        trainer = Trainer(model, dataloader, optimizer, scheduler, args)
        preds1, preds2, preds3 = trainer.inference()
        sample_submission = pd.read_csv('../submit.csv')
        for idx, pred in enumerate(preds3):
            sample_submission.loc[idx, 'cat3'] = reverse_dict[pred]
        sample_submission.to_csv('../sample_submission_CHJ(VIT+BERT).csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('batch_size', type=int, help='Batch Size')
    parser.add_argument('train', help='boolean flag', default=False, type=str2bool)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    main(args) # torch run 하면 끝