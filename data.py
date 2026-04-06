"""
data.py
MicroVideo 데이터셋 + 이미지 벡터 로드
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle


def get_data(interaction_path, item_path):
    """
    Interaction + Item 데이터 로드
    Returns:
        user_train, user_valid, user_test : {user: [item_id]}
        usernum, itemnum
        item_visual_features : (num_items+1, visual_dim) - 0번은 패딩용 제로벡터
    """
    # ── Interaction 로드 ──
    with open('bigMatrix.pkl', 'rb') as f:
        df = pickle.load(f)

    df['user_id'] = df['user_id'] + 1
    df['video_id'] = df['video_id'] + 1
    df = df.sort_values(by=['user_id', 'timestamp'], kind='mergesort').reset_index(drop=True)

    usernum = df['user_id'].max()
    itemnum = df['video_id'].max()

    # ── Item features 로드 ──
    item_df = pd.read_parquet(item_path)
    
    # 이미지 벡터 추출 (video_feature)
    visual_features = []
    visual_features.append(np.zeros(2048))  # 0번 인덱스: PAD용 제로벡터
    
    for item_id in range(itemnum):
        if item_id < len(item_df):
            feat = item_df.iloc[item_id]['video_feature']
            if isinstance(feat, np.ndarray):
                visual_features.append(feat)
            else:
                visual_features.append(np.array(feat))
        else:
            visual_features.append(np.zeros(2048))
    
    item_visual_features = np.stack(visual_features)  # (itemnum+1, 2048)
    
    print(f"✓ Visual features loaded: {item_visual_features.shape}")

    # ── Leave-two-out split ──
    User = defaultdict(list)
    for u, i in zip(df['user_id'], df['video_id']):
       User[u].append(int(i))

    user_train, user_valid, user_test = {}, {}, {}
    for user, seq in User.items():
        n = len(seq)
        if n < 3:
            user_train[user] = seq
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = seq[:-2]
            user_valid[user] = [seq[-2]]
            user_test[user] = [seq[-1]]

    print(f"[Split] train users: {len(user_train)}, "
          f"valid: {sum(1 for v in user_valid.values() if v)}, "
          f"test: {sum(1 for v in user_test.values() if v)}")

    return user_train, user_valid, user_test, usernum, itemnum, item_visual_features


class MicroVideoDataset(Dataset):
    """
    V-BERT4REC Dataset (이미지 벡터 포함)
    """
    def __init__(self, user_train, user_valid, user_test,
                 itemnum, item_visual_features, maxlen, 
                 mask_prob=0.2, neg_sample_size=100, mode='train', usernum=0):
        
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test

        self.itemnum = itemnum
        self.item_visual_features = item_visual_features  # ✅ 이미지 벡터
        self.maxlen = maxlen
        self.mask_prob = mask_prob
        self.neg_sample_size = neg_sample_size
        self.mask_token = itemnum + 1
        self.mode = mode
        self.item_size = itemnum

        # 사용할 유저 목록 생성
        if mode == 'train':
            self.users = [
                u for u, seq in user_train.items()
                if len(seq) >= 2
            ]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))
        else:
            ref = user_valid if mode == 'valid' else user_test
            self.users = [
                u for u in user_train
                if ref.get(u)
            ]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        u = self.users[idx]

        if self.mode == 'train':
            return self._train_item(u)
        else: 
            return self._eval_item(u)

    def _train_item(self, u):
        """
        Train: random masking + 이미지 벡터
        """
        seq = self.user_train[u]

        tokens, labels, visual_seq = [], [], []

        for item in seq:
            if random.random() < self.mask_prob:
                tokens.append(self.mask_token)
                labels.append(item)
                # ✅ 마스킹된 아이템은 제로벡터 (논문 방식)
                visual_seq.append(np.zeros(self.item_visual_features.shape[1]))
            else:
                tokens.append(item)
                labels.append(0)
                # ✅ 아이템의 이미지 벡터
                visual_seq.append(self.item_visual_features[item])

        # Truncate
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]
        visual_seq = visual_seq[-self.maxlen:]

        # Padding
        pad_len = self.maxlen - len(tokens)
        tokens = [0] * pad_len + tokens
        labels = [0] * pad_len + labels
        
        # ✅ Visual feature도 padding
        zero_visual = np.zeros(self.item_visual_features.shape[1])
        visual_seq = [zero_visual] * pad_len + visual_seq

        return (
            torch.LongTensor(tokens), 
            torch.LongTensor(labels),
            torch.FloatTensor(visual_seq)  # ✅ (T, visual_dim)
        )

    def _eval_item(self, u):
        """
        Eval: 마지막 아이템 예측 + 이미지 벡터
        """
        train_seq = self.user_train.get(u, [])

        if self.mode == 'valid':
            history = train_seq
            target = self.user_valid[u][0]
        else:
            history = self.user_train[u] + self.user_valid.get(u, [])
            target = self.user_test[u][0]
        
        item_seq = [iid for iid in history]
        item_seq = item_seq[-(self.maxlen - 1):]
        
        seq = item_seq + [self.mask_token]

        # ✅ Visual features 구성
        visual_seq = [self.item_visual_features[iid] for iid in item_seq]
        # 마지막은 제로벡터 (MASK)
        visual_seq.append(np.zeros(self.item_visual_features.shape[1]))

        # Padding
        pad_len = self.maxlen - len(seq)
        seq = [0] * pad_len + seq
        
        zero_visual = np.zeros(self.item_visual_features.shape[1])
        visual_seq = [zero_visual] * pad_len + visual_seq

        # Negative sampling
        rated = set(self.user_train[u]); rated.add(0)
        negs = []
        while len(negs) < self.neg_sample_size:
            t = np.random.randint(1, self.itemnum + 1)
            if t not in rated:
                negs.append(t)
            
        candidates = [target] + negs
        labels = [1] + [0] * self.neg_sample_size

        return (
            torch.LongTensor(seq),
            torch.LongTensor(candidates),
            torch.LongTensor(labels),
            torch.FloatTensor(visual_seq)  # ✅ (T, visual_dim)
        )
