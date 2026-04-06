"""
datamodule.py
V-BERT4REC DataModule (이미지 벡터 포함)
"""

import pytorch_lightning as pl  
from torch.utils.data import DataLoader
from typing import Optional
from data import MicroVideoDataset, get_data

# 기본 데이터 경로
INTERACTION_PATH = 'bigMatrix.pkl'
ITEM_PATH = 'item_used.parquet'


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()

        # 하이퍼파라미터
        self.max_len = args.max_len
        self.mask_prob = args.mask_prob
        self.neg_sample_size = args.neg_sample_size
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.interaction_path = args.interaction_path
        self.item_path = args.item_path  # ✅ 아이템 데이터 경로

        # ✅ 데이터 로드 (이미지 벡터 포함)
        (self.user_train, self.user_valid, self.user_test, 
         self.usernum, self.itemnum, self.item_visual_features) = get_data(
            self.interaction_path, self.item_path
        )
        
        # args에 주입
        args.item_size = self.itemnum
        args.visual_dim = self.item_visual_features.shape[1]  # ✅ 이미지 벡터 차원

        print(f"✓ DataModule initialized")
        print(f"  - Items: {self.itemnum}")
        print(f"  - Visual dim: {args.visual_dim}")

    def setup(self, stage=None):
        """
        Dataset 생성
        """
        if stage == 'fit' or stage is None:
            # Train dataset
            self.train_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.item_visual_features,  # ✅ 이미지 벡터 전달
                self.max_len, self.mask_prob, mode='train'
            )

            # Validation dataset
            self.valid_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.item_visual_features,  # ✅ 이미지 벡터 전달
                self.max_len,
                neg_sample_size=self.neg_sample_size,
                mode='valid',
                usernum=self.usernum
            )
        
        if stage == 'test' or stage is None:
            # Test dataset
            self.test_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.item_visual_features,  # ✅ 이미지 벡터 전달
                self.max_len,
                neg_sample_size=self.neg_sample_size,
                mode='test',
                usernum=self.usernum
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    @staticmethod    
    def add_to_argparse(parser):
        """
        CLI argument 정의
        """
        parser.add_argument('--max_len',         type=int,   default=50)
        parser.add_argument('--mask_prob',       type=float, default=0.2)
        parser.add_argument('--neg_sample_size', type=int,   default=100)
        parser.add_argument('--batch_size',      type=int,   default=256)
        parser.add_argument('--pin_memory',      type=bool,  default=True)
        parser.add_argument('--num_workers',     type=int,   default=4)
        parser.add_argument('--item_size',       type=int,   default=0)
        parser.add_argument('--visual_dim',      type=int,   default=128)  # ✅ 이미지 차원
        parser.add_argument('--interaction_path', type=str,  default=INTERACTION_PATH)
        parser.add_argument('--item_path',       type=str,  default=ITEM_PATH)  # ✅ 아이템 경로

        return parser
