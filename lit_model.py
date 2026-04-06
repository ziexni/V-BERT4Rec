"""
lit_model.py
V-BERT4REC Model (이미지 벡터 통합)
"""

import pytorch_lightning as pl
from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG, RetrievalMRR
import torch
import torch.nn as nn
import numpy as np

from bert import BERT


class VBERT4REC(pl.LightningModule):
    """
    V-BERT4REC: BERT4REC + Visual Features (이미지 벡터)
    """
    def __init__(self, args):
        super(VBERT4REC, self).__init__()

        # 하이퍼파라미터
        self.learning_rate = args.learning_rate
        self.max_len = args.max_len
        self.hidden_dim = args.hidden_dim
        self.encoder_num = args.encoder_num
        self.head_num = args.head_num
        self.dropout_rate = args.dropout_rate
        self.dropout_rate_attn = args.dropout_rate_attn

        # Vocab (0: PAD / 1~item_size: 실제 item / item_size+1: MASK)
        self.vocab_size = args.item_size + 2

        self.initializer_range = args.initializer_range
        self.weight_decay = args.weight_decay
        self.decay_step = args.decay_step
        self.gamma = args.gamma

        # ✅ Visual feature 사용 여부
        self.use_visual = getattr(args, 'use_visual', True)
        self.visual_dim = getattr(args, 'visual_dim', 128)

        # ✅ V-BERT encoder (이미지 벡터 포함)
        self.model = BERT(
            vocab_size = self.vocab_size,
            max_len = self.max_len,
            hidden_dim = self.hidden_dim,
            encoder_num = self.encoder_num,
            head_num = self.head_num,
            dropout_rate = self.dropout_rate,
            dropout_rate_attn = self.dropout_rate_attn,
            initializer_range = self.initializer_range,
            visual_dim = self.visual_dim,      # ✅ 이미지 차원
            use_visual = self.use_visual        # ✅ Visual 사용
        )

        # Output head: (B, T, hidden_dim) -> (B, T, item_size+1)
        self.out = nn.Linear(self.hidden_dim, args.item_size + 1)

        self.batch_size = args.batch_size

        # Loss (PAD는 무시)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 평가 metric
        self.HR   = RetrievalHitRate(top_k=10)
        self.NDCG = RetrievalNormalizedDCG(top_k=10)
        self.MRR  = RetrievalMRR()

    def training_step(self, batch, batch_idx):
        """
        Train: BERT4REC masking + 이미지 벡터
        
        batch:
            seq: (B, T)
            labels: (B, T)
            visual_features: (B, T, visual_dim)  ✅
        """
        seq, labels, visual_features = batch

        # ✅ Visual features 전달
        logits = self.model(seq, visual_features)  # (B, T, hidden_dim)
        preds = self.out(logits)                    # (B, T, item_size+1)

        # Loss 계산
        loss = self.criterion(preds.transpose(1, 2), labels)

        self.log("train_loss", loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation: 101개 후보 평가 + 이미지 벡터
        
        batch:
            seq: (B, T)
            candidates: (B, 101)
            labels: (B, 101)
            visual_features: (B, T, visual_dim)  ✅
        """
        seq, candidates, labels, visual_features = batch

        # ✅ Visual features 전달
        logits = self.model(seq, visual_features)
        preds = self.out(logits)

        # 마지막 위치 prediction
        preds = preds[:, -1, :]  # (B, item_size+1)
        
        # 정답 item
        targets = candidates[:, 0]

        # Loss
        loss = self.criterion(preds, targets)

        # Candidate scores
        recs = torch.gather(preds, 1, candidates)

        # Metric 계산용 index
        steps = batch_idx * self.batch_size
        indexes = torch.arange(
            steps, steps + seq.size(0),
            dtype=torch.long,
            device=seq.device
        ).unsqueeze(1).repeat(1, 101)

        # Logging
        self.log("val_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("HR_val",
                 self.HR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("NDCG_val",
                 self.NDCG(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("MRR_val",
                 self.MRR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        """
        Test: validation과 동일
        """
        seq, candidates, labels, visual_features = batch

        # ✅ Visual features 전달
        logits = self.model(seq, visual_features)
        preds  = self.out(logits)

        preds   = preds[:, -1, :]
        targets = candidates[:, 0]
        loss    = self.criterion(preds, targets)

        recs = torch.gather(preds, 1, candidates)

        steps   = batch_idx * self.batch_size
        indexes = torch.arange(
            steps, steps + seq.size(0),
            dtype=torch.long,
            device=seq.device
        ).unsqueeze(1).repeat(1, 101)

        self.log("test_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("HR_test",
                 self.HR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("NDCG_test",
                 self.NDCG(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("MRR_test",
                 self.MRR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Inference (추천 결과 생성)
        """
        if len(batch) == 4:
            seq, _, _, visual_features = batch
        else:
            seq = batch
            # Visual features가 없으면 None
            visual_features = None

        logits = self.model(seq, visual_features)
        preds = self.out(logits)

        preds = preds[:, -1, :]

        # Top-10 추천
        indexes, _ = torch.topk(preds, 10)

        return indexes.cpu().numpy()
    
    def configure_optimizers(self):
        """
        Optimizer + Scheduler
        """
        # Weight decay 분리
        no_decay = ['bias', 'LayerNorm.weight']

        params = [
            {
                'params' : [p for n, p in self.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                'weight_decay' : self.weight_decay
            },
            {
                'params' : [p for n, p in self.named_parameters()
                            if any(nd in n for nd in no_decay)],
                'weight_decay' : 0.0
            }
        ]

        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        # Learning rate decay
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.decay_step,
            gamma=self.gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    @staticmethod
    def add_to_argparse(parser):
        """
        CLI argument
        """
        parser.add_argument("--learning_rate",     type=float, default=1e-3)
        parser.add_argument("--hidden_dim",        type=int,   default=256)
        parser.add_argument("--encoder_num",       type=int,   default=2)
        parser.add_argument("--head_num",          type=int,   default=4)
        parser.add_argument("--dropout_rate",      type=float, default=0.1)
        parser.add_argument("--dropout_rate_attn", type=float, default=0.1)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument("--weight_decay",      type=float, default=0.01)
        parser.add_argument("--decay_step",        type=int,   default=25)
        parser.add_argument("--gamma",             type=float, default=0.1)
        parser.add_argument("--use_visual",        type=bool,  default=True)  # ✅ Visual 사용
        return parser
