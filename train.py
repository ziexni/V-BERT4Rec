"""
train.py
V-BERT4REC 학습 파이프라인
"""

import argparse
import datetime

from datamodule import DataModule
from lit_model import VBERT4REC  # ✅ V-BERT4REC 모델

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def _setup_parser():
    """
    CLI argument parser
    """
    parser = argparse.ArgumentParser(description='V-BERT4REC - MicroVideo')
    
    # Data 관련
    data_group = parser.add_argument_group("Data Args")
    DataModule.add_to_argparse(data_group)
    
    # Model 관련
    model_group = parser.add_argument_group("Model Args")
    VBERT4REC.add_to_argparse(model_group)  # ✅ V-BERT4REC
    
    return parser


def _set_trainer_args(args):
    """
    Trainer 설정
    """
    args.max_epochs = 100
    args.gradient_clip_val = 5.0
    args.gradient_clip_algorithm = "norm"
    
    args.save_dir = "Training/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.name = "VBERT4REC_MicroVideo"  # ✅ V-BERT4REC
    
    args.monitor = "val_loss"
    args.mode = "min"
    args.patience = 5
    
    args.logging_interval = "step"


def main():
    """
    학습 파이프라인
    """
    parser = _setup_parser()
    args = parser.parse_args()
    _set_trainer_args(args)
            
    # DataModule (이미지 벡터 포함)
    data = DataModule(args)
    
    # ✅ V-BERT4REC 모델
    lit_model = VBERT4REC(args)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir = args.save_dir,
        name = args.name
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor = args.monitor,
        mode = args.mode,
        patience = args.patience
    )
    
    lr_monitor = LearningRateMonitor(
        logging_interval = args.logging_interval
    )
    
    checkpoint = ModelCheckpoint(
        monitor    = 'val_loss',
        mode       = 'min',
        save_top_k = 1,
        filename   = 'best'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs             = args.max_epochs,
        gradient_clip_val      = args.gradient_clip_val,
        gradient_clip_algorithm= args.gradient_clip_algorithm,
        logger                 = logger,
        callbacks              = [early_stop, lr_monitor, checkpoint],
        accelerator            = 'gpu',
        devices                = 1,
    )
    
    # 학습
    trainer.fit(lit_model, datamodule=data)
    
    # 테스트
    trainer.test(lit_model, datamodule=data, ckpt_path='best')


if __name__ == "__main__":
    main()
