import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from train import *
from args import parse_global_args, parse_dataset_specific_args


if __name__ == '__main__':

    # hyper parameters
    parser, args = parse_global_args()
    args = parse_dataset_specific_args(args)

    # training module
    module: pl.LightningModule = TrainHiFGL(**vars(args))

    # train & test
    early_stopping_callback = EarlyStopping(monitor=args.monitor, mode='max', min_delta=0.0, patience=args.patience, verbose=False, check_finite=True)
    model_checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=args.monitor, mode='max')
    trainer = Trainer(
        gpus=1,
        auto_select_gpus=True,
        accelerator='gpu',
        max_epochs=args.epochs,
        logger=TensorBoardLogger(
            save_dir='../',
            name='log',
        ),
        weights_summary='top',  # 'full'
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )

    print('Training model...')
    trainer.fit(module)
    print('Testing model...')
    trainer.test(module, ckpt_path='best')

    # import pandas as pd
    # acc = pd.DataFrame.from_dict(module.acc)
    # acc.to_csv('../out/acc.csv')
