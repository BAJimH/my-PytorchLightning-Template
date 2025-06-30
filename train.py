# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This main entrance of the whole project.

Most of the code should not be changed, please directly
add all the input arguments of your model's constructor
and the dataset file's constructor. The MInterface and
DInterface can be seen as transparent to all your args.
"""
import lightning as pl
from argparse import ArgumentParser
from lightning import Trainer
import lightning.pytorch.callbacks as plc
from lightning.pytorch.tuner import Tuner

from models import MInterface
from dataset import DInterface
import torch
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    # callbacks.append(
    #    plc.EarlyStopping(monitor="loss", mode="max", patience=10, min_delta=0.01)
    # )

    callbacks.append(
        plc.ModelCheckpoint(
            monitor="val_loss",
            filename="best-{epoch:02d}-{train_loss:.4f}",
            save_top_k=1,
            mode="min",
            save_last=True,
        )
    )
    callbacks.append(
        # 这里check_on_train_epoch_end=False一定要关掉
        plc.EarlyStopping(
            monitor="val_loss",
            patience=300,
            min_delta=0.00,
            mode="min",
            verbose=True,
            check_on_train_epoch_end=False,
        )
    )

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    data_module.setup(stage="fit")
    model = MInterface(**vars(args))

    args.callbacks = load_callbacks()
    trainer = Trainer(
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        benchmark=True,
        devices=args.devices,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator="auto",
        limit_val_batches=1.0,
        enable_checkpointing=True,
        # inference_mode=True,
        callbacks=load_callbacks(),
        check_val_every_n_epoch=1,
        default_root_dir=f"./train_logs/{args.model_name}",
        # strategy='ddp_find_unused_parameters_true',
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=args.enable_progress_bar,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )

    # 自动发现最佳学习率（如果指定了自动发现学习率）
    if args.auto_lr_find:
        tuner = Tuner(trainer)
        # 运行学习率寻找器
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
            min_lr=1e-8,
            max_lr=1e-2,
            num_training=100,
        )

        # 获取建议的学习率并应用
        new_lr = lr_finder.suggestion()
        print(f"自动发现的最佳学习率: {new_lr}")
        args.lr = new_lr
        model.hparams.lr = new_lr

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
        ckpt_path=load_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--devices", default=-1, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)

    # LR Scheduler
    parser.add_argument(
        "--lr_scheduler",
        choices=["step", "cosine", "cosine_warmup", "step_warmup"],
        type=str,
    )
    parser.add_argument("--lr_decay_steps", default=20, type=int)
    parser.add_argument("--lr_decay_rate", default=0.9, type=float)
    parser.add_argument("--lr_decay_min_lr", default=1e-7, type=float)
    parser.add_argument(
        "--warmup_epochs",
        default=20,
        type=int,
        help="Number of epochs for learning rate warmup",
    )
    parser.add_argument(
        "--warmup_start_lr",
        default=1e-7,
        type=float,
        help="Initial learning rate for warmup",
    )

    # Restart Control
    parser.add_argument("--load_best", action="store_true")
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--load_ver", default=None, type=str)
    parser.add_argument("--load_v_num", default=None, type=int)

    # Training Info

    parser.add_argument("--dataset", default="RGB_spike_sync_data", type=str)
    parser.add_argument(
        "--data_dir", default="/mnt/data/spike/gray_vids_simulate", type=str
    )
    parser.add_argument("--val_rate", default=0.05, type=float)

    parser.add_argument("--model_name", default="RGB_spike_deblur_net", type=str)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--min_epochs", default=500, type=int)
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--enable_progress_bar", default=True, type=bool)
    parser.add_argument("--spike_seq_len", default=8, type=int)
    parser.add_argument("--log_every_n_steps", default=10, type=int)
    # Add pytorch lightning's args to parser as a group.

    parser.add_argument(
        "--auto_lr_find", action="store_true", help="自动寻找最佳学习率"
    )

    args = parser.parse_args()

    main(args)
