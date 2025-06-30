# Copyright 2021 Zhongyang Zhang
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

import inspect
import torch
import numpy as np
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning as pl


class MInterface(pl.LightningModule):
    def __init__(self, model_name="RGB_deblur_model", **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model(model_name)

    def make_loss_dict(self, loss_dict, stage):
        new_loss_dict = {}
        for key, value in loss_dict.items():
            new_loss_dict[f"{stage}_{key}"] = value
        return new_loss_dict

    def training_step(self, batch, batch_idx):
        # 由于这里只是一个接口，因此直接把通用的Batch传进去而不管形式，要求具体的模型封装好，输出output和loss
        out, loss_dict = self.model(batch, stage="train")
        loss_dict = self.make_loss_dict(loss_dict, "train")
        with torch.no_grad():
            self.log_dict(
                loss_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
            )
        return loss_dict["train_loss"]

    def validation_step(self, batch, batch_idx):
        out, loss_dict = self.model(batch, stage="val")
        loss_dict = self.make_loss_dict(loss_dict, "val")
        with torch.no_grad():
            self.log_dict(
                loss_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
            )
        return loss_dict["val_loss"]


    def load_model(self, name):
        # 支持下划线命名的文件名和类名
        # 需要保证文件名和类名一致，文件名用下划线，类名用驼峰
        if(name.find("_v")!=-1):
            camel_name = "".join([i[0].upper() + i[1:] for i in name.split("_v")[0].split("_")])
        else:
            camel_name = "".join([i[0].upper() + i[1:] for i in name.split("_")])

        try:
            model = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            try:
                model = getattr(
                    importlib.import_module("." + name, package=__package__), name
                )
            except Exception as e:
                raise ValueError(
                    f"Invalid Model File Name or Invalid Class Name models.{name}.{camel_name}, {e}"
                )
        self.model = self.instancialize(model)

    def configure_optimizers(self):
        if hasattr(self.hparams, "weight_decay"):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
        )

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == "step":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
            elif self.hparams.lr_scheduler == "cosine":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_epochs,  # 修改为使用总epoch数
                    eta_min=self.hparams.lr_decay_min_lr,
                )
            elif self.hparams.lr_scheduler == "cosine_warmup":
                # 创建预热期+余弦退火的复合调度器
                warmup_scheduler = lrs.LinearLR(
                    optimizer, 
                    start_factor=self.hparams.warmup_start_lr / self.hparams.lr,
                    end_factor=1.0,
                    total_iters=self.hparams.warmup_epochs
                )
                cosine_scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                    eta_min=self.hparams.lr_decay_min_lr
                )
                
                # 使用SequentialLR组合两个调度器
                scheduler = lrs.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.hparams.warmup_epochs]
                )
            elif self.hparams.lr_scheduler == "step_warmup":
                # 创建预热期+阶梯退火的复合调度器
                warmup_scheduler = lrs.LinearLR(
                    optimizer, 
                    start_factor=self.hparams.warmup_start_lr / self.hparams.lr,
                    end_factor=1.0,
                    total_iters=self.hparams.warmup_epochs
                )
                step_scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
                
                # 使用SequentialLR组合两个调度器
                scheduler = lrs.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, step_scheduler],
                    milestones=[self.hparams.warmup_epochs]
                )
            else:
                raise ValueError("Invalid lr_scheduler type!")
            
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]  # 明确指定按epoch更新

    def instancialize(self, cls, **other_args):
        """Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(cls.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return cls(**args1)
    
