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
import importlib
import lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


class DInterface(pl.LightningDataModule):
    # 本质上提供了一个选择数据集的接口
    # 通过传入参数dataset选择数据集
    # 通过传入参数kwargs选择数据集的参数
    # 并且自动封装成为DataLoader
    def __init__(self, num_workers=8, dataset="", val_rate=0.05, **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.val_rate = val_rate
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs["batch_size"]
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.trainset, self.valset = random_split(
                self.instancialize(stage="train"), [1 - self.val_rate, self.val_rate]
            )
            if self.valset is not None:
                self.valset.stage = "val"
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.testset = self.instancialize(stage="test")

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def load_data_module(self):
        name = self.dataset
        # 支持下划线命名的文件名和类名
        # 需要保证文件名和类名一致
        # 我只想首字母大写，其他字母不变，直接用capitalize会把其他字母变成小写
        # 通过import_module和getattr实现动态导入
        camel_name = "".join([i[0].upper() + i[1:] for i in name.split("_")])
        # camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except Exception as e:
            print(e,"try no camel")
            try:
                self.data_module = getattr(
                    importlib.import_module("." + name, package=__package__), name
                )
            except Exception as e:
                print(e)
                raise ValueError(
                    f"Invalid Dataset File Name or Invalid Class Name dataset.{name}.{camel_name}"
                )

    def instancialize(self, **other_args):
        """Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
