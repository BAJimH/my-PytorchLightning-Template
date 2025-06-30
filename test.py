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
from lightning import Trainer, LightningModule
import lightning.pytorch.callbacks as plc
from lightning.pytorch.tuner import Tuner

from models import MInterface
from dataset import DInterface
from utils import load_model_path_by_args
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch


def writeImgTensor(img, path, gray=False, YCrCb=True):

    img = img.squeeze(0).detach().cpu().numpy()
    img = img / img.max()
    img = (img * 255).astype("uint8")
    img = np.transpose(img, (1, 2, 0))
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif YCrCb:
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(path, img)


def main(args):
    assert not (args.load_dir == args.load_ver == args.load_v_num == None)

    load_path = load_model_path_by_args(args)
    args.batch_size = 1
    args.YCrCb = False
    args.stage = "test"
    device = args.device
    data_module = DInterface(**vars(args), val_rate=0.0)
    data_module.setup(stage="test")

    model = MInterface.load_from_checkpoint(
        load_path, map_location=device, strict=False
    )
    model = model.eval()
    model = model.model
    output_dir = (
        Path(args.result_dir)
        / f"{args.model_name}"
        / f"{Path(load_path).parent.parent.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    test_loader = data_module.test_dataloader()

    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        if type(batch) == list:
            for id in range(len(batch)):
                batch[id] = batch[id].to(device)
        else:
            batch = batch.to(device)

        with torch.no_grad():
            output = model(batch, stage="test")
        # Save the result
        if output.shape[1] == 1:
            writeImgTensor(
                output,
                str(output_dir / f"{idx}.png"),
                YCrCb=args.YCrCb == "YCrCb",
            )

        else:
            for i in range(output.shape[1]):
                #      writeImgTensor(
                #          batch[0][:, i : i + 1],
                #          str(output_dir / f"{idx}_{i}_input.png"),
                #          YCrCb=args.YCrCb == "YCrCb",
                #      )
                writeImgTensor(
                    output[:, i : i + 1],
                    str(output_dir / f"{idx}_{i}.png"),
                    YCrCb=args.YCrCb == "YCrCb",
                )
        #     writeImgTensor(
        #         batch[1][:, i : i + 1],
        #         str(output_dir / f"{idx}_{i}_gt.png"),
        #         gray=True,
        #     )
        ##     writeImgTensor(
        #        batch[2][:, i : i + 1],
        #        str(output_dir / f"{idx}_{i}_demasked.png"),
        #        gray=True,
        #    )
        # writeImgTensor(spike[:, 4:5], str(output_dir / f"{idx}_spike.png"), gray=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    # Restart Control

    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--load_ver", default=None, type=str)
    parser.add_argument("--load_v_num", default=None, type=int)
    parser.add_argument("--load_best", default=False, action="store_true")

    # Training Info
    parser.add_argument("--dataset", default="RGB_spike_sync_data", type=str)
    parser.add_argument("--result_dir", default="./results/", type=str)
    parser.add_argument(
        "--data_dir", default="/mnt/sdb/data/spike/yizhuang/dataset/", type=str
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--model_name", default="RGB_spike_fuse_net", type=str)
    parser.add_argument("--spike_seq_len", default=8, type=int)

    # Add pytorch lightning's args to parser as a group.

    args = parser.parse_args()

    main(args)
