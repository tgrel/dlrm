# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset
import data_loader_terabyte
from dlrm_data_pytorch import *
import os.path as path
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini_batch_size", type=int, default=2048)
    parser.add_argument("--test_mini_batch_size", type=int, default=4096)
    parser.add_argument("--raw_data_file", type=str)
    args = parser.parse_args()

    data_directory = path.dirname(args.raw_data_file)
    data_filename = args.raw_data_file.split("/")[-1]

    new_train_loader = data_loader_terabyte.DataLoader(
        data_directory=data_directory,
        data_filename=data_filename,
        days=list(range(23)),
        batch_size=args.mini_batch_size,
        split="train"
    )

    new_test_loader = data_loader_terabyte.DataLoader(
        data_directory=data_directory,
        data_filename=data_filename,
        days=[23],
        batch_size=args.test_mini_batch_size,
        split="test"
    )

    train_data = CriteoDataset(
        dataset='terabyte',
        max_ind_range=10 * 1000 * 1000,
        sub_sample_rate=1,
        randomize=True,
        split="train",
        raw_path=args.raw_data_file,
        pro_data='dummy_string',
        memory_map=True
    )

    test_data = CriteoDataset(
        dataset='terabyte',
        max_ind_range=10 * 1000 * 1000,
        sub_sample_rate=1,
        randomize=True,
        split="test",
        raw_path=args.raw_data_file,
        pro_data='dummy_string',
        memory_map=True
    )

    old_train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,
    )
    old_test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,
    )

    for idx, batch in enumerate(zip(old_train_loader, new_train_loader)):
        batch_old, batch_new = batch
        print('train batch idx: ', idx)

        for j, (old_elem, new_elem) in enumerate(zip(batch_old, batch_new)):
            diff_norm = np.linalg.norm(old_elem - new_elem)
            print('\t', j, 'diff norm: ', diff_norm)
            if not np.allclose(old_elem, new_elem):
                raise ValueError('Train data loading discrepancy')

    for idx, batch in enumerate(zip(old_test_loader, new_test_loader)):
        batch_old, batch_new = batch
        print('test batch idx: ', idx)

        for j, (old_elem, new_elem) in enumerate(zip(batch_old, batch_new)):
            diff_norm = np.linalg.norm(old_elem - new_elem)
            print('\t', j, 'diff norm: ', diff_norm)
            if not np.allclose(old_elem, new_elem):
                raise ValueError('Test data loading discrepancy')


if __name__ == '__main__':
    main()