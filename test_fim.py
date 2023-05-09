# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os

"""Train"""
from megatron.neox_arguments import NeoXArgs
from megatron.training import get_batch, initialize_megatron
from megatron.data.data_utils import build_train_valid_test_data_iterators

if __name__ == "__main__":
    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    initialize_megatron(neox_args=neox_args)
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_train_valid_test_data_iterators(neox_args=neox_args)
    for i in range(10):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            neox_args=neox_args, data_iterator=train_data_iterator
        )
        print(tokens.shape)
        print(labels.shape)
        for seq in tokens:
            print(neox_args.tokenizer.detokenize(seq.tolist()))
            print("=====================================")
