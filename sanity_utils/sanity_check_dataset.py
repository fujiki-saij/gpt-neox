from megatron.data import build_the_dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_prefix",
    type=str,
    required=True,
    help="The directory where the data is stored.",
)


args = parser.parse_args()
data_prefix = args.data_prefix


dataset_built = build_the_dataset(
    data_prefix=data_prefix,
    name=data_prefix.split("/")[-1],
    data_impl="mmap",
)
print(dataset_built)