# DeMo
This package contains the supplementary material for DeMo: Decoupled Momentum Optimization

A standalone PyTorch optimizer is provided in `demo.py`.

To reproduce the experiments in the paper, apply `0001-DeMo.patch` to https://github.com/allenai/OLMo/commit/46f06cbc3b42ed94a2400dec4aa479197d1ba0b6.
To launch the training jobs run `torchrun --nodes=8 --nproc-per-node=8 scripts/train.py CONFIG_FILE` where `CONFIG_FILE` is any of the `.yaml` files provided in this package.
