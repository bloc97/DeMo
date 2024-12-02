# DeMo
This package contains the supplementary material for [DeMo: Decoupled Momentum Optimization](https://arxiv.org/abs/2411.19870) (arXiv)

A standalone PyTorch optimizer is provided in `demo.py`.

To reproduce the experiments in the paper, apply `0001-DeMo.patch` to https://github.com/allenai/OLMo/commit/46f06cbc3b42ed94a2400dec4aa479197d1ba0b6.
To launch the training jobs run `torchrun --nodes=8 --nproc-per-node=8 scripts/train.py CONFIG_FILE` where `CONFIG_FILE` is any of the `.yaml` files provided in this package.

For implementation in other PyTorch training pipelines, the standalone DeMo optimizer can be used as-is, the only additional modification needed is to disable the native Distributed Data Parallel gradient synchronization/all-reduce.

Future updates will be on the [DisTrO](https://github.com/NousResearch/DisTrO) repo.
