# TopTimeNet: Topologically-Assisted Time-Series Classification Model

TopTimeNet classifies periodic vs. chaotic time series using a fixed,
non-parametric topological feature extractor (Takens delay embedding +
persistent homology) feeding a lightweight, learnable fusion-and-classification
stage. This repository also includes CNN and Transformer baselines trained
directly on the raw time series, for comparison.

## How to install

To install the package, use:

```bash
sh install.sh
```

## How to generate the dataset

The dataset used in this project is generated from the
[teaspoon](https://teaspoontda.github.io/teaspoon/) library. Scripts live in
`dataset/extended_teaspoon_dataset`.

To generate the dataset locally:

```bash
python generate_dataset_extended_teaspoon.py [path/to/output/dir]
```

To generate it as a SLURM job on an HPC cluster instead:

```bash
sbatch submit_generate_dataset_teaspoon.sh
```

The underlying dataset registry and loading utilities are in
`src/dataset_generation`.

## How to run the ML codes

The machine learning code is organized under `ml_classification`, with two
model families:

### TopTimeNet (topologically-assisted model)

The main model: TDA features (geometric, entropy, lifetime, Betti curve, and
persistence image statistics) computed from a Takens embedding of the raw
signal, followed by a learnable fusion and classification stage.

Code lives in `ml_classification/TDA_stat_summary_time_series_classification`.
See the README in `ml_classification/TDA_stat_summary_time_series_classification/config_related`
for how to generate a config, run a hyperparameter search, and train the
final model.

### CNN and Transformer baselines

Baseline models trained directly on the raw time series, with no
topological feature extraction, used for comparison against TopTimeNet.

Code lives in `ml_classification/transformer_cnn_time_series`.
See the README in `ml_classification/transformer_cnn_time_series/config_related`
for how to generate a config, run a hyperparameter search, and train the
final model.

## Citation

If you find this code useful in your research, please consider citing our
paper:

> Sharareh Sayyad and Sophia Bazzi. *TopTimeNet: Topologically-Assisted
> Time-Series Classification Model*. To appear on arXiv, 2026.

```bibtex
@article{sayyad2026toptimenet,
  title   = {TopTimeNet: Topologically-Assisted Time-Series Classification Model},
  author  = {Sayyad, Sharareh and Bazzi, Sophia},
  journal = {arXiv preprint},
  year    = {2026}
}
```
