<div align="center">
<img alt="Flower" src="./docs/static/images/banner.png" />
</div>


## Installation

Install dependencies using `uv`:

```bash
uv sync
```

## Usage

To train your own `Flower` model, have a look at `flower_standalone.py`. It's a self-contained model file with no dependencies beyond `torch`, so you can drop it directly into your project.

To reproduce results from the paper, use the `flowers.train` entrypoint. It uses a modular configuration system with separate configs for data, model, and training parameters.

### Running Training

```bash
uv run python -m flowers.train \
  --data configs/data/<dataset>.yaml \
  --model configs/models/<model>.yaml \
  --train configs/train.yaml
```

For example, to train a UNet model on the viscoelastic instability dataset:

```bash
uv run python -m flowers.train \
  --data configs/data/viscoelastic_instability.yaml \
  --model configs/models/unet_convnext.yaml \
  --train configs/train.yaml
```

## Acknowledgments

### Model and Code
If you use this repo, please cite the appropriate publications:
- If you use Flower, please cite our paper (currently on hold at arxiv, will be updated ASAP) <!-- TODO: Add here once we have something citeable -->
- The training framework is built on [The Well](https://polymathic-ai.org/the_well/#citation), as are many of the datasets used here.
- The scOT model implementation is adapted from [POSEIDON](https://github.com/camlab-ethz/poseidon?tab=readme-ov-file#citation)

### Data
- Primary datasets are from [The Well](https://polymathic-ai.org/the_well/#citation).
- PDEBench data is taken from [PDEBench](https://github.com/pdebench/PDEBench?tab=readme-ov-file#publications--citations)
- The anisotropic helmholtz dataset is from [WaveBench](https://github.com/wavebench/wavebench?tab=readme-ov-file#citation)
