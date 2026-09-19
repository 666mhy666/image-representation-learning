# Comparing PCA, Autoencoders, and VAEs

Compare linear and nonlinear image representations using the same splits, latent dimension, and reconstruction metric.

**Author:** Heyang Ma · Independent UCLA graduate course project, revised for this portfolio.
**Tools:** Python / PyTorch, Representation learning, Held-out reconstruction. **Scope:** 10,000 grayscale images.

## Question and result

How do PCA, a dense autoencoder, and a variational autoencoder compare at the same latent dimension?

The revised experiment uses 8,000 training, 1,000 validation, and 1,000 test images, with 64 latent dimensions and ten training epochs. Held-out pixel MSE was 0.00750 for PCA, 0.01677 for the autoencoder, and 0.03600 for the VAE. PCA performed best under this training budget and metric. No images or fitted weights are distributed.

![Main result](results/reconstruction-comparison.png)

## What the analysis does

The executable analysis is [analysis.py](analysis.py). [Methods and interpretation](REPORT.md) explains the scope; [revision notes](REVISION_NOTES.md) distinguish the original analysis from the portfolio revision.

## Run locally

Install Python dependencies with `python -m pip install -r requirements.txt`.
Read [data access and input requirements](DATA_ACCESS.md), then run from this repository:

```sh
python analysis.py --images /path/to/authorized/images --limit 10000 --epochs 10 --output results
```

## Results and limits

The original notebook showed training-image reconstructions at unequal latent dimensions. The revised experiment is a separate controlled comparison, not confirmation of those original results. Random image splits do not ensure identity separation in CelebA. Pixel MSE does not measure perceptual quality or privacy. The VAE is evaluated at its posterior mean.

The committed `results/` files are generated summaries from the portfolio revision. Source records, credentials, fitted models, and original notebook outputs are excluded.

