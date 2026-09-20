# Comparing PCA, Autoencoders, and VAEs

Compare linear and nonlinear image representations using the same splits, latent dimension, and reconstruction metric.

**Author:** Heyang Ma · Independent UCLA graduate course project, revised for this portfolio.
**Tools:** Python / PyTorch, Representation learning, Held-out reconstruction. **Scope:** 10,000 grayscale images.

## Question

How do PCA, a dense autoencoder, and a variational autoencoder compare at the same latent dimension?

## What I did

I created a shared 80/10/10 image split, matched the latent dimension across methods, selected neural-network checkpoints on validation reconstruction error, and evaluated every model with held-out pixel MSE.

## Main finding

The revised experiment uses 8,000 training, 1,000 validation, and 1,000 test images, with 64 latent dimensions and ten training epochs. Held-out pixel MSE was 0.00750 for PCA, 0.01677 for the autoencoder, and 0.03600 for the VAE. PCA performed best under this training budget and metric. No images or fitted weights are distributed.

![Held-out pixel MSE for three 64-dimensional representations; lower is better.](results/reconstruction-comparison.png)

_Held-out pixel MSE for three 64-dimensional representations; lower is better._

## Important limitations

The original notebook showed training-image reconstructions at unequal latent dimensions. The revised experiment is a separate controlled comparison, not confirmation of those original results. Random image splits do not ensure identity separation in CelebA. Pixel MSE does not measure perceptual quality or privacy. The VAE is evaluated at its posterior mean.

## Code and reproducibility

Install Python dependencies with `python -m pip install -r requirements.txt`.
Read [data access and input requirements](DATA_ACCESS.md), then run from this repository:

```sh
python analysis.py --images /path/to/authorized/images --limit 10000 --epochs 10 --output results
```

After the analysis finishes, run `python plot_results.py` to regenerate the aggregate comparison figure. Smoke tests write only to `results/smoke-test/`, separate from formal result metadata.


The executable analysis is [analysis.py](analysis.py). See [REPORT.md](REPORT.md) for model details and interpretation, [DATA_ACCESS.md](DATA_ACCESS.md) for inputs, and [REVISION_NOTES.md](REVISION_NOTES.md) for the distinction between the course project and portfolio revision. The committed `results/` files are generated summaries from the revision.

