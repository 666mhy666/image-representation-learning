# Data access

Use your authorized CelebA image directory. The local benchmark uses the first 10,000 filenames in sorted order from the supplied archive, resized to 64 × 64 grayscale and scaled to [0, 1]. Obtain images from the original provider under its terms; images and model weights are not included.

## Input contract

A directory of JPEG or PNG images. The run records split sizes, seed, latent dimension, and epochs.

Source context: [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Keep inputs in a local `data/` directory. Input data and model files are excluded from the public release. Course access is not assumed to confer redistribution permission.
