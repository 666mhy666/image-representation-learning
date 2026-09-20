"""PCA, AE and VAE with shared train/validation/test splits.

No source images or trained weights are included. --smoke-test uses generated
noise solely to check tensor shapes/training, not to measure project performance.
"""

from pathlib import Path
import argparse, copy, json, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)).reshape(-1, 1, 64, 64)


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.Sigmoid(),
        )

    def forward(self, x, sample=True):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decoder(z).reshape(-1, 1, 64, 64), mu, logvar


def vae_loss(reconstruction, x, mu, logvar):
    # Mean per image of summed pixel SSE plus KL, matching the source objective.
    reconstruction_loss = (reconstruction - x).square().flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return reconstruction_loss + kl


def split_indices(n, seed):
    if n < 10:
        raise ValueError("At least ten images are required")
    order = np.random.default_rng(seed).permutation(n)
    return (
        order[: int(0.8 * n)],
        order[int(0.8 * n) : int(0.9 * n)],
        order[int(0.9 * n) :],
    )


def reconstruction_mse(model, loader, device, is_vae):
    model.eval()
    total = 0.0
    pixels = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            pred = model(x, sample=False)[0] if is_vae else model(x)
            total += (pred - x).square().sum().item()
            pixels += x.numel()
    return total / pixels


def train(model, train_loader, val_loader, epochs, device, is_vae):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = float("inf")
    state = None
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for (x,) in train_loader:
            x = x.to(device)
            opt.zero_grad()
            if is_vae:
                prediction, mu, logvar = model(x)
                loss = vae_loss(prediction, x, mu, logvar)
            else:
                loss = nn.functional.mse_loss(model(x), x)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(x)
            n += len(x)
        validation = reconstruction_mse(model, val_loader, device, is_vae)
        history.append(
            dict(
                epoch=epoch + 1,
                training_objective=loss_sum / n,
                validation_mse=validation,
            )
        )
        if validation < best:
            best = validation
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return history


def run(a):
    if a.epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.smoke_test:
        torch.set_num_threads(2)
        X = np.random.default_rng(a.seed).uniform(0, 1, (20, 4096)).astype("float32")
    else:
        if not a.images:
            raise ValueError(
                "Provide --images with your locally licensed image directory"
            )
        files = sorted(
            p
            for p in a.images.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )[: a.limit]
        if len(files) < 10:
            raise ValueError("Too few images")
        rows = []
        for file in files:
            with Image.open(file) as im:
                rows.append(
                    np.asarray(
                        im.convert("L").resize((64, 64), Image.Resampling.BILINEAR),
                        dtype="float32",
                    ).reshape(-1)
                    / 255.0
                )
        X = np.stack(rows)
    train_idx, val_idx, test_idx = split_indices(len(X), a.seed)
    assert (
        not set(train_idx) & set(test_idx)
        and not set(train_idx) & set(val_idx)
        and not set(val_idx) & set(test_idx)
    )
    # Same latent dimension for PCA and AE/VAE in this revision.
    dim = min(a.latent_dim, len(train_idx) - 1, X.shape[1])
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=a.seed).fit(
        X[train_idx]
    )
    prediction = pca.inverse_transform(pca.transform(X[test_idx]))
    metrics = [
        dict(
            model="PCA",
            test_pixel_mse=float(np.mean((prediction - X[test_idx]) ** 2)),
            latent_dim=dim,
        )
    ]
    tensor = torch.from_numpy(X.reshape(-1, 1, 64, 64))
    loader = lambda idx, shuffle: DataLoader(
        TensorDataset(tensor[idx]),
        batch_size=a.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(a.seed),
    )
    training = loader(train_idx, True)
    validation = loader(val_idx, False)
    test = loader(test_idx, False)
    device = torch.device(
        "cpu" if a.smoke_test else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    result_dir = a.output / "smoke-test" if a.smoke_test else a.output
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, model, is_vae in [
        ("AE", Autoencoder(dim), False),
        ("VAE", VAE(dim), True),
    ]:
        history = train(model, training, validation, a.epochs, device, is_vae)
        pd.DataFrame(history).to_csv(
            result_dir / f"{name.lower()}-history.csv", index=False
        )
        metrics.append(
            dict(
                model=name,
                test_pixel_mse=reconstruction_mse(model, test, device, is_vae),
                latent_dim=dim,
            )
        )
    pd.DataFrame(metrics).to_csv(result_dir / "reconstruction-metrics.csv", index=False)
    (result_dir / "run.json").write_text(
        json.dumps(
            {
                "data": (
                    "synthetic noise software check"
                    if a.smoke_test
                    else "user supplied images"
                ),
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
                "seed": a.seed,
                "epochs": a.epochs,
                "latent_dim": dim,
                "VAE_evaluation": "decoder at posterior mean; validation checkpoint selected on reconstruction MSE",
            },
            indent=2,
        )
    )
    print(
        "Synthetic smoke test passed; no project performance claim."
        if a.smoke_test
        else pd.DataFrame(metrics).to_string(index=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results"))
    run(p.parse_args())
