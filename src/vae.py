from dataclasses import dataclass
from typing import Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor

import lightning as L
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from torchmetrics.functional import concordance_corrcoef


@dataclass(slots=True)
class MultiCategoricalLoss:
    n_values: int
    n_classes: int
    is_categorical: bool
    is_ordinal: bool
    weight: torch.Tensor = None

    @property
    def last_layer_size(self) -> int:
        if self.is_categorical and self.is_ordinal:
            raise ValueError("Cannot be both categorical and ordinal.")

        if self.is_categorical:
            return self.n_values * self.n_classes
        elif self.is_ordinal:
            return self.n_values * (self.n_classes - 1)
        else:
            return self.n_values

    def calculate_reconstruction(self, y_raw: torch.Tensor) -> torch.LongTensor:
        if self.is_categorical:
            return y_raw.reshape((-1, self.n_values, self.n_classes)).argmax(dim=-1)
        elif self.is_ordinal:
            return corn_label_from_logits(
                y_raw.reshape(-1, self.n_classes - 1)
            ).reshape((-1, self.n_values))
        else:
            return (F.sigmoid(y_raw) * (self.n_classes - 1)).round().long()

    def calculate_reconstruction_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        if self.is_categorical:
            y_pred = y_pred.reshape((-1, self.n_values, self.n_classes))
            return F.cross_entropy(
                input=y_pred.reshape(-1, self.n_classes),
                target=y_true.reshape(-1).long(),
                weight=self.weight,
                reduction="mean",
            )
        elif self.is_ordinal:
            y_pred = y_pred.reshape((-1, self.n_values, self.n_classes - 1))
            return corn_loss(
                logits=y_pred.reshape(-1, self.n_classes - 1),
                y_train=y_true.reshape(-1).long(),
                num_classes=self.n_classes,
            )
        else:
            return (
                F.mse_loss(
                    input=F.sigmoid(y_pred),
                    target=y_true / (self.n_classes - 1),
                    reduction="none",
                )
                .sum(axis=-1)
                .mean()
            )


@dataclass(slots=True)
class VariationalAutoencoderOutput:
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.LongTensor

    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_generative: torch.Tensor

    def log(self, logger, batch, prefix: str):
        logger.log(f"{prefix}_loss", self.loss)
        logger.log(f"{prefix}_loss_recon", self.loss_recon)
        logger.log(f"{prefix}_loss_generative", self.loss_generative)

        # Log evaluation metrices
        logger.log(
            f"{prefix}_rel_wrong_items",
            ((batch != self.x_recon).sum(axis=-1) / batch.shape[-1]).median(),
        )
        real_sum = batch.sum(axis=-1)
        reconstructed_sum = self.x_recon.sum(axis=-1)

        concordance = concordance_corrcoef(
            target=real_sum.to(torch.float), preds=reconstructed_sum.to(torch.float)
        )
        if torch.isnan(concordance):
            concordance = torch.tensor(0.0)
        logger.log(f"{prefix}_concordance", concordance)

    def calculate_total_error(self, batch):
        real_sum = batch.sum(axis=-1)
        reconstructed_sum = self.x_recon.sum(axis=-1)
        return F.l1_loss(input=reconstructed_sum, target=real_sum, reduction="mean")


class GenerativeLoss:
    def __call__(self, dist: torch.distributions.Distribution, z: Tensor):
        raise NotImplementedError()


class KullbackLeiblerLoss(GenerativeLoss):
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, dist: torch.distributions.Distribution, z: Tensor):
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(dist.mean, device=dist.mean.device),
            scale_tril=torch.eye(dist.mean.shape[-1], device=dist.mean.device)
            .unsqueeze(0)
            .expand(dist.mean.shape[0], -1, -1),
        )
        return self.beta * torch.distributions.kl.kl_divergence(dist, std_normal).mean()


class WassersteinLoss(GenerativeLoss):
    def __init__(self, reg_weight: float, kernel_type: str, z_var: float):
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = z_var

    def __call__(self, dist: torch.distributions.Distribution, z: Tensor):
        # Calculate the corrected reg_weight
        batch_size = z.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = (
            reg_weight * prior_z__kernel.mean()
            + reg_weight * z__kernel.mean()
            - 2 * reg_weight * priorz_z__kernel.mean()
        )
        return mmd

    def compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == "rbf":
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == "imq":
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError("Undefined kernel type.")

        return result

    def compute_rbf(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(
        self, x1: Tensor, x2: Tensor, eps: float = 1e-7
    ) -> Tensor:
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result


class Encoder(nn.Module):
    def __init__(self, layers: Sequence[int], dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential()
        for i, (i_size, out_size) in enumerate(zip(layers, layers[1:])):
            if i < len(layers) - 2:
                self.encoder.append(nn.Linear(i_size, out_size))
                if dropout > 0.0:
                    self.encoder.append(nn.Dropout(dropout))
                self.encoder.append(nn.SiLU())
            else:
                self.encoder.append(
                    nn.Linear(i_size, 2 * out_size)
                )  # 2 for mean and variance.
        self.softplus = nn.Softplus()

    def forward(self, x, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        return mu, scale

    def forward_and_reparameterize(
        self, x
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        mu, scale = self(x)
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        return dist, dist.rsample()


class Decoder(nn.Module):
    def __init__(self, layers: Sequence[int], dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential()
        for i, (i_size, out_size) in enumerate(zip(layers, layers[1:])):
            self.decoder.append(nn.Linear(i_size, out_size))
            if i < len(layers) - 2:
                if dropout > 0.0:
                    self.decoder.append(nn.Dropout(dropout))
                self.decoder.append(nn.SiLU())

    def forward(self, z):
        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        reconstruction_loss: MultiCategoricalLoss,
        generative_loss: GenerativeLoss,
        intermediate_layers: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.generative_loss = generative_loss
        self.encoder = Encoder(
            [reconstruction_loss.n_values] + intermediate_layers, dropout=dropout
        )
        self.decoder = Decoder(
            list(reversed(intermediate_layers)) + [reconstruction_loss.last_layer_size],
            dropout=dropout,
        )

    def forward(self, x, compute_loss: bool = True):
        dist, z = self.encoder.forward_and_reparameterize(x)
        recon_x = self.decoder(z)

        if not compute_loss:
            return VariationalAutoencoderOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=self.reconstruction_loss.calculate_reconstruction(recon_x),
                loss=None,
                loss_recon=None,
                loss_generative=None,
            )

        loss_recon = self.reconstruction_loss.calculate_reconstruction_loss(
            y_true=x, y_pred=recon_x
        )
        loss_generative = self.generative_loss(dist, z)

        loss = loss_recon + loss_generative
        return VariationalAutoencoderOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=self.reconstruction_loss.calculate_reconstruction(recon_x),
            loss=loss,
            loss_recon=loss_recon,
            loss_generative=loss_generative,
        )


class VariationAutoencoderModule(L.LightningModule):
    def __init__(
        self,
        reconstruction_loss: MultiCategoricalLoss,
        generative_loss: GenerativeLoss,
        intermediate_layers: Sequence[int],
        learning_rate: float,
        patience: int,
        dropout: float,
    ):
        super().__init__()
        self.model = VariationalAutoencoder(
            reconstruction_loss, generative_loss, intermediate_layers, dropout=dropout
        )
        self.learning_rate = learning_rate
        self.patience = patience
        self.dropout = dropout
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        output = self.model(batch, compute_loss=True)
        output.log(self, batch, "train")
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch, compute_loss=True)
        output.log(self, batch, "val")
        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", patience=self.patience, min_lr=1e-6
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
