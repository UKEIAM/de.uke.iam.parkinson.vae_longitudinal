import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, Subset
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from vae import VariationAutoencoderModule, WassersteinLoss, MultiCategoricalLoss


class UpdrsData(Dataset):
    def __init__(self, path):
        super().__init__()
        data = pd.read_csv(path, sep="\t").sort_values(["PatientID", "Age"])
        measurements = (
            data[[column for column in data.columns if column.startswith("3.")]]
            .dropna()
            .astype(int)
        )
        self.covariates = data.loc[
            measurements.index,
            [
                "PatientID",
                "Age",
                "Deep brain stimulation available",
                "Deep brain stimulation",
                "Medication",
            ],
        ].reset_index(drop=True)
        self.measurements = torch.tensor(measurements.to_numpy(), dtype=torch.float32)

    def __getitem__(self, index):
        return self.measurements[index]

    def __len__(self):
        return len(self.measurements)

    @property
    def participant_covariate(self) -> str:
        return "PatientID"


class UpdrsDataQoL(Dataset):
    COLUMNS = [f"UPDRS 1.{i}" for i in range(1, 14)] + [
        f"UPDRS 2.{i}" for i in range(1, 14)
    ]

    def __init__(self, path):
        super().__init__()
        data = pd.read_csv(path, sep=",").sort_values(["Participant", "Age"])
        measurements = data[UpdrsDataQoL.COLUMNS].dropna().astype(int)
        self.covariates = data.loc[
            measurements.index,
            ["Participant", "Age"],
        ].reset_index(drop=True)
        self.measurements = torch.tensor(measurements.to_numpy(), dtype=torch.float32)

    def __getitem__(self, index):
        return self.measurements[index]

    def __len__(self):
        return len(self.measurements)

    @property
    def participant_covariate(self) -> str:
        return "Participant"


class UpdrsDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        percentage_subjects_in_valid_dataset: float,
        batch_size: int,
    ):
        super().__init__()
        assert 0 < percentage_subjects_in_valid_dataset <= 1

        self.data = dataset
        self.batch_size = batch_size

        if percentage_subjects_in_valid_dataset < 1:
            patients = self.data.covariates[dataset.participant_covariate].unique()
            num_patients_valid = int(
                len(patients) * percentage_subjects_in_valid_dataset
            )
            first_patient_valid = len(patients) - num_patients_valid
            # Find the index of the first patient in valid set
            self.val_start = self.data.covariates[
                self.data.covariates[dataset.participant_covariate]
                == patients[first_patient_valid]
            ].index[0]
        else:
            self.val_start = 0

    def calculate_class_weights(self):
        return torch.tensor(
            compute_class_weight(
                "balanced",
                classes=range(5),
                y=self.data.measurements[: self.val_start].flatten().long().numpy(),
            )
        ).float()

    def train_dataloader(self):
        if self.val_start == 0:
            raise ValueError("Only the validation set is used.")
        return DataLoader(
            Subset(self.data, range(0, self.val_start)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.data, range(self.val_start, len(self.data))),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )


def objective(trial: optuna.trial.Trial) -> float:
    NAME = "updrs_qol_optuna"

    bottleneck_size = trial.suggest_int("bottleneck_size", 6, 20)
    n_layers = trial.suggest_int("n_layers", 1, 6)
    reg_weight = trial.suggest_int("reg_weight", 10, 150)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    use_ordinal = trial.suggest_categorical("use_ordinal", [True, False])

    layers = list(
        reversed([bottleneck_size * i * 2 for i in range(1, n_layers + 1)])
    ) + [bottleneck_size]

    data = UpdrsDataQoL(
        "/workspaces/de.uke.iam.parkinson.vae_longitudinal/data/updrs_amp.csv"
    )
    data_module = UpdrsDataModule(
        data,
        percentage_subjects_in_valid_dataset=0.2,
        batch_size=512,
    )

    reconstruction_loss = MultiCategoricalLoss(
        n_values=len(UpdrsDataQoL.COLUMNS),
        n_classes=5,
        is_categorical=False if use_ordinal else True,
        is_ordinal=True if use_ordinal else False,
        weight=data_module.calculate_class_weights().to("cuda"),
    )

    generative_loss = WassersteinLoss(
        reg_weight=reg_weight, kernel_type="imq", z_var=2.0
    )
    model = VariationAutoencoderModule(
        reconstruction_loss,
        generative_loss,
        layers,
        patience=80,
        learning_rate=1e-3,
        dropout=dropout,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{NAME}_{trial.number}",
        save_top_k=1,
        verbose=True,
        monitor="val_concordance",
        mode="max",
    )
    early_stopping = EarlyStopping(monitor="val_concordance", patience=120, mode="max")
    logger = TensorBoardLogger("logs", name=NAME)

    # Initialize the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=1000,
        callbacks=[
            early_stopping,
            checkpoint_callback,
            PyTorchLightningPruningCallback(trial, monitor="val_concordance"),
        ],
        logger=logger,
        log_every_n_steps=6,
    )

    trainer.fit(model, data_module)
    return trainer.callback_metrics["val_concordance"].item()


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=250)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
