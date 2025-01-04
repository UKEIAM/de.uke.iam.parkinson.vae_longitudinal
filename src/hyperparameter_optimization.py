import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from vae import VariationAutoencoderModule, WassersteinLoss, MultiCategoricalLoss
from data import UpdrsDataQoL, UpdrsDataModule


def objective(trial: optuna.trial.Trial) -> float:
    NAME = "updrs_qol_optuna_val_loss"

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
        dirpath="checkpoints_val_loss/",
        filename=f"{NAME}_{trial.number}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=120, mode="min")
    logger = TensorBoardLogger("logs", name=NAME)

    # Initialize the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=1000,
        callbacks=[
            early_stopping,
            checkpoint_callback,
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ],
        logger=logger,
        log_every_n_steps=6,
    )

    trainer.fit(model, data_module)
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=250)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
