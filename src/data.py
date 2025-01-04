import pandas as pd
import lightning as L

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.utils.class_weight import compute_class_weight


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
