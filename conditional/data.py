import pandas as pd

GENERATIVE_COLUMNS = ["UPDRS I", "UPDRS II", "UPDRS III", "UPDRS IV", "PDQ", "MoCA"]
MAX_SCORES_UPDS = {
    "Mobility": 40,
    "Daily living": 24,
    "Emotion": 24,
    "Stigmatization": 16,
    "Social support": 12,
    "Cognition": 16,
    "Communication": 12,
    "Bodily discomfort": 12,
}

MAX_SCORES_UPDRS = {
    "UPDRS I": 52,
    "UPDRS II": 52,
    "UPDRS III": 132,
    "UPDRS IV": 24,
}


def load_amp(path: str):
    X_amp = pd.read_csv(path, na_values="Unknown")

    # Filter out multiple measurements
    # X_amp = X_amp[X_amp["Visit ID"] == "M0"].reset_index()

    X_amp.columns = X_amp.columns.str.replace("ADL", "Daily living", regex=False)
    X_amp.columns = X_amp.columns.str.replace("Stigma", "Stigmatization", regex=False)
    X_amp.columns = X_amp.columns.str.replace("Social", "Social support", regex=False)
    X_amp.columns = X_amp.columns.str.replace(
        "Discomfort", "Bodily discomfort", regex=False
    )

    value_map = {
        "Never": 0,
        "Occasionally": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always or cannot do at all": 4,
    }

    # Convert relevant columns using the map to numerical values
    for col in X_amp.columns:
        if col.startswith("PDQ39"):
            X_amp[col] = X_amp[col].map(value_map).astype(pd.Int64Dtype())

    for key, value in MAX_SCORES_UPDS.items():
        selected_columns = X_amp.filter(like=key)
        assert len(selected_columns.columns) * 4 == value
        X_amp[key] = selected_columns.sum(axis=1, skipna=False)
        X_amp[key] = (X_amp[key] * 100) / value
        X_amp = X_amp.drop(columns=selected_columns.columns)

    # Cast the UPDRS scoresx
    for col in X_amp.columns:
        if col.startswith("UPDRS"):
            X_amp[col] = X_amp[col].astype(pd.Int64Dtype())

    X_amp["PDQ"] = X_amp[list(MAX_SCORES_UPDS.keys())].mean(axis=1)
    X_amp["MoCA"] = X_amp["MoCA"] / 30 * 100

    for uprds_subscore, max_value in MAX_SCORES_UPDRS.items():
        X_amp[uprds_subscore] = pd.to_numeric(
            X_amp[uprds_subscore] / max_value * 100
        ).astype(float)

    X_amp = X_amp.dropna(subset=GENERATIVE_COLUMNS, thresh=1)

    scores = pd.DataFrame(
        {
            "Time since diagnosis": ((X_amp["Age"] - X_amp["Age (Diagnosis)"]) / 20) * 2
            - 1,
            "Age": X_amp["Age"] / 100 * 2 - 1,
            "Sex": X_amp["Sex"].map({"Female": -1.0, "Male": 1.0}),
            "Medication": X_amp["Medication"].map({True: 1.0, False: -1.0}),
            "Surgery": X_amp["Surgery"].map({True: 1.0, False: -1.0}),
            "Education": X_amp["Education"].map(
                {
                    "Less than 12 years": -1.0,
                    "12-16 years": 0.0,
                    "Greater than 16 years": 1.0,
                }
            ),
        }
    )

    return X_amp, scores


def load_uke(path: str):
    SCORES = ["UPDRS I", "UPDRS II", "UPDRS III", "UPDRS IV", "PDQ", "MoCA"]
    COVARIATES = ["Age (Diagnosis)", "Age", "Sex", "Education", "Medication"]

    data = pd.read_csv(
        path,
        sep=",",
        na_values=[
            "Keine_Angabe",
            "Nicht_durchgeführt",
            "Keine_Angaben",
            "Keine_angabe",
        ],
    )

    def education_mapper(age):
        if pd.isna(age):
            return pd.NA
        elif age < 12:
            return -1.0
        elif age > 16:
            return 1.0
        else:
            return 0.0

    def normalize_scores(scores):
        for key, value in MAX_SCORES_UPDRS.items():
            scores[key] = scores[key] / value * 100
        scores["MoCA"] *= 100
        return scores

    scores = []
    targets = []
    for _, group in data.groupby("Patient"):
        group = group.sort_values("Month").reset_index(drop=True)

        baseline = group[group["Month"] == 0][SCORES + COVARIATES + ["Month"]].dropna(
            thresh=len(SCORES) - 3
        )
        treated = group[group["Month"] > 0][SCORES + ["Month"]].dropna()
        if len(baseline) == 0 or len(treated) == 0:
            continue

        baseline = baseline.iloc[0]
        baseline["Time since last test"] = (
            treated.iloc[0]["Month"] - baseline["Month"]
        ) / 12
        baseline["Treatment"] = 1.0
        scores.append(
            baseline[SCORES + COVARIATES + ["Time since last test", "Treatment"]]
        )

        targets.append(treated.iloc[0][SCORES])

    uke_scores = pd.DataFrame(scores).reset_index(drop=True)
    uke_covariates = uke_scores[COVARIATES].copy()
    uke_covariates["Education"] = pd.to_numeric(
        uke_covariates["Education"].map(education_mapper), errors="coerce"
    )
    uke_covariates["Sex"] = uke_covariates["Sex"].map({1.0: -1.0, 0.0: 1.0})
    uke_covariates["Time since diagnosis"] = (
        (uke_covariates["Age"] * 100 - uke_covariates["Age (Diagnosis)"]) / 20
    ) * 2 - 1
    uke_covariates["Medication"] = uke_covariates["Medication"].map(
        {"ON": 1.0, "OFF": -1.0}
    )
    uke_covariates["Surgery"] = -1.0
    uke_covariates["Age"] = uke_covariates["Age"] * 2 - 1

    uke_scores = uke_scores.drop(
        columns=COVARIATES + ["Time since last test", "Treatment"]
    )
    uke_scores = normalize_scores(uke_scores)

    uke_targets = pd.DataFrame(targets).reset_index(drop=True)
    uke_targets = normalize_scores(uke_targets)

    return (
        uke_scores,
        uke_covariates[
            ["Time since diagnosis", "Age", "Sex", "Medication", "Surgery", "Education"]
        ],
        uke_targets,
    )
