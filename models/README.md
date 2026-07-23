# Model Artifacts

This directory stores locally generated model artifacts.

Run the training entry point after placing the raw Kaggle file at
`data/raw/cs-training.csv`:

```bash
python -m src.train
```

The command creates:

- `models/credit_risk_model.pkl`
- `models/model_metadata.json`

Model artifacts are generated locally and are not committed to the repository.
