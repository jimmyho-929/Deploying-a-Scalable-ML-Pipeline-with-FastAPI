# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model type:** Logistic Regression (via scikit-learn), wrapped in a StandardScaler pipeline
- **Version:** 1.0
- **Training framework:** scikit-learn
- **Saved artifacts:** `model/model.pkl`, `model/encoder.pkl`

## Intended Use
- **Primary use:** Binary classification to predict whether an individual earns more or less than $50K/year based on census data.
- **Intended users:** Data scientists and ML engineers exploring income prediction models.
- **Out-of-scope uses:** This model should not be used for making real financial or employment decisions about individuals.

## Training Data
- **Source:** UCI Census Income dataset (`census.csv`)
- **Size:** ~32,000 rows after cleaning
- **Split:** 80% training / 20% test (`random_state=42`)
- **Categorical features encoded** with OneHotEncoder:
  - `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- **Label:** `salary` — binarized to `>50K` (1) or `<=50K` (0) using LabelBinarizer

## Evaluation Data
- The test set is the held-out 20% split from the same census dataset.
- Processed using the same encoder and label binarizer fitted on training data (`training=False`).

## Metrics
Performance is evaluated using **Precision**, **Recall**, and **F1 score** (beta=1).

| Metric    | Score  |
|-----------|--------|
| Precision | 0.7376 |
| Recall    | 0.6288 |
| F1        | 0.6789 |

Slice performance is recorded in `slice_output.txt`. Example slices:

| Feature   | Value       | Precision | Recall | F1     |
|-----------|-------------|-----------|--------|--------|
| workclass | Federal-gov | 0.8197    | 0.7353 | 0.7752 |
| workclass | Private     | 0.7381    | 0.6245 | 0.6766 |
| workclass | State-gov   | 0.8136    | 0.7059 | 0.7559 |
| sex       | Male        | ~0.74     | ~0.64  | ~0.69  |
| sex       | Female      | ~0.72     | ~0.55  | ~0.62  |

## Ethical Considerations
- The dataset contains sensitive demographic attributes such as `race`, `sex`, and `native-country`. The model may reflect historical biases present in census data.
- Performance varies across demographic slices (see `slice_output.txt`), which means the model may be less accurate for certain groups.
- This model should **not** be used to make decisions that impact individuals' financial opportunities or employment without careful bias auditing.

## Caveats and Recommendations
- The model is a simple logistic regression baseline. Tree-based models (e.g., Random Forest, XGBoost) would likely yield higher performance.
- Feature engineering (e.g., binning `age`, interaction terms) could improve recall, which is the weakest metric.
- Slice performance should be reviewed before any real-world deployment to ensure fairness across demographic groups.
- The census data reflects a historical snapshot and may not generalize to current income distributions.

