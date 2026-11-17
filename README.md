# Sepsis Neuro-Fuzzy Risk Prediction

Semester project: an interpretable first-order Takagi-Sugeno ANFIS-style model for early sepsis-risk estimation.

**Medical disclaimer:** educational/research prototype only; not a diagnostic or treatment system.

## Official dataset
PhysioNet/CinC Challenge 2019 — Early Prediction of Sepsis:
https://physionet.org/content/challenge-2019/1.0.0/

The public expanded training database contains 40,336 subjects. Each patient has an hourly pipe-delimited record with vital signs, laboratory values, demographics and `SepsisLabel`. The challenge targeted early prediction, with the label defined six hours before the sepsis onset used by the challenge.

The clinical dataset is NOT bundled in this ZIP. See `scripts/download_data.md`.

## Exact seven inputs
1. HR — heart rate
2. O2Sat — pulse oximetry
3. Temp — temperature
4. SBP — systolic blood pressure
5. Resp — respiration rate
6. Lactate — lactic acid
7. WBC — white blood cell count

These are all available in the official dataset. They were selected because they are easy to explain and give a useful mixture of vital-sign and laboratory information.

## Fuzzy model
Each input has two trainable Gaussian membership functions: LOW and HIGH.

7 inputs × 2 MFs = **128 rules**.

Gaussian MF:
`mu(x) = exp(-0.5*((x-c)/sigma)^2)`

Rule firing:
`w_r = product_i mu_i(x_i)`

Normalized firing:
`wbar_r = w_r / sum_j(w_j)`

First-order Sugeno consequent:
`f_r = b_r + sum_i(a_ri*x_i)`

Output:
`z = sum_r(wbar_r*f_r)`

Risk:
`sigmoid(z)`

The neuro-fuzzy model learns Gaussian centers, widths, and rule-consequent parameters using gradient descent.

## Architecture
```text
7 clinical inputs
      |
Gaussian membership functions
      |
128 fuzzy-rule firing strengths
      |
normalized rule strengths
      |
128 first-order Sugeno consequents
      |
weighted aggregation
      |
sigmoid
      |
risk score
```

## Software
Python 3.10+, PyTorch, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, Streamlit, joblib.

## Run the included demo
```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py
python -m src.train --data data/demo/sepsis_demo.csv --epochs 40
streamlit run app.py
```

The included data are synthetic and only verify the pipeline.

## Train on PhysioNet
Download the official files and extract them to `data/raw/training/`, then:
```bash
python -m src.train_physionet --data data/raw/training --max-patients 5000 --epochs 20
```

For the final report, use patient-level splitting for real-data evaluation. Do not randomly split hourly records from the same patient across train/test.

## Metrics
Report precision, recall/sensitivity, F1, ROC-AUC, PR-AUC and confusion matrix. Accuracy alone is inadequate for the imbalanced sepsis task.

## Four-person team
1. **Biomedical/medical:** sepsis background, feature rationale, physiology, ethics.
2. **Fuzzy systems:** membership functions, rule firing, Sugeno inference, ANFIS mathematics.
3. **ML/data:** preprocessing, missing values, patient split, training, metrics, baselines.
4. **Software/simulation:** PyTorch integration, Streamlit UI, visualization, testing, packaging.

## Suggested experiments
- Hand-designed FIS demonstration vs learned neuro-fuzzy model.
- ANFIS vs Logistic Regression vs Random Forest.
- 3 vs 5 vs 7 input features.
- Before/after membership-function learning.
- Explain strongest rules for an example patient.

## Final presentation
Problem → dataset → features → fuzzy sets → MFs → rules → ANFIS architecture → training → results → interactive UI → limitations/ethics → future work.
