# Official dataset instructions

Download the PhysioNet/CinC Challenge 2019 Early Prediction of Sepsis dataset from:

https://physionet.org/content/challenge-2019/1.0.0/

The expanded public training database contains 40,336 subjects. Each patient is one pipe-delimited `.psv` file with hourly measurements.

Extract to:
```text
data/raw/training/
```

Then:
```bash
python -m src.train_physionet --data data/raw/training --max-patients 5000 --epochs 20
```

The clinical dataset is not included in this project ZIP.
