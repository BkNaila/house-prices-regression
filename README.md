# House Price Prediction (Ames, Iowa)

Predicts house sale prices from 79 features using regularized linear models.
Dataset: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Results

| Model | R² (validation) |
|---|---|
| **ElasticNet** | **0.922** |
| Ridge | 0.918 |
| Linear Regression | 0.915 |
| Lasso | 0.897 |

## What's in it
- Missing-value handling, ordinal / binary / one-hot encoding
- Log-transformed target (skewed prices)
- Feature engineering (house age, total area, bathrooms)
- Ridge, Lasso and ElasticNet with tuned alpha
- Predictions on `test.csv` (`submission.csv`)

## Run
```bash
pip install -r requirements.txt
```
Put the Kaggle CSVs in `data/`, then run `src/eda.py`.