# Caloric Expenditure Prediction from Physiological Running Metrics

Linear regression model for predicting caloric expenditure during running workouts using lactate threshold heart rate zones and athlete demographics. Built on physiologically grounded single-subject data.

## Repository Structure

```
├── linear_model.py                                # OLS regression model
├── physiologically_grounded_dataset_corrected.csv  # Training dataset
├── requirements.txt
└── .gitignore
```

## Setup

Requires Python 3.10+.

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python linear_model.py
```

Prints MAE, R^2, and per-feature coefficients to stdout.

## Model Details

- **Algorithm**: Ordinary Least Squares (sklearn `LinearRegression`) with `StandardScaler` preprocessing
- **Features**: `gender`, `age`, `weight_kg`, `duration_min`, `time_below_lthr`, `time_above_lthr`
- **Target**: `calories` (kcal)
- **Validation**: 80/20 train/test split (`random_state=42`)
- **Dataset**: Physiologically grounded synthetic dataset corrected against real WHOOP wearable outputs from a single male subject (~25 y/o, ~70 kg)

## Motivation

Energy expenditure models in consumer wearables treat their calorie algorithms as proprietary black boxes. This project explores whether a transparent linear model using time spent above and below lactate threshold heart rate can produce competitive calorie estimates, making the relationship between HR zone intensity and energy cost interpretable.
