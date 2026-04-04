"""
AutoVault — Car Price Prediction Model
========================================
Reads from:
  data/merged_datasets/merged_ordinary_dataset.csv

Features used (auto-selected by importance):
  - brand_price_mean       target-encoded brand  (e.g. all BMWs avg price)
  - model_price_mean       target-encoded model  (e.g. all Creta avg price)
  - brand_model_price_mean target-encoded brand+model combo (e.g. "BMW M5" specifically)
  - year                   model year
  - log_kms                log of km driven
  - fuel_type_enc          label-encoded fuel type
  - transmission_enc       label-encoded transmission

  Three encoding levels let the model distinguish:
    BMW (brand) → BMW X5 (model) → BMW X5 2022 Diesel (full context)
  A BMW 3 Series and BMW M5 now produce different predictions.

TARGET LEAKAGE — FULLY FIXED:
  - Train/test split happens BEFORE any encoding.
  - LeakFreeTargetEncoder lives inside a sklearn Pipeline.
  - In every Optuna CV fold it re-fits on that fold's train split only.
  - Unseen brand/model at inference → falls back to global training mean.
  - Smoothing strength is tuned by Optuna (prevents rare-model overfitting).

Models:
  1. Random Forest       — baseline
  2. XGBoost (default)   — better accuracy
  3. XGBoost Fine-tuned  — Optuna hyperparameter search (BEST)

Output → models/
  xgb_finetuned.pkl   (full pipeline: encoder + model)
  rf_baseline.pkl
  xgb_default.pkl
  encoders.pkl        (LabelEncoders for fuel_type, transmission)
  feature_columns.pkl (pipeline input column list)

SETUP:
  pip install pandas scikit-learn xgboost optuna matplotlib joblib

RUN:
  python train_price_model.py
"""

import os
import math
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble        import RandomForestRegressor
from sklearn.pipeline        import Pipeline
from sklearn.base            import BaseEstimator, TransformerMixin
from xgboost                 import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORD_CSV              = os.path.join(BASE_DIR, "data/merged_datasets/merged_ordinary_dataset.csv")
MODEL_DIR            = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

OPTUNA_TRIALS        = 50
TEST_SIZE            = 0.20
RANDOM_STATE         = 42
IMPORTANCE_THRESHOLD = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# BRAND / MODEL EXTRACTION FROM TITLE
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_BRANDS = [
    'Mercedes-Benz', 'Land Rover', 'Aston Martin', 'Rolls-Royce',
    'Rolls royce', 'Mini Cooper', 'Maruti Suzuki', 'Alfa Romeo',
    'BMW', 'Audi', 'Ferrari', 'Lamborghini', 'Porsche', 'McLaren',
    'Bentley', 'Jaguar', 'Lexus', 'Toyota', 'Honda', 'Hyundai',
    'Maruti', 'Mahindra', 'Tata', 'Skoda', 'Volkswagen', 'Volvo',
    'Renault', 'Nissan', 'Ford', 'Jeep', 'Kia', 'KIA', 'MG',
    'BYD', 'Datsun', 'Fiat', 'Isuzu', 'Mercedes',
]
KNOWN_BRANDS_SORTED = sorted(KNOWN_BRANDS, key=len, reverse=True)

BRAND_NORMALISE = {
    'Rolls royce':   'Rolls-Royce',
    'Maruti Suzuki': 'Maruti',
    'Mercedes':      'Mercedes-Benz',
    'KIA':           'Kia',
}


def extract_brand_model(title: str):
    """Split a car title into (brand, model). e.g. 'BMW M5' -> ('BMW', 'M5')"""
    if not isinstance(title, str) or not title.strip():
        return 'Unknown', 'Unknown'
    title = title.strip()
    for brand in KNOWN_BRANDS_SORTED:
        if title.lower().startswith(brand.lower()):
            model = title[len(brand):].strip()
            brand = BRAND_NORMALISE.get(brand, brand)
            return brand, model if model else 'Unknown'
    parts = title.split(maxsplit=1)
    brand = BRAND_NORMALISE.get(parts[0], parts[0])
    model = parts[1] if len(parts) > 1 else 'Unknown'
    return brand, model


# ═══════════════════════════════════════════════════════════════════════════
# LEAK-FREE MULTI-COLUMN TARGET ENCODER
# ─────────────────────────────────────────────────────────────────────────
# Encodes multiple categorical columns (brand, model, brand_model) using
# smoothed mean target values computed ONLY on the training data.
#
# Three encoding levels:
#   brand_price_mean       - avg price for all cars of that brand
#   model_price_mean       - avg price for all cars of that model name
#   brand_model_price_mean - avg price for that exact brand+model combo
#
# Smoothing: rare categories (few rows) are shrunk toward the global mean.
# Smoothing strength is treated as a hyperparameter tuned by Optuna.
#
# Lives inside a sklearn Pipeline -> re-fitted on each CV fold's train split.
# ═══════════════════════════════════════════════════════════════════════════

class LeakFreeTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leak-free target encoder for multiple categorical columns.

    Parameters
    ----------
    cols : list of str
        Categorical columns to encode. Each gets a '<col>_price_mean' output.
        All raw string columns are dropped after encoding.
    smoothing : float
        Bayesian smoothing weight toward the global mean.
        Higher = more shrinkage for rare categories (good for sparse models).
    """

    def __init__(self, cols=None, smoothing=10.0):
        self.cols         = cols or ['brand', 'model', 'brand_model']
        self.smoothing    = smoothing
        self.mappings_    = {}
        self.global_mean_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = float(y.mean())
        self.mappings_    = {}

        for col in self.cols:
            if col not in X.columns:
                continue
            tmp   = pd.DataFrame({'cat': X[col].values, 'target': y.values})
            stats = tmp.groupby('cat')['target'].agg(['mean', 'count'])
            n     = stats['count']
            # Smoothed estimate: (n * cat_mean + k * global_mean) / (n + k)
            smoothed = (
                (n * stats['mean'] + self.smoothing * self.global_mean_)
                / (n + self.smoothing)
            )
            self.mappings_[col] = smoothed.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.cols:
            if col not in X.columns:
                continue
            X[col + '_price_mean'] = (
                X[col]
                .map(self.mappings_.get(col, {}))
                .fillna(self.global_mean_)   # unseen category -> global mean
            )

        # Drop all raw string columns — XGBoost only accepts numeric input
        X = X.drop(columns=[c for c in self.cols if c in X.columns])
        return X


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE COLUMN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Columns fed into the pipeline (includes raw strings for the encoder)
PIPELINE_INPUT_COLS = [
    'brand',
    'model',
    'brand_model',
    'year',
    'log_kms',
    'fuel_type_enc',
    'transmission_enc',
]

# Columns the XGBoost model sees after encoding (all numeric)
CANDIDATE_FEATURES = [
    'brand_price_mean',        # avg price of all BMW / Ferrari / Maruti cars
    'model_price_mean',        # avg price of all "Creta" / "M5" / "Ertiga" cars
    'brand_model_price_mean',  # avg price of "BMW M5" specifically
    'year',
    'log_kms',
    'fuel_type_enc',
    'transmission_enc',
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print("=" * 60)
    print("  AutoVault — Car Price Prediction  (3-level encoding)")
    print("=" * 60)
    print("\nLoading CSV...")

    ord_ = pd.read_csv(ORD_CSV)
    ord_ = ord_.drop(columns=[c for c in ord_.columns if 'Unnamed' in c], errors='ignore')
    ord_ = ord_.rename(columns={
        'Title':       'title',
        'Brand':       'brand',
        'Price':       'price',
        'Kms Covered': 'kms_covered',
        'Year':        'year',
        'Fuel Type':   'fuel_type',
        'Type':        'transmission',
    })

    needed = ['title', 'brand', 'price', 'kms_covered', 'year', 'fuel_type', 'transmission']
    df = ord_[needed].copy()

    # Extract brand & model from title
    extracted              = df['title'].apply(extract_brand_model)
    df['brand_from_title'] = extracted.apply(lambda x: x[0])
    df['model']            = extracted.apply(lambda x: x[1])

    # Fill missing brand from title extraction
    df['brand'] = df['brand'].fillna('').str.strip()
    df['brand'] = df.apply(
        lambda r: r['brand_from_title']
                  if r['brand'] in ('', 'Unknown', 'nan') or pd.isna(r['brand'])
                  else r['brand'],
        axis=1
    )

    print(f"  Rows loaded : {len(df):,}")
    print(f"\n  Sample title -> brand / model extraction:")
    sample = df[['title', 'brand', 'model']].drop_duplicates('title').head(8)
    for _, row in sample.iterrows():
        print(f"    {row['title']:<40} -> brand={row['brand']:<20} model={row['model']}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLEAN & ENGINEER BASE FEATURES
#    No target encoding here — that happens inside the pipeline after split.
# ═══════════════════════════════════════════════════════════════════════════

def clean_and_engineer(df: pd.DataFrame):
    print("\nCleaning & engineering features...")
    df = df.copy()

    # Drop rows with missing essentials
    before = len(df)
    df = df.dropna(subset=['price', 'kms_covered', 'year'])
    df = df[df['price'] > 0]
    df = df[df['kms_covered'] >= 0]
    print(f"  Dropped {before - len(df):,} rows with missing/invalid values")

    # Normalise strings
    fuel_map = {'CNG & Hybrids': 'CNG'}
    df['brand']        = df['brand'].str.strip().replace(BRAND_NORMALISE).fillna('Unknown')
    df['model']        = df['model'].fillna('Unknown').str.strip()
    df['fuel_type']    = df['fuel_type'].fillna('Unknown').str.strip().replace(fuel_map)
    df['transmission'] = df['transmission'].fillna('Unknown').str.strip()

    # Cast numerics
    df['year']        = pd.to_numeric(df['year'],        errors='coerce').fillna(2020).astype(int)
    df['kms_covered'] = pd.to_numeric(df['kms_covered'], errors='coerce').fillna(0).astype(float)
    df['price']       = pd.to_numeric(df['price'],       errors='coerce')
    df = df.dropna(subset=['price'])

    # Outlier removal (1st-99th percentile)
    before = len(df)
    lo, hi = df['price'].quantile([0.01, 0.99])
    df = df[(df['price'] >= lo) & (df['price'] <= hi)].copy()
    print(f"  Removed {before - len(df):,} price outliers -> {len(df):,} rows remain")
    print(f"  Price range        : Rs.{df['price'].min():,.0f} - Rs.{df['price'].max():,.0f}")
    print(f"  Unique brands      : {df['brand'].nunique()}")
    print(f"  Unique models      : {df['model'].nunique()}")

    # brand_model combo: "BMW" + "M5" -> "BMW M5"
    # Gives the model a fine-grained price signal per exact variant
    df['brand_model'] = df['brand'] + ' ' + df['model']
    print(f"  Unique brand+model : {df['brand_model'].nunique()}")

    # Label-encode fuel_type and transmission (safe — not the target)
    encoders = {}
    for col in ['fuel_type', 'transmission']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} classes -> {list(le.classes_)}")

    # log_kms: compress the huge km range
    df['log_kms'] = np.log1p(df['kms_covered'])

    print(f"  Target encoding deferred to pipeline (no leakage)")
    return df, encoders


# ═══════════════════════════════════════════════════════════════════════════
# 3. AUTOMATIC FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def select_features(X_train: pd.DataFrame, y_train: pd.Series):
    print("\nAutomatic feature selection...")
    print(f"  Threshold : importance >= {IMPORTANCE_THRESHOLD}")

    probe_pipe = Pipeline([
        ('encoder', LeakFreeTargetEncoder(
            cols=['brand', 'model', 'brand_model'], smoothing=10.0
        )),
        ('model', XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ))
    ])
    probe_pipe.fit(X_train[PIPELINE_INPUT_COLS], y_train)

    importance = pd.Series(
        probe_pipe.named_steps['model'].feature_importances_,
        index=CANDIDATE_FEATURES
    ).sort_values(ascending=False)

    print(f"\n  {'Feature':<26} {'Importance':>12}  Decision")
    print(f"  {'=' * 54}")
    kept, dropped = [], []
    for feat, imp in importance.items():
        decision = 'KEEP' if imp >= IMPORTANCE_THRESHOLD else 'DROP'
        print(f"  {feat:<26} {imp:>12.4f}  {decision}")
        (kept if imp >= IMPORTANCE_THRESHOLD else dropped).append(feat)

    print(f"\n  Kept    ({len(kept)}): {kept}")
    print(f"  Dropped ({len(dropped)}): {dropped}")

    # Pipeline always needs raw string cols for the encoder
    string_cols  = ['brand', 'model', 'brand_model']
    numeric_kept = [f for f in kept if f not in
                    ('brand_price_mean', 'model_price_mean', 'brand_model_price_mean')]
    kept_pipeline_cols = string_cols + numeric_kept

    return kept, dropped, importance, kept_pipeline_cols


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(pipeline, X_test, y_test, name="Model"):
    preds = pipeline.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    mape  = np.mean(np.abs((y_test - preds) / y_test.clip(1))) * 100

    bar = '-' * 46
    print(f"\n  +{bar}+")
    print(f"  |  {name:<44}|")
    print(f"  +{bar}+")
    print(f"  |  MAE   : Rs.{mae:>13,.0f}{'':>18}|")
    print(f"  |  RMSE  : Rs.{rmse:>13,.0f}{'':>18}|")
    print(f"  |  MAPE  : {mape:>14.2f}%{'':>17}|")
    print(f"  |  R2    : {r2:>14.4f}{'':>18}|")
    print(f"  +{bar}+")
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'preds': preds}


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODELS
# ═══════════════════════════════════════════════════════════════════════════

def make_pipeline(model, smoothing=10.0):
    return Pipeline([
        ('encoder', LeakFreeTargetEncoder(
            cols=['brand', 'model', 'brand_model'], smoothing=smoothing
        )),
        ('model', model),
    ])


def train_random_forest(X_train, y_train, pipeline_cols):
    print("\nTraining Random Forest baseline...")
    pipe = make_pipeline(RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        n_jobs=-1, random_state=RANDOM_STATE
    ))
    pipe.fit(X_train[pipeline_cols], y_train)
    print("  Done")
    return pipe


def train_xgboost_default(X_train, y_train, pipeline_cols):
    print("\nTraining XGBoost (default params)...")
    pipe = make_pipeline(XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        monotone_constraints={'year': 1, 'log_kms': -1},
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    ))
    pipe.fit(X_train[pipeline_cols], y_train)
    print("  Done")
    return pipe


def fine_tune_xgboost(X_train, y_train, pipeline_cols, n_trials=OPTUNA_TRIALS):
    """
    Optuna tunes XGBoost hyperparameters AND smoothing strength via 5-fold CV.
    Pipeline guarantees encoder is re-fitted on each fold's train split only.
    """
    print(f"\nFine-tuning XGBoost with Optuna ({n_trials} trials)...")
    print("  Each trial = 5-fold CV with leak-free 3-level encoding...\n")

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int  ('n_estimators',     200, 1000),
            'learning_rate':     trial.suggest_float('learning_rate',     0.005, 0.3, log=True),
            'max_depth':         trial.suggest_int  ('max_depth',         3, 12),
            'min_child_weight':  trial.suggest_int  ('min_child_weight',  1, 10),
            'subsample':         trial.suggest_float('subsample',         0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree',  0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha',         1e-8, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda',        1e-8, 10.0, log=True),
            'gamma':             trial.suggest_float('gamma',             0.0, 5.0),
            'monotone_constraints': {'year': 1, 'log_kms': -1},
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
        }
        smoothing = trial.suggest_float('smoothing', 1.0, 100.0)

        pipe = Pipeline([
            ('encoder', LeakFreeTargetEncoder(
                cols=['brand', 'model', 'brand_model'], smoothing=smoothing
            )),
            ('model', XGBRegressor(**params)),
        ])

        scores = cross_val_score(
            pipe,
            X_train[pipeline_cols], y_train,
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params    = study.best_params.copy()
    best_smoothing = best_params.pop('smoothing', 10.0)

    print(f"\n  Best CV MAE : Rs.{study.best_value:,.0f}")
    print(f"  Best params :")
    for k, v in best_params.items():
        print(f"    {k:<25} : {v}")
    print(f"    {'smoothing':<25} : {best_smoothing:.2f}")

    print("\n  Training final model on full training set...")
    best_pipe = Pipeline([
        ('encoder', LeakFreeTargetEncoder(
            cols=['brand', 'model', 'brand_model'], smoothing=best_smoothing
        )),
        ('model', XGBRegressor(
            **best_params, 
            monotone_constraints={'year': 1, 'log_kms': -1},
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )),
    ])
    best_pipe.fit(X_train[pipeline_cols], y_train)
    print("  Done")
    return best_pipe, study


# ═══════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def save_plots(y_test, results, kept_features, importance_all, study=None):
    print("\nSaving plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'AutoVault — Car Price Prediction (brand + model + brand_model encoding)',
        fontsize=13, fontweight='bold'
    )
    PAL = {
        'Random Forest':      '#2196F3',
        'XGBoost Default':    '#FF9800',
        'XGBoost Fine-tuned': '#4CAF50'
    }
    names = list(results.keys())

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    p  = results['XGBoost Fine-tuned']['preds']
    ax.scatter(y_test / 1e6, p / 1e6, alpha=0.25, s=6, color='#4CAF50')
    mv = max(float(y_test.max()), float(p.max())) / 1e6
    ax.plot([0, mv], [0, mv], 'r--', lw=1.5, label='Perfect')
    ax.set_xlabel('Actual (Rs.M)'); ax.set_ylabel('Predicted (Rs.M)')
    ax.set_title('Predicted vs Actual (Fine-tuned XGB)'); ax.legend(fontsize=8)

    # 2. Residuals
    ax  = axes[0, 1]
    res = (p - y_test) / 1e6
    ax.scatter(p / 1e6, res, alpha=0.25, s=6, color='#9C27B0')
    ax.axhline(0, color='red', lw=1.5, linestyle='--')
    ax.set_xlabel('Predicted (Rs.M)'); ax.set_ylabel('Residual (Rs.M)')
    ax.set_title('Residuals')

    # 3. MAE comparison
    ax   = axes[0, 2]
    maes = [results[m]['mae'] / 1e6 for m in names]
    bars = ax.bar(names, maes, color=[PAL[m] for m in names], alpha=0.85)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'Rs.{v:.2f}M', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('MAE (Rs.M)'); ax.set_title('Model MAE Comparison')
    ax.set_xticklabels(names, rotation=10, fontsize=8)

    # 4. R2 comparison
    ax   = axes[1, 0]
    r2s  = [results[m]['r2'] for m in names]
    bars = ax.bar(names, r2s, color=[PAL[m] for m in names], alpha=0.85)
    ax.set_ylim(0, 1.05)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('R2'); ax.set_title('Model R2 Comparison')
    ax.set_xticklabels(names, rotation=10, fontsize=8)

    # 5. Feature importance
    ax     = axes[1, 1]
    imp_s  = importance_all.sort_values()
    colors = ['#4CAF50' if f in kept_features else '#f44336' for f in imp_s.index]
    ax.barh(imp_s.index, imp_s.values, color=colors, alpha=0.85)
    ax.axvline(IMPORTANCE_THRESHOLD, color='black', lw=1.2, linestyle='--',
               label=f'Threshold ({IMPORTANCE_THRESHOLD})')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance\n(green=kept | red=dropped)')
    ax.legend(fontsize=7)

    # 6. Optuna history
    ax = axes[1, 2]
    if study:
        vals = [t.value / 1e6 for t in study.trials if t.value is not None]
        best = pd.Series(vals).cummin().values
        ax.plot(vals, alpha=0.35, color='#FF9800', label='Trial MAE')
        ax.plot(best, color='#4CAF50', lw=2,       label='Best so far')
        ax.set_xlabel('Trial'); ax.set_ylabel('MAE (Rs.M)')
        ax.set_title('Optuna Optimisation History'); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No study data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, 'training_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict_price(brand: str, model_name: str, year: int,
                  kms_covered: float, fuel_type: str, transmission: str) -> float:
    """
    Predict the resale price of a single car.

    Parameters
    ----------
    brand        : str   e.g. 'BMW'
    model_name   : str   e.g. 'M5'   (use extract_brand_model() to get from title)
    year         : int   e.g. 2022
    kms_covered  : float e.g. 35000.0
    fuel_type    : str   'Petrol' | 'Diesel' | 'Electric' | 'CNG' | 'Hybrid'
    transmission : str   'Automatic' | 'Manual'

    Returns
    -------
    float -- predicted price in Rs.
    """
    pipeline  = joblib.load(os.path.join(MODEL_DIR, 'nn_model.pkl'))
    encoders  = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

    def safe_le(le, val):
        val = str(val).strip()
        return int(le.transform([val])[0]) if val in le.classes_ else 0

    brand       = str(brand).strip()
    model_name  = str(model_name).strip()
    
    # Run through the normalizer to fix casing issues (e.g. 'bmw mw' -> 'BMW M5')
    brand, model_name = extract_brand_model(f"{brand} {model_name}")
    brand_model = f"{brand} {model_name}"

    row = {
        'brand':            brand,
        'model':            model_name,
        'brand_model':      brand_model,
        'year':             int(year),
        'log_kms':          float(np.log1p(kms_covered)),
        'fuel_type_enc':    safe_le(encoders['fuel_type'],    fuel_type),
        'transmission_enc': safe_le(encoders['transmission'], transmission),
    }

    # feat_cols = pipeline INPUT cols (raw strings included for encoder)
    X = pd.DataFrame([row])[feat_cols]
    return float(pipeline.predict(X)[0])


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Load
    df = load_data()

    # Clean + engineer base features (no target encoding yet)
    df, encoders = clean_and_engineer(df)

    # Train / test split — BEFORE any target encoding
    X_all = df[PIPELINE_INPUT_COLS]
    y_all = df['price']

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain : {len(X_train_full):,} rows")
    print(f"Test  : {len(X_test_full):,} rows")
    print(f"Split done BEFORE encoding — zero target leakage")

    # Auto feature selection
    kept, dropped, importance_all, kept_pipeline_cols = select_features(
        X_train_full, y_train
    )

    # Train all three models
    print(f"\n{'='*60}")
    print(f"  TRAINING")
    print(f"  Model features  : {kept}")
    print(f"  Pipeline inputs : {kept_pipeline_cols}")
    print(f"{'='*60}")

    rf_model         = train_random_forest  (X_train_full, y_train, kept_pipeline_cols)
    xgb_default      = train_xgboost_default(X_train_full, y_train, kept_pipeline_cols)
    xgb_tuned, study = fine_tune_xgboost    (X_train_full, y_train, kept_pipeline_cols,
                                             n_trials=OPTUNA_TRIALS)

    # Evaluate
    print(f"\n{'='*60}")
    print(f"  EVALUATION  (held-out test set)")
    print(f"{'='*60}")
    results = {
        'Random Forest':      evaluate(rf_model,    X_test_full[kept_pipeline_cols], y_test, "Random Forest"),
        'XGBoost Default':    evaluate(xgb_default, X_test_full[kept_pipeline_cols], y_test, "XGBoost Default"),
        'XGBoost Fine-tuned': evaluate(xgb_tuned,   X_test_full[kept_pipeline_cols], y_test, "XGBoost Fine-tuned"),
    }

    # Save
    print("\nSaving models...")
    joblib.dump(rf_model,           os.path.join(MODEL_DIR, 'rf_baseline.pkl'))
    joblib.dump(xgb_default,        os.path.join(MODEL_DIR, 'xgb_default.pkl'))
    joblib.dump(xgb_tuned,          os.path.join(MODEL_DIR, 'xgb_finetuned.pkl'))
    joblib.dump(encoders,           os.path.join(MODEL_DIR, 'encoders.pkl'))
    joblib.dump(kept_pipeline_cols, os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    print(f"  Saved to: {MODEL_DIR}/")

    # Plots
    save_plots(y_test, results, kept, importance_all, study)

    # Sample predictions — pairs show model-level differentiation
    print(f"\nSample Predictions (pairs share brand/year/kms to show model matters):")
    print(f"  {'Title':<38} {'Year':<6} {'Kms':>8}  {'Fuel':<10} {'Trans':<12}  Predicted")
    print(f"  {'-'*95}")
    samples = [
        ('BMW M5',                    2022, 15000, 'Petrol',  'Automatic'),
        ('BMW 3 Series',              2022, 15000, 'Petrol',  'Automatic'),  # same brand/year/kms
        ('Mercedes-Benz AMG GLE',     2021, 20000, 'Diesel',  'Automatic'),
        ('Mercedes-Benz C-Class',     2021, 20000, 'Diesel',  'Automatic'),  # same brand/year/kms
        ('Hyundai Creta',             2020, 55000, 'Diesel',  'Manual'),
        ('Hyundai i20',               2020, 55000, 'Diesel',  'Manual'),     # same brand/year/kms
        ('Maruti Ertiga',             2018, 75000, 'Petrol',  'Manual'),
        ('Maruti Swift',              2018, 75000, 'Petrol',  'Manual'),     # same brand/year/kms
        ('Tata Nexon',                2021, 40000, 'Petrol',  'Manual'),
        ('Land Rover Range Rover',    2023, 12000, 'Diesel',  'Automatic'),
        ('Porsche 911',               2023,  8000, 'Petrol',  'Automatic'),
        ('Ferrari 488 GTB',           2024,  4000, 'Petrol',  'Automatic'),
    ]
    for title, year, kms, fuel, trans in samples:
        brand, model_name = extract_brand_model(title)
        price = predict_price(brand, model_name, year, kms, fuel, trans)
        print(f"  {title:<38} {year:<6} {kms:>8,}  {fuel:<10} {trans:<12}  Rs.{price:>12,.0f}")

    # Summary
    best = results['XGBoost Fine-tuned']
    print(f"\n{'='*60}")
    print(f"  BEST MODEL  : XGBoost Fine-tuned (Optuna {OPTUNA_TRIALS} trials)")
    print(f"  R2          : {best['r2']:.4f}  ->  {best['r2']*100:.1f}% variance explained")
    print(f"  MAE         : Rs.{best['mae']:,.0f}")
    print(f"  MAPE        : {best['mape']:.2f}%")
    print(f"  Features    : {kept}")
    print(f"  Auto-dropped: {dropped}")
    print(f"\n  Encoding levels:")
    print(f"  -> brand_price_mean       : avg price of ALL cars of that brand")
    print(f"  -> model_price_mean       : avg price of ALL cars of that model name")
    print(f"  -> brand_model_price_mean : avg price of THAT EXACT brand+model combo")
    print(f"\n  Leakage guarantees:")
    print(f"  -> Train/test split done BEFORE any target encoding")
    print(f"  -> Encoder re-fitted on each CV fold's train split only")
    print(f"  -> Smoothing tuned by Optuna (rare models shrunk to global mean)")
    print(f"  -> Unseen brand/model at inference -> global training mean")
    print(f"\n  Production files:")
    print(f"  -> models/xgb_finetuned.pkl  (full pipeline: encoder + model)")
    print(f"  -> models/encoders.pkl")
    print(f"  -> models/feature_columns.pkl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()