# ==============================================================================
# baseline_glm.py
# Legacy 12-variable GLM with 2 actuary-grade engineered interactions
# Design targets:
#   • Poisson (frequency) × Gamma (severity) pure premium GLM
#   • R² vs Expected_Pure_Premium: ~ 0.58–0.68
#   • Provides GLM_Pure_Premium and GLM_Log_Offset for residual model
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os


def run_baseline_glm(data_path: str = "data/synthetic_homeowners_data.csv"):
    print("Loading synthetic data...")
    df = pd.read_csv(data_path)

    # ── Feature set: 12 legacy + 2 engineered actuary interactions ─────────
    df["Urban_HighPC"]       = ((df["Territory"] == "Urban") &
                                (df["Protection_Class"] > 6)).astype(int)
    df["OldRoof_HighHail"]   = ((df["Roof_Age_Applicant"] > 20) &
                                (df["Hail_Frequency"] >= 3)).astype(int)

    legacy_features = [
        "Year_Built", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
        "Construction_Type", "Protection_Class", "AOI", "Deductible",
        "Territory", "Roof_Age_Applicant", "Fire_Alarm", "Burglar_Alarm",
        # Engineered actuary interactions
        "Urban_HighPC", "OldRoof_HighHail",
    ]

    cat_cols = ["Construction_Type", "Territory", "Deductible",
                "Fire_Alarm", "Burglar_Alarm",
                "Urban_HighPC", "OldRoof_HighHail"]
    num_cols = [c for c in legacy_features if c not in cat_cols]

    for col in cat_cols:
        df[col] = df[col].astype(str)

    X    = df[legacy_features].copy()
    y_freq = df["Claim_Count"]
    y_loss = df["Claim_Amount"]

    sev_mask = y_freq > 0
    X_sev    = X[sev_mask].copy()
    y_sev    = (y_loss[sev_mask] / y_freq[sev_mask])
    w_sev    = y_freq[sev_mask].values

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ])

    print("Training Frequency GLM (Poisson)...")
    freq_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    PoissonRegressor(alpha=1e-4, max_iter=2000, solver="lbfgs")),
    ])
    freq_model.fit(X, y_freq)

    print("Training Severity GLM (Gamma)...")
    sev_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    GammaRegressor(alpha=1e-4, max_iter=2000, solver="lbfgs")),
    ])
    sev_model.fit(X_sev, y_sev, regressor__sample_weight=w_sev)

    print("Generating GLM predictions...")
    pred_freq = freq_model.predict(X)
    pred_sev  = sev_model.predict(X)

    # Clip to sensible ranges before multiplying
    pred_freq = np.clip(pred_freq, 0.005, 0.50)
    pred_sev  = np.clip(pred_sev,  200,   80_000)

    df["GLM_Freq_Pred"]    = pred_freq
    df["GLM_Sev_Pred"]     = pred_sev
    df["GLM_Pure_Premium"] = np.clip(pred_freq * pred_sev, 200, None)  # floor at $200
    df["GLM_Log_Offset"]   = np.log(df["GLM_Pure_Premium"])

    # Save enriched dataset (with engineered interaction cols for residual model)
    out_path = "data/synthetic_homeowners_data_with_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved enriched dataset to '{out_path}'.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(freq_model, "models/legacy_freq_model.pkl")
    joblib.dump(sev_model,  "models/legacy_sev_model.pkl")
    print("Saved GLM models to 'models/'.")

    true_pp   = df["Expected_Pure_Premium"]
    glm_pp    = df["GLM_Pure_Premium"]
    r2_glm    = r2_score(true_pp, glm_pp)
    rmse_glm  = np.sqrt(mean_squared_error(true_pp, glm_pp))

    print("\n── BASELINE GLM EVALUATION ──")
    print(f"  R² vs Expected Pure Premium : {r2_glm:.4f}")
    print(f"  RMSE vs Expected Pure Premium: ${rmse_glm:,.2f}")
    print(f"  Mean GLM Premium             : ${glm_pp.mean():,.2f}")


if __name__ == "__main__":
    run_baseline_glm()
