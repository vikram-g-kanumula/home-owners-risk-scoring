# ==============================================================================
# residual_model.py
# GA2M (EBM) residual layer on top of baseline GLM
#
# Key design choices vs v1:
#   1. LOG-SCALE TARGET: learns log(True/GLM), i.e. a multiplicative uplift
#      → Final Premium = GLM_Premium × exp(EBM_log_prediction)
#      → guarantees final premium > 0 by construction
#   2. BOUNDED CORRIDOR: uplift clamped to [0.65×, 1.60×] GLM estimate
#      → max downward adjustment: -35%  |  max upward: +60%
#   3. SMOOTHER EBM: lower learning rate, more bags → stable shape functions
#   4. BACKWARD-COMPATIBLE OUTPUT: EBM_Residual_Pred stored as dollar adj
#      (Final - GLM) so app.py needs no changes
# ==============================================================================

import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os, warnings
warnings.filterwarnings("ignore")

# Uplift corridor — adjust to taste
MIN_UPLIFT = 0.65    # GLM can be reduced by at most 35%
MAX_UPLIFT = 1.60    # GLM can be increased by at most 60%
LOG_MIN    = np.log(MIN_UPLIFT)
LOG_MAX    = np.log(MAX_UPLIFT)

ALL_25_FEATURES = [
    "Year_Built", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
    "Construction_Type", "Protection_Class", "AOI", "Deductible",
    "Territory", "Roof_Age_Applicant", "Fire_Alarm", "Burglar_Alarm",
    "Roof_Vulnerability_Satellite", "Wildfire_Exposure_Daily",
    "Water_Loss_Recency_Months", "RCV_Appraised", "Fire_Hydrant_Distance",
    "Tree_Canopy_Density", "Crime_Severity_Index", "Pluvial_Flood_Depth",
    "Building_Code_Compliance", "Slope_Steepness", "Attic_Ventilation",
    "Hail_Frequency", "Soil_Liquefaction_Risk",
]
CAT_COLS = [
    "Construction_Type", "Territory", "Deductible",
    "Fire_Alarm", "Burglar_Alarm", "Attic_Ventilation", "Soil_Liquefaction_Risk",
]


def train_residual_ebm(data_path: str = "data/synthetic_homeowners_data_with_baseline.csv"):
    print("Loading data with baseline GLM predictions...")
    df = pd.read_csv(data_path)

    for col in CAT_COLS:
        df[col] = df[col].astype(str)

    X = df[ALL_25_FEATURES].copy()

    # ── Log-scale residual target ──────────────────────────────────────────
    eps          = 1e-6
    log_true     = np.log(df["Expected_Pure_Premium"] + eps)
    log_glm      = np.log(df["GLM_Pure_Premium"] + eps)
    y_log_resid  = log_true - log_glm   # log uplift factor
    print(f"  Log-residual: mean={y_log_resid.mean():.4f}  "
          f"std={y_log_resid.std():.4f}  "
          f"p5={np.percentile(y_log_resid,5):.3f}  "
          f"p95={np.percentile(y_log_resid,95):.3f}")

    # ── Fit EBM on log residuals ───────────────────────────────────────────
    print(f"\nTraining GA2M (EBM) on log residuals using {len(ALL_25_FEATURES)} features...")
    ebm = ExplainableBoostingRegressor(
        interactions=10,
        max_bins=256,
        max_interaction_bins=32,
        learning_rate=0.02,     # smoother than default
        outer_bags=8,
        inner_bags=0,
        random_state=42,
    )
    ebm.fit(X, y_log_resid)
    print("EBM training complete.")

    # ── Predictions with bounded corridor ─────────────────────────────────
    raw_log_pred = ebm.predict(X)
    clipped_log  = np.clip(raw_log_pred, LOG_MIN, LOG_MAX)
    uplift_factor = np.exp(clipped_log)

    df["EBM_Log_Uplift"]     = clipped_log
    df["EBM_Uplift_Factor"]  = uplift_factor
    df["Final_Pure_Premium"] = df["GLM_Pure_Premium"] * uplift_factor

    # Dollar adjustment — kept for app.py display compatibility
    df["EBM_Residual_Pred"]  = df["Final_Pure_Premium"] - df["GLM_Pure_Premium"]

    print(f"\n  Uplift factor range : "
          f"{uplift_factor.min():.3f}× – {uplift_factor.max():.3f}×")
    print(f"  Mean uplift         : {uplift_factor.mean():.4f}×")
    print(f"  Mean Final Premium  : ${df['Final_Pure_Premium'].mean():,.2f}")
    print(f"  Min Final Premium   : ${df['Final_Pure_Premium'].min():,.2f}")

    # ── Save outputs ──────────────────────────────────────────────────────
    out_path = "data/final_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to '{out_path}'.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(ebm, "models/ebm_residual_model.pkl")
    print("Saved EBM model to 'models/ebm_residual_model.pkl'.")

    # ── Performance summary ───────────────────────────────────────────────
    true_pp  = df["Expected_Pure_Premium"]
    glm_r2   = r2_score(true_pp, df["GLM_Pure_Premium"])
    final_r2 = r2_score(true_pp, df["Final_Pure_Premium"])
    glm_rmse = np.sqrt(mean_squared_error(true_pp, df["GLM_Pure_Premium"]))
    fin_rmse = np.sqrt(mean_squared_error(true_pp, df["Final_Pure_Premium"]))

    print("\n" + "=" * 54)
    print("  MODEL PERFORMANCE vs TRUE EXPECTED PREMIUM")
    print("=" * 54)
    print(f"  Legacy GLM  (14 features, linear)   R²: {glm_r2:.4f}   RMSE: ${glm_rmse:,.0f}")
    print(f"  GLM + GA2M  (25 features, glass-box) R²: {final_r2:.4f}   RMSE: ${fin_rmse:,.0f}")
    print(f"  Incremental Variance Captured  ΔR²:  +{final_r2 - glm_r2:.4f}")
    print("=" * 54)


if __name__ == "__main__":
    train_residual_ebm()
