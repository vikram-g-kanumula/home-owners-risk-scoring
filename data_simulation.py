# ==============================================================================
# data_simulation.py
# Realistic synthetic homeowners data
# Design targets:
#   • Average pure premium ~$1,200–$1,500 (AOI-anchored, typical US HO market)
#   • GLM (12 legacy vars + 2 actuary interactions) → R² ~ 0.55–0.68
#   • GA2M residual layer → incremental ΔR² ~ 0.07–0.12
#   • All premiums floored at $300; no policy < $300 by construction
# ==============================================================================

import pandas as pd
import numpy as np
import os


def generate_homeowners_data(n_samples: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    print(f"Generating {n_samples:,} synthetic homeowners policies...")

    # ── 12 Legacy Features ─────────────────────────────────────────────────
    Year_Built         = np.random.randint(1950, 2023, n_samples)
    Square_Footage     = np.random.normal(2200, 600, n_samples).clip(800, 5000)
    CLUE_Loss_Count    = np.random.poisson(0.3, n_samples)
    Credit_Score       = np.random.normal(700, 80, n_samples).clip(300, 850)
    Construction_Type  = np.random.choice(
        ["Frame", "Masonry", "Fire Resistive"], n_samples, p=[0.70, 0.20, 0.10])
    Protection_Class   = np.random.randint(1, 11, n_samples)       # 1=best, 10=worst
    AOI                = Square_Footage * np.random.uniform(150, 260, n_samples)  # Amount of Insurance
    Deductible         = np.random.choice([500, 1000, 2000, 5000], n_samples,
                                          p=[0.15, 0.50, 0.25, 0.10])
    Territory          = np.random.choice(
        ["Urban", "Suburban", "Rural"], n_samples, p=[0.30, 0.50, 0.20])
    Roof_Age_Applicant = np.random.randint(1, 31, n_samples)
    Fire_Alarm         = np.random.binomial(1, 0.40, n_samples)
    Burglar_Alarm      = np.random.binomial(1, 0.30, n_samples)

    # ── 13 Modern / Advanced Features ──────────────────────────────────────
    Roof_Vulnerability_Satellite = (
        Roof_Age_Applicant + np.random.normal(1, 4, n_samples)).clip(0, 38)
    Wildfire_Exposure_Daily      = np.random.beta(0.5, 2.0, n_samples) * 100   # 0–100
    Water_Loss_Recency_Months    = np.where(
        CLUE_Loss_Count > 0, np.random.randint(1, 36, n_samples), 120)
    RCV_Appraised                = Square_Footage * np.random.uniform(140, 220, n_samples)
    Fire_Hydrant_Distance        = np.random.lognormal(-0.5, 0.8, n_samples).clip(0.05, 10)
    Tree_Canopy_Density          = np.random.beta(2, 5, n_samples) * 100        # 0–100
    Crime_Severity_Index         = np.random.normal(50, 20, n_samples).clip(0, 100)
    Pluvial_Flood_Depth          = np.random.exponential(5, n_samples).clip(0, 36)
    Building_Code_Compliance     = np.where(
        Year_Built > 2000,
        np.random.randint(75, 100, n_samples),
        np.random.randint(35, 78, n_samples))
    Slope_Steepness              = np.random.exponential(10, n_samples).clip(0, 45)
    Attic_Ventilation            = np.random.choice(
        ["Poor", "Adequate", "Excellent"], n_samples, p=[0.30, 0.50, 0.20])
    Hail_Frequency               = np.random.poisson(1.5, n_samples)
    Soil_Liquefaction_Risk       = np.random.choice(
        ["Low", "Moderate", "High"], n_samples, p=[0.70, 0.20, 0.10])

    # ── Exposure base (AOI in $k, typical HO base rate ~$2.5 per $1k) ──────
    exposure = AOI / 1_000.0
    base_log = np.log(2.5) + np.log(exposure)   # anchors mean ~$1,100–$1,400

    # ──────────────────────────────────────────────────────────────────────
    # LEGACY SIGNAL  (linear / log-linear, designed for GLM to capture well)
    # Includes 2 "actuary-grade" interactions that a good GLM would already add
    # ──────────────────────────────────────────────────────────────────────
    legacy_log_signal = (
        # main effects — directionally validated against ISO relativities
          (Protection_Class - 5) * 0.06           # higher PC → higher risk
        + CLUE_Loss_Count * 0.22                   # prior claims surcharge
        - (Credit_Score - 700) / 1000 * 1.50       # credit inverse risk
        + (2023 - Year_Built) * 0.004              # building age
        + (Construction_Type == "Frame") * 0.20    # frame vs masonry
        + (Deductible == 500) * 0.08               # low deductible uptick
        - (Deductible == 5000) * 0.10              # high deductible credit
        + (Territory == "Urban") * 0.12            # urban surcharge
        - Fire_Alarm * 0.09                        # fire alarm credit
        - Burglar_Alarm * 0.07                     # burglar alarm credit
        + Roof_Age_Applicant * 0.006               # continuous roof age
        # Actuary-grade interactions (GLM would engineer these explicitly)
        + ((Territory == "Urban") & (Protection_Class > 6)) * 0.12
        + ((Roof_Age_Applicant > 20) & (Hail_Frequency >= 3)) * 0.10
    )
    # Standardize and scale to target component variance
    legacy_log_signal = (legacy_log_signal - legacy_log_signal.mean()) / legacy_log_signal.std()
    legacy_log_signal = legacy_log_signal * 0.36   # controls legacy R² component

    # ──────────────────────────────────────────────────────────────────────
    # MODERN SIGNAL  (non-linear effects + pairwise interactions the GLM misses)
    # These are the patterns the GA2M layer is designed to recover
    # ──────────────────────────────────────────────────────────────────────
    rcv_overstatement = np.maximum(0, AOI - RCV_Appraised) / 100_000

    modern_log_signal = (
        # Non-linear main effects
          (Roof_Vulnerability_Satellite / 20) ** 2 * 0.18       # accelerating roof decay
        - np.log1p(Fire_Hydrant_Distance) * 0.10                # diminishing credit w/ distance
        + (Building_Code_Compliance < 60) * 0.12                # threshold non-linearity
        # Pairwise interactions
        + (Wildfire_Exposure_Daily / 100) * (Roof_Vulnerability_Satellite / 20) * 0.28
        + np.exp(-Water_Loss_Recency_Months / 12) * (Tree_Canopy_Density / 100) * 0.22
        + rcv_overstatement * (Crime_Severity_Index / 100) * 0.18
        + (Pluvial_Flood_Depth > 15) * (2023 - Year_Built > 30) * 0.18
        + (Slope_Steepness / 45) * (Wildfire_Exposure_Daily / 100) * 0.14
        + (Hail_Frequency > 3) * (Roof_Vulnerability_Satellite > 18) * 0.18
    )
    modern_log_signal = (modern_log_signal - modern_log_signal.mean()) / modern_log_signal.std()
    modern_log_signal = modern_log_signal * 0.10   # smaller incremental component

    # ──────────────────────────────────────────────────────────────────────
    # TARGET PREMIUM ASSEMBLY
    # Noise calibrated so max theoretical R² ≈ 0.74
    #   Var(signal) / Var(signal+noise) = (0.36²+0.10²) / (0.36²+0.10²+0.21²)
    #                                   = 0.1396 / 0.1837 ≈ 0.760
    # ──────────────────────────────────────────────────────────────────────
    total_signal = legacy_log_signal + modern_log_signal
    noise        = np.random.normal(0, 0.21, n_samples)

    true_log_premium  = base_log + total_signal + noise
    expected_pure_premium = np.exp(true_log_premium)
    # Hard floor: no HO policy should be priced below $300
    expected_pure_premium = np.clip(expected_pure_premium, 300, None)

    # ── Simulate claims (frequency/severity) ───────────────────────────────
    lambda_freq  = np.clip(expected_pure_premium / 6_000, 0.01, 0.40)
    claim_count  = np.random.poisson(lambda_freq)
    claim_amount = np.zeros(n_samples)
    has_claim    = claim_count > 0
    sev_mean     = expected_pure_premium[has_claim] / lambda_freq[has_claim]
    claim_amount[has_claim] = np.random.gamma(
        shape=2.0, scale=sev_mean / 2.0)

    data = pd.DataFrame({
        # Legacy
        "Year_Built":              Year_Built,
        "Square_Footage":          Square_Footage.round(0),
        "CLUE_Loss_Count":         CLUE_Loss_Count,
        "Credit_Score":            Credit_Score.round(0),
        "Construction_Type":       Construction_Type,
        "Protection_Class":        Protection_Class,
        "AOI":                     AOI.round(0),
        "Deductible":              Deductible,
        "Territory":               Territory,
        "Roof_Age_Applicant":      Roof_Age_Applicant,
        "Fire_Alarm":              Fire_Alarm.astype(bool),
        "Burglar_Alarm":           Burglar_Alarm.astype(bool),
        # Modern
        "Roof_Vulnerability_Satellite": Roof_Vulnerability_Satellite.round(2),
        "Wildfire_Exposure_Daily":      Wildfire_Exposure_Daily.round(2),
        "Water_Loss_Recency_Months":    Water_Loss_Recency_Months,
        "RCV_Appraised":                RCV_Appraised.round(0),
        "Fire_Hydrant_Distance":        Fire_Hydrant_Distance.round(3),
        "Tree_Canopy_Density":          Tree_Canopy_Density.round(2),
        "Crime_Severity_Index":         Crime_Severity_Index.round(2),
        "Pluvial_Flood_Depth":          Pluvial_Flood_Depth.round(2),
        "Building_Code_Compliance":     Building_Code_Compliance,
        "Slope_Steepness":              Slope_Steepness.round(2),
        "Attic_Ventilation":            Attic_Ventilation,
        "Hail_Frequency":               Hail_Frequency,
        "Soil_Liquefaction_Risk":       Soil_Liquefaction_Risk,
        # Targets
        "Expected_Pure_Premium": expected_pure_premium.round(2),
        "Claim_Count":           claim_count,
        "Claim_Amount":          claim_amount.round(2),
    })

    return data


if __name__ == "__main__":
    df = generate_homeowners_data(50_000)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_homeowners_data.csv", index=False)
    print("\n── DATA GENERATION COMPLETE ──")
    print(f"  Policies generated  : {len(df):,}")
    print(f"  Mean Pure Premium   : ${df['Expected_Pure_Premium'].mean():,.2f}")
    print(f"  Median Pure Premium : ${df['Expected_Pure_Premium'].median():,.2f}")
    print(f"  Min / Max           : ${df['Expected_Pure_Premium'].min():,.0f} / "
          f"${df['Expected_Pure_Premium'].max():,.0f}")
    print(f"  Claim frequency     : {(df['Claim_Count'] > 0).mean():.2%}")
