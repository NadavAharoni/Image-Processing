import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def main():
    # ── 1. Load ──────────────────────────────────────────────────────────────────
    # fetch_openml returns the Cleveland Heart Disease dataset (14 features)
    raw = fetch_openml(name='heart-c', version=1, as_frame=True, parser='auto')
    df = raw.frame.copy()

    print("Available features:")
    for i, col in enumerate(df.columns):
        print(f"  {i:2d}: {col}")

    print("\nShape before cleaning:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # ── 2. Clean ─────────────────────────────────────────────────────────────────
    # The dataset has 6 missing values (in 'ca' and 'thal').
    # Strategy: drop the affected rows — acceptable here because it's only 6/303 (~2%).
    df_clean = df.dropna()
    print(f"\nShape after dropping rows with missing values: {df_clean.shape}")
    print(f"Rows removed: {len(df) - len(df_clean)}")

    # ── 3. Prepare features and labels ───────────────────────────────────────────
    # Target: 0 = no heart disease, 1-4 = heart disease present → binarise to 0/1
    df_clean = df_clean.copy()
    df_clean['target'] = (df_clean['num'] == '>50_1').astype(int)

    # Pick two features for 2D visualisation
    # 'age'    — age in years
    # 'thalach' — maximum heart rate achieved during exercise test
    x_vals = df_clean['age'].astype(float).values
    y_vals = df_clean['thalach'].astype(float).values
    labels = df_clean['target'].values

    # Split by class
    no_disease = labels == 0
    has_disease = labels == 1

    print(f"\nClass distribution after cleaning:")
    print(f"  No heart disease : {no_disease.sum()}")
    print(f"  Heart disease    : {has_disease.sum()}")

    # ── 4. Plot ───────────────────────────────────────────────────────────────────

    # --- Exercise: fill in this section ---

    # --- End of exercise section ---

    plt.xlabel('Age (years)')
    plt.ylabel('Max heart rate achieved (thalach)')
    plt.title('Cleveland Heart Disease — Age vs. Max Heart Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
