#########################################################################
# not much interogation of these functions apart from simply using them #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve


def visualize_binary_profiles(X, y):
    print("\n=== Analyzing Binary Feature Profiles ===")

    # 1. Clean and combine data safely
    combined = pd.concat([X, y], axis=1).dropna()
    target_col = y.name if y.name else 'target'

    # 2. Extract and convert features to pure 0 and 1 flags instantly
    # get_dummies automatically handles text, categoricals, and booleans safely
    X_numeric = pd.get_dummies(combined[list(X.columns)], drop_first=True).astype(int)

    # 3. Clean target column just in case it contains string flags like 'Yes' or 'Positive'
    binary_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'True': 1, 'False': 0, True: 1, False: 0, '1': 1, '0': 0}
    y_numeric = combined[target_col].map(binary_map).fillna(combined[target_col]).astype(float)

    # Reassemble a clean, perfectly numeric workspace
    working_df = X_numeric.copy()
    working_df['TARGET_VAL'] = y_numeric

    # 4. Group by all binary features to discover patient profiles
    profile_stats = working_df.groupby(list(X_numeric.columns)).agg(
        Total_Count=('TARGET_VAL', 'count'),
        Positive_Rate=('TARGET_VAL', 'mean')
    ).reset_index()

    # 5. Sort and grab the top 20 most frequent profiles
    profile_stats = profile_stats.sort_values(by='Total_Count', ascending=False).head(20)

    if profile_stats.empty:
        print("No profiles found.")
        return

    profile_stats = profile_stats.reset_index(drop=True)
    features_matrix = profile_stats[list(X_numeric.columns)]

    # 6. Build the layout
    plt.figure(figsize=(14, 8))

    # Left side: Heatmap showing feature combinations
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(features_matrix, annot=True, cmap="Blues", cbar=False, fmt='g', linewidths=0.5)
    ax1.set_title("Top 20 Patient Profiles\n(1 = Feature Present / 'Yes')", fontsize=12, weight='bold')
    ax1.set_xlabel("Binary Features")
    ax1.set_ylabel("Profile Rank (Most Common at Top)")

    # Right side: Target distribution bar plot
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    sns.barplot(
        x=profile_stats['Positive_Rate'] * 100,
        y=profile_stats.index,
        orient='h',
        hue=profile_stats['Positive_Rate'],
        legend=False
    )

    # Dotted line shows baseline 1:17 risk profile
    ax2.axvline(x=5.5, color='gray', linestyle='--', label='Baseline Rate (1:17)')
    ax2.set_title("Positivity Rate (%)\nfor each Profile", fontsize=12, weight='bold')
    ax2.set_xlabel("Positive Rate (%)")
    ax2.set_ylabel("")
    ax2.tick_params(labelleft=False)

    plt.tight_layout()
    plt.show()


def get_optimal_threshold(model, X_val, y_val):
    """
    Finds the decision threshold that maximizes the F1-score
    using the precision-recall curve on validation data.
    """
    # 1. Get predicted probabilities for the positive class
    y_proba = model.predict_proba(X_val)[:, 1]

    # 2. Calculate precisions, recalls, and thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    # 3. Calculate F1-scores for each threshold point
    # Add a tiny epsilon (1e-10) to prevent division by zero errors
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # 4. Find the index of the highest F1-score
    best_idx = np.argmax(f1_scores)

    # precision_recall_curve returns one more value for precision/recall than thresholds,
    # so we handle the edge case where the best index lands on the last element.
    if best_idx >= len(thresholds):
        best_threshold = 0.5  # Fallback to default if out of bounds
    else:
        best_threshold = thresholds[best_idx]

    print(f"--- Threshold Tuning Optimization ---")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Expected Val Precision: {precisions[best_idx]:.4f}")
    print(f"Expected Val Recall: {recalls[best_idx]:.4f}")
    print(f"Expected Val F1-Score: {f1_scores[best_idx]:.4f}\n")

    return best_threshold


def print_top_positivity_profiles(X, y):
    print("\n" + "=" * 50)
    print("      PROFILES WITH HIGHEST CORONA POSITIVITY RATES      ")
    print("=" * 50)

    # 1. Standardize text features and target safely using dummy encoding
    combined = pd.concat([X, y], axis=1).dropna()
    target_col = y.name if y.name else 'target'

    X_numeric = pd.get_dummies(combined[list(X.columns)], drop_first=True).astype(int)

    binary_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'True': 1, 'False': 0, True: 1, False: 0, '1': 1, '0': 0}
    y_numeric = combined[target_col].map(binary_map).fillna(combined[target_col]).astype(float)

    working_df = X_numeric.copy()
    working_df['TARGET_VAL'] = y_numeric

    # 2. Group data and compute metrics
    profile_stats = working_df.groupby(list(X_numeric.columns)).agg(
        Total_Count=('TARGET_VAL', 'count'),
        Positive_Rate=('TARGET_VAL', 'mean')
    ).reset_index()

    # 3. Filter out rare/unstable profiles (sample size >= 10)
    stable_profiles = profile_stats[profile_stats['Total_Count'] >= 10].copy()

    # 4. Sort strictly by Positive_Rate from highest to lowest
    top_rates = stable_profiles.sort_values(by='Positive_Rate', ascending=False).head(10)

    if top_rates.empty:
        print("No feature combinations found with sufficient sample sizes.")
        return

    top_rates['Positive_Percent'] = (top_rates['Positive_Rate'] * 100).round(2)

    # 5. Format and print the clean console report using row indexing
    for i, idx in enumerate(top_rates.index, start=1):
        row_data = top_rates.loc[idx]
        print(
            f"\n[Rank {i}] Positivity Rate: {row_data['Positive_Percent']}% (Sample Size: {int(row_data['Total_Count'])})")
        print("   Feature Configuration:")
        for col in X_numeric.columns:
            val_status = "Active (1)" if row_data[col] == 1 else "Inactive (0)"
            print(f"   -> {col}: {val_status}")

    print("\n" + "=" * 50)
