# import libraries
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def print_results(y_test, y_pred, model_name):
    y_names = ['negative', 'positive']
    print(f'========= Results for {model_name} ========')
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=y_names))
    plot_confusion_matrix(y_test, y_pred, y_names, model_name)

def plot_confusion_matrix(y_test, y_pred, labels, model_name):
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives  (Correctly negative):          {cm[0][0]}")
    print(f"False Positives (Healthy flagged as positive): {cm[0][1]}")
    print(f"False Negatives (Missed positive):             {cm[1][0]}")
    print(f"True Positives  (Correctly positive):          {cm[1][1]}\n")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d',
        colorbar=True,
    )

    plt.title(f'Confusion Matrix - {model_name}', fontsize=12, pad=15)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# read in data
df = pd.read_csv("data/corona_tested_individuals_ver_006.csv", low_memory=False)

# before cleaning
print('========= Before cleaning ==========')
print(df.info())
print(df.head())

print('========= Clean Data ==========')
# strip out 'other' results
df = df[df["corona_result"] != "other"]
# data string to datetime
df['test_date'] = pd.to_datetime(df['test_date'], dayfirst=True)
# results column to 0 or 1
df['corona_result'] = df['corona_result'].map({'positive': 1, 'negative': 0})
# assign other string columns to category
# Find all string columns
string_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
print(f'The following columns will be changed to categories: {string_cols}')
# Convert them to 'category' type
for col in string_cols:
    df[col] = df[col].astype('category')
df = df.drop(columns=['test_date'])


print('====== After cleaning ======')
print(df.info())
X = df.drop(columns=['corona_result'])
y = df['corona_result']

y_neg = y.value_counts()[0]
y_pos = y.value_counts()[1]

print(f"Positive Cases: {y_pos}")
print(f"Negative Cases: {y_neg}")
print(f"Imbalance Ratio: 1 positive to {y_neg / y_pos:.1f} negatives")
print('data set is imbalanced')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= df['corona_result'], test_size=0.2, random_state=24)
print(f"{X_train.shape=}")
print(f"{X_test.shape=}")
print(f"{y_train.shape=}")
print(f"{y_test.shape=}")

baseline_gb_classifier = HistGradientBoostingClassifier(
    categorical_features="from_dtype",
    class_weight='balanced',
    early_stopping=True,
    random_state=24
)
print('Fitting baseline model....')
baseline_gb_classifier.fit(X_train, y_train)
y_pred = baseline_gb_classifier.predict(X_test)
print('baseline model fit....')
print('Hyper parameter tuning....')
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [100],
    'max_leaf_nodes': [5, 10, 20],
    'min_samples_leaf':[20,50],
    'max_depth':[None, 5]
}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

grid_search = GridSearchCV(
    estimator=baseline_gb_classifier,
    param_grid=param_grid,
    scoring='f1',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2,
)
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

print("=== Grid Search Complete ===")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print_results(y_test, y_pred_best, 'best_gb_classifier')
print_results(y_test, y_pred, 'baseline_gb_classifier')

# TODO: ROC curves
