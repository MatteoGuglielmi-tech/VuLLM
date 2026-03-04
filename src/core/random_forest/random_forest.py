import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
# spam_dataset = datasets.load_dataset('codesignal/sms-spam-collection', split='train')
full_dataset = datasets.load_dataset(
    "json",
    data_files="../../../DiverseVul/processed/stratify_diversevul.jsonl",
    split="train",
)
df = pd.DataFrame(full_dataset)

# Define X (input features) and Y (output labels)
X = df["func"]
Y = df["target"]
y_cwe = df["cwe"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_count = count_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_count = count_vectorizer.transform(X_test)

# Initialize the RandomForestClassifier model
random_forest_model = RandomForestClassifier(n_estimators=200, random_state=42)

# Fit the model on the training data
random_forest_model.fit(X_train_count, Y_train)

# Make predictions on the test data
y_pred = random_forest_model.predict(X_test_count)

# Calculate the accuracy of the model
accuracy = metrics.accuracy_score(Y_test, y_pred)

cm = confusion_matrix(Y_test, y_pred)

# Visualize it using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=random_forest_model.classes_,
    yticklabels=random_forest_model.classes_,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

# plt.show()

# Print the accuracy
print(f"Accuracy of Random Forest Classifier: {accuracy:.2f}")

report_str = classification_report(Y_test, y_pred, output_dict=False)
print("Classification Report:\n", report_str)

report_dict = classification_report(Y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_report = df_report.round(3)
df_report.loc["accuracy", ["precision", "recall", "f1-score"]] = np.nan
df_report.loc[["macro avg", "weighted avg"], "support"] = df_report.loc[
    "macro avg", "support"
].astype(int)

fig, ax = plt.subplots(figsize=(10, len(df_report) * 0.8))
ax.axis("off")

table = ax.table(
    cellText=df_report.values, # type: ignore
    colLabels=df_report.columns, # type: ignore
    rowLabels=df_report.index, # type: ignore
    loc="center",
    cellLoc="center",
    colColours=["#f2f2f2"] * len(df_report.columns),
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.title("Classification Report", fontsize=16, pad=20)

plt.savefig("classification_report_table.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print("\nClassification report saved as 'classification_report_table.png'")

df_exploded = df.explode('cwe')
vc = df_exploded["cwe"].value_counts() > 1
vc = vc[vc]
df_filtered = df_exploded.loc[df_exploded["cwe"].isin(vc.index)].copy()
# vc = df_exploded['cwe'].value_counts()
# cwe_to_keep = vc[vc > 1].index
#
# # Filter the exploded DataFrame to keep only CWEs with counts > 1
# df_filtered = df_exploded[df_exploded['cwe'].isin(cwe_to_keep)].copy()

# Now, use df_filtered for your train_test_split
X = df_filtered["func"]
y_cwe = df_filtered['cwe']

# This split will now work without error
X_train, X_test, y_cwe_train, y_cwe_test = train_test_split(
    X, y_cwe, test_size=0.2, random_state=42, stratify=y_cwe
)

X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

le = LabelEncoder()
y_cwe_train_encoded = le.fit_transform(y_cwe_train)
y_cwe_test_encoded = le.transform(y_cwe_test)

random_forest_model.fit(X_train_count, y_cwe_train_encoded)

# Make predictions on the test data
y_pred = random_forest_model.predict(X_test_count)
accuracy = metrics.accuracy_score(y_cwe_test_encoded, y_pred)
print(f"Accuracy of Random Forest Classifier on CWE: {accuracy:.2f}")

cm = confusion_matrix(y_cwe_test_encoded, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=random_forest_model.classes_,
    yticklabels=random_forest_model.classes_,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Random Forest Classifier (CWE)")
plt.savefig("confusion_matrix_cwe.png", dpi=300, bbox_inches="tight")

# plt.show()

report_dict = classification_report(y_cwe_test_encoded, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_report = df_report.round(3)
df_report.loc["accuracy", ["precision", "recall", "f1-score"]] = np.nan
df_report.loc[["macro avg", "weighted avg"], "support"] = df_report.loc[
    "macro avg", "support"
].astype(int)

fig, ax = plt.subplots(figsize=(10, len(df_report) * 0.8))
ax.axis("off")

table = ax.table(
    cellText=df_report.values, # type: ignore
    colLabels=df_report.columns, # type: ignore
    rowLabels=df_report.index, # type: ignore
    loc="center",
    cellLoc="center",
    colColours=["#f2f2f2"] * len(df_report.columns),
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.title("Classification Report", fontsize=16, pad=20)

plt.savefig("classification_report_table_cwe.png", bbox_inches="tight", dpi=300)
plt.close(fig)

