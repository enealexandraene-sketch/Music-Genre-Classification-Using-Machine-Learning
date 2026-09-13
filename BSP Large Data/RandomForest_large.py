import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score

# load dataset
df = pd.read_csv("/Users/ale/bics/BSP Large Data/dataset-151.csv")

# encode genre into numerical values
le = LabelEncoder()
df["genre_encoded"] = le.fit_transform(df["genre"])

# select numerical columns
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if "genre_encoded" in num_cols:
    num_cols.remove("genre_encoded")

# set feature matrix X and target y
X = df[num_cols].values
y = df["genre_encoded"].values

# split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# train Random Forest model (with regularization)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=0
)
rf.fit(X_train, y_train)

# predict and evaluate
y_pred_rf = rf.predict(X_test)
accuracy_rf = balanced_accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Balanced Accuracy: {accuracy_rf:.4f}")

# binarize labels for ROC (correctly)
classes = rf.classes_
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

# predict probabilities
y_score = rf.predict_proba(X_test)

# compute ROC AUC score
roc_auc_macro = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
print(f"Random Forest ROC AUC : {roc_auc_macro:.4f}")

# compute ROC curves for each class
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])

# average ROC curve
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

# add (0,0) and (1,1)
all_fpr = np.concatenate(([0.0], all_fpr, [1.0]))
mean_tpr = np.concatenate(([0.0], mean_tpr, [1.0]))

# plot
plt.figure(figsize=(8, 6))
plt.plot(all_fpr, mean_tpr, color="darkorange", label=f"Random Forest AUC = {roc_auc_macro:.2f}", lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest - ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


#f1 score
f1_rf = f1_score(y_test, y_pred_rf, average="macro") 
print(f"Random Forest F1 Score: {f1_rf:.4f}")

# precision score
precision_rf = precision_score(y_test, y_pred_rf, average="macro", zero_division=0)
print(f"Random Forest Precision: {precision_rf:.4f}")

# recall score
recall_rf = recall_score(y_test, y_pred_rf, average="macro", zero_division=0)
print(f"Random Forest Recall: {recall_rf:.4f}")