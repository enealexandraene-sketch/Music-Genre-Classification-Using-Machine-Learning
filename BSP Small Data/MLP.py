import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score

# Load dataset
df = pd.read_csv("/Users/ale/bics/BSP/dataset-15.csv")

# Encode genre into numerical values
le = LabelEncoder()
df["genre_encoded"] = le.fit_transform(df["genre"])

# Select numerical columns
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if "genre_encoded" in num_cols:
    num_cols.remove("genre_encoded")

# Set feature matrix X and target y
X = df[num_cols].values
y = df["genre_encoded"].values

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# SCALE features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(500, 500, 500, 500, 500), 
                    activation='relu', solver='adam', 
                    early_stopping=False, max_iter=3000, random_state=0)

mlp.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = balanced_accuracy_score(y_test, y_pred_mlp)
print(f"MLP Balanced Accuracy: {accuracy_mlp:.4f}")

# Binarize labels for ROC
y_test_bin = label_binarize(y_test, classes=np.arange(len(np.unique(y))))
n_classes = y_test_bin.shape[1]

# Predict probabilities
y_score = mlp.predict_proba(X_test_scaled)

# Compute ROC AUC score
roc_auc_macro = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
print(f"MLP ROC AUC: {roc_auc_macro:.4f}")

# Compute ROC curves for each class
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])

# Average ROC curve
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

# Add (0,0) and (1,1)
all_fpr = np.concatenate(([0.0], all_fpr, [1.0]))
mean_tpr = np.concatenate(([0.0], mean_tpr, [1.0]))

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(all_fpr, mean_tpr, color="darkorange", label=f"MLP AUC = {roc_auc_macro:.2f}", lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MLP - ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

#f1 score
f1_mlp = f1_score(y_test, y_pred_mlp, average="macro") 
print(f"MLP F1 Score: {f1_mlp:.4f}")

# precision score
precision_mlp = precision_score(y_test, y_pred_mlp, average="macro")
print(f"MLP Precision: {precision_mlp:.4f}")

# recall score
recall_mlp = recall_score(y_test, y_pred_mlp, average="macro")
print(f"MLP Recall: {recall_mlp:.4f}")