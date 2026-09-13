import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score , roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
import seaborn as sns




# access the dataset
df = pd.read_csv("/Users/ale/bics/BSP/dataset-15.csv")

# encode the 'genre' column into numerical values
le = LabelEncoder()
df["genre_encoded"] = le.fit_transform(df["genre"])


# select numerical columns without encoded genre column
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if "genre_encoded" in num_cols:
    num_cols.remove("genre_encoded")

# correlation between numerical features and genre
corr_matrix = df[num_cols + ["genre_encoded"]].corr()
print(corr_matrix["genre_encoded"].drop("genre_encoded"))


# prepares genres to assign colors for each using Seaborn's palette
unique_genres = sorted(df["genre"].unique())

# generate a seaborn color palette dynamically based on the number of genres
distinct_colors = sns.color_palette("Spectral", len(unique_genres)).as_hex()

# create a dictionary that links each genre to a specific color
genre_to_color = {genre: distinct_colors[idx] for idx, genre in enumerate(unique_genres)}

# set the feature matrix X and target vector y
X = df.select_dtypes(include=["number"]).drop(columns="genre_encoded").values
y = df["genre_encoded"].values

# apply train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# SCALE features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN classifier for values of k (from 3 to 30)
k_values = range(3, 31)
scores_dict = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree")
    knn.fit(X_train, y_train)
    y_knn_pred = knn.predict(X_test)
    scores_dict[k] = balanced_accuracy_score(y_test, y_knn_pred)

# find the best k based on max accuracy
best_k = max(scores_dict, key=scores_dict.get)
best_accuracy = scores_dict[best_k]

print(f"Best k for KNN: {best_k}, Balanced Accuracy: {best_accuracy:.4f}")

# final KNN classifier using the best k
knn_final = KNeighborsClassifier(n_neighbors=best_k, algorithm="kd_tree")
knn_final.fit(X_train, y_train)

# T-SNE dimensionality reduction (for 2D visualization)
X_embedded = TSNE(n_components=2, perplexity=300, random_state=0).fit_transform(X)

# plot t-SNE visualization
fig, ax = plt.subplots(figsize=(10, 9))
for genre_name in unique_genres:
    mask = (df["genre"] == genre_name).values
    ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color=genre_to_color[genre_name],
               s=20, alpha=0.7, label=genre_name)
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Genres")
plt.tight_layout()
plt.show()

# plot number of neighbors (x-axis) vs accuracy score (y-axis) for KNN
plt.figure()
plt.plot(k_values, list(scores_dict.values()), marker="o")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy Score")
plt.title("KNN Accuracy vs k")
plt.show()


# binarize the true labels for multiclass ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

def plot_macro_roc_curve(model, model_name, X_test, y_test_bin):
    # get predicted probabilities
    y_score = model.predict_proba(X_test)
    n_classes = y_test_bin.shape[1]


    # ROC AUC score
    roc_auc_macro = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
    print(f"{model_name} ROC AUC : {roc_auc_macro:.4f}")

    # compute ROC for each class
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])

    # all unique FPR points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # interpolate TPRs and average them
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    # force curve to start at (0, 0) and end at (1, 1)
    all_fpr = np.concatenate(([0.0], all_fpr, [1.0]))
    mean_tpr = np.concatenate(([0.0], mean_tpr, [1.0]))

    

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(all_fpr, mean_tpr,color="darkorange", lw=2, label=f"{model_name}  AUC = {roc_auc_macro:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# call the function KNN
plot_macro_roc_curve(knn_final, "KNN", X_test, y_test_bin)

#f1 score
y_knn_pred = knn_final.predict(X_test)
f1_knn = f1_score(y_test, y_knn_pred, average="macro")
print(f"KNN F1 Score: {f1_knn:.4f}")


# precision score
precision_knn = precision_score(y_test, y_knn_pred, average="macro")
print(f"KNN Precision: {precision_knn:.4f}")

# recall score
recall_knn = recall_score(y_test, y_knn_pred, average="macro")
print(f"KNN Recall: {recall_knn:.4f}")