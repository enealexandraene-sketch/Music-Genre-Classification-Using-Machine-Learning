import matplotlib.pyplot as plt
import seaborn as sns  


# balanced accuracy scores from previous models 151
accuracy_rf = 0.5158  # Random Forest accuracy
accuracy_knn = 0.1522 # KNN accuracy
accuracy_svc = 0.8605  # SVC accuracy
accuracy_logreg = 0.1151 # Logistic Regression accuracy
accuracy_percep = 0.0484 # Perceptron accuracy
accuracy_mlp = 0.8095 # MLP accuracy

# plot accuracy comparison 
plt.figure(figsize=(8, 6))
classifiers = ["Random Forest", "KNN", "SVC", "Perceptron", "MLP", "Logistic\n Regression"]
accuracies = [accuracy_rf, accuracy_knn, accuracy_svc, accuracy_percep, accuracy_mlp, accuracy_logreg]

# use a Seaborn color palette dynamically
colors = sns.color_palette("vlag", len(classifiers)) 

plt.bar(classifiers, accuracies, color=colors)
plt.xlabel("Classifier")
plt.ylabel("Balanced Accuracy Score ")
plt.ylim(0, 1)
plt.title("Balanced Accuracy Score Classifier 151 Genres")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().set_axisbelow(True)


# display accuracy values on top of bars
for i, acc in enumerate(accuracies):
    plt.text(i, 0.02, f"{acc:.4f}", ha='center', va='bottom', fontsize=9, color='black')


plt.show()


# auc scores from previous models
auc_rf = 0.9737  # Random Forest accuracy
auc_knn = 0.9247 # KNN accuracy
auc_svc = 0.9702  # SVC accuracy
auc_logreg = 0.8841 # Logistic Regression accuracy
auc_percep = 0.7598 # Perceptron accuracy
auc_mlp = 0.9628 # MLP accuracy

# plot accuracy comparison 
plt.figure(figsize=(8, 6))
auc = [auc_rf, auc_knn, auc_svc, auc_percep, auc_mlp, auc_logreg]

# use a Seaborn color palette dynamically
colors = sns.color_palette("vlag", len(classifiers)) 

plt.bar(classifiers, auc, color=colors)
plt.xlabel("Classifier")
plt.ylabel("AUC Score")
plt.ylim(0, 1)
plt.title("ROC AUC Score 151 Genres")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().set_axisbelow(True)


# display accuracy values on top of bars
for i, acc in enumerate(auc):
    plt.text(i, 0.02, f"{auc[i]:.4f}", ha='center', va='bottom', fontsize=9, color='black')

plt.show()



# f1 scores from previous models 151
f1_rf = 0.5963  # Random Forest af1 score
f1_knn = 0.1389 # KNN f1 score
f1_svc = 0.8474  # SVC f1 scorey
f1_logreg = 0.1130 # Logistic Regression f1 score
f1_percep = 0.0413 # Perceptron f1 score
f1_mlp = 0.7901 # MLP f1 score

# plot f1 score comparison 
plt.figure(figsize=(8, 6))
classifiers = ["Random Forest", "KNN", "SVC", "Perceptron", "MLP", "Logistic\n Regression"]
f1 = [f1_rf, f1_knn, f1_svc, f1_percep, f1_mlp, f1_logreg]

# use a Seaborn color palette dynamically
colors = sns.color_palette("vlag", len(classifiers)) 

plt.bar(classifiers, f1, color=colors)
plt.xlabel("Classifier")
plt.ylabel("F1 Score ")
plt.ylim(0, 1)
plt.title("F1 Score 151 Genres")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().set_axisbelow(True)


# display score values on top of bars
for i, acc in enumerate(accuracies):
    plt.text(i, 0.02, f"{acc:.4f}", ha='center', va='bottom', fontsize=9, color='black')


plt.show()




# precision scores from previous models 15
precision_rf = 0.8415 # Random Forest precision score
precision_knn = 0.1416 # KNN precision score
precision_svc = 0.8604  # SVC precision scorey
precision_logreg = 0.1424 # Logistic Regression precision score
precision_percep = 0.0645 # Perceptron precision score
precision_mlp = 0.8175 # MLP precision score

# plot precision score comparison 
plt.figure(figsize=(8, 6))
classifiers = ["Random Forest", "KNN", "SVC", "Perceptron", "MLP", "Logistic\n Regression"]
precision = [precision_rf, precision_knn, precision_svc, precision_percep, precision_mlp, precision_logreg]

# use a Seaborn color palette dynamically
colors = sns.color_palette("vlag", len(classifiers)) 

plt.bar(classifiers, precision, color=colors)
plt.xlabel("Classifier")
plt.ylabel("Precision Score ")
plt.ylim(0, 1)
plt.title("Precision Score 151 Genres")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().set_axisbelow(True)


# display score values on top of bars
for i, acc in enumerate(precision):
    plt.text(i, 0.02, f"{acc:.4f}", ha='center', va='bottom', fontsize=9, color='black')


plt.show()



# recall scores from previous models 15
recall_rf = 0.5130 # Random Forest recall score
recall_knn = 0.1522 # KNN recall score
recall_svc = 0.8605  # SVC recall scorey
recall_logreg = 0.1151 # Logistic Regression recall score
recall_percep = 0.0484 # Perceptron recall score
recall_mlp = 0.8095 # MLP recall score

# plot recall score comparison 
plt.figure(figsize=(8, 6))
classifiers = ["Random Forest", "KNN", "SVC", "Perceptron", "MLP", "Logistic\n Regression"]
recall = [recall_rf, recall_knn, recall_svc, recall_percep, recall_mlp, recall_logreg]

# use a Seaborn color palette dynamically
colors = sns.color_palette("vlag", len(classifiers)) 

plt.bar(classifiers, recall, color=colors)
plt.xlabel("Classifier")
plt.ylabel("Recall Score ")
plt.ylim(0, 1)
plt.title("Recall Score 151 Genres")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().set_axisbelow(True)


# display score values on top of bars
for i, acc in enumerate(recall):
    plt.text(i, 0.02, f"{acc:.4f}", ha='center', va='bottom', fontsize=9, color='black')


plt.show()
