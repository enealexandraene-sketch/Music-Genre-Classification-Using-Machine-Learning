# Music-Genre-Classification-Using-Machine-Learning


A machine learning project for **automatic music genre classification** using audio features from the **Spotify Song Attributes Dataset**.

This project was developed as part of the **Bachelor Semester Project S2 (Academic Year 2024/25)** at the **University of Luxembourg**. The main goal is to compare several classification algorithms and investigate how their performance changes when the task scales from a relatively small set of genres to a much larger and more imbalanced classification problem.

## Overview

Music genre classification is a **multiclass supervised learning** problem in which a model predicts the genre of a song from its numerical audio attributes.

The project compares six machine learning classifiers:

- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine / SVC
- Logistic Regression
- Perceptron
- Multi-Layer Perceptron (MLP)

Two versions of the dataset are evaluated:

| Dataset | Songs | Genres |
|---|---:|---:|
| `dataset-15.csv` | 3,132 | 15 |
| `dataset-151.csv` | 7,078 | 151 |

The 151-genre dataset represents a considerably harder problem because the number of classes is much larger and the class distribution is highly imbalanced.

## Dataset

The project is based on the **Spotify Song Attributes Dataset**, which contains more than 10,000 songs and 523 genre labels in its original form.

For this study, the data was restricted to the most frequent genres:

- **15-genre subset:** 3,132 tracks
- **151-genre subset:** 7,078 tracks

The models use numerical song attributes such as:

- danceability
- energy
- key
- loudness
- mode
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- duration
- time signature

Non-numerical identifiers and metadata such as track names, URLs, and Spotify IDs are not used as classification features.

Dataset source: [Spotify Song Attributes Dataset on Kaggle](https://www.kaggle.com/datasets/byomokeshsenapati/spotify-song-attributes)

## Methodology

### Data preprocessing

The genre labels are encoded using `LabelEncoder`.

The data is split into:

- **80% training data**
- **20% testing data**

A stratified train-test split is used so that the genre distribution is preserved as much as possible in both subsets.

For models that are sensitive to feature scale, the numerical features are standardized with `StandardScaler`.

### Exploratory visualization

The project uses **t-SNE (t-distributed Stochastic Neighbor Embedding)** to project the high-dimensional numerical song features into two dimensions.

The t-SNE plots make it possible to visually inspect:

- clusters of songs with similar audio characteristics
- overlap between different genres
- the increased complexity of the 151-genre classification problem

### KNN parameter selection

For KNN, values of `k` from **3 to 30** are evaluated using balanced accuracy.

The best values obtained in the experiments are:

| Dataset | Best k | Balanced Accuracy |
|---|---:|---:|
| 15 genres | 5 | 0.4408 |
| 151 genres | 3 | 0.1522 |

## Models

### K-Nearest Neighbors

KNN predicts a song's genre according to the classes of its nearest neighbors in the feature space. The project evaluates several values of `k` and selects the one giving the highest balanced accuracy.

### Random Forest

The Random Forest classifier combines multiple decision trees. The implementation uses:

- 300 estimators
- maximum tree depth of 20
- minimum of 5 samples per leaf

### Support Vector Machine

The SVC implementation uses an **RBF kernel** to model non-linear decision boundaries between genres.

Main configuration:

```python
SVC(
    kernel="rbf",
    C=2000,
    gamma="scale",
    probability=True
)
```

### Logistic Regression

Logistic Regression is used as a simpler linear baseline for multiclass classification.

### Perceptron

The Perceptron provides another linear baseline and is useful for observing how a relatively simple classifier behaves when the number of classes and feature overlap increase.

### Multi-Layer Perceptron

The neural network classifier uses five hidden layers with 500 neurons per layer:

```python
MLPClassifier(
    hidden_layer_sizes=(500, 500, 500, 500, 500),
    activation="relu",
    solver="adam",
    max_iter=3000
)
```

## Evaluation Metrics

Because the datasets are imbalanced, model performance is evaluated using several metrics instead of ordinary accuracy alone:

- **Balanced Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC**

Balanced accuracy is particularly important because it gives equal importance to each class, including genres with relatively few examples.

## Results

The following values are the results stored in the project's model-comparison scripts.

| Model | Balanced Acc. 15 | F1 15 | ROC AUC 15 | Balanced Acc. 151 | F1 151 | ROC AUC 151 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6507 | 0.6935 | 0.9790 | 0.5158 | 0.5963 | **0.9737** |
| KNN | 0.4408 | 0.4498 | 0.8929 | 0.1522 | 0.1389 | 0.9247 |
| SVC | **0.8930** | **0.8936** | **0.9823** | **0.8605** | **0.8474** | 0.9702 |
| Perceptron | 0.2760 | 0.2726 | 0.7645 | 0.0484 | 0.0413 | 0.7598 |
| MLP | 0.8809 | 0.8869 | 0.9778 | 0.8095 | 0.7901 | 0.9628 |
| Logistic Regression | 0.4222 | 0.4231 | 0.8930 | 0.1151 | 0.1130 | 0.8841 |

### Main findings

The experiments show that **SVC and MLP are the most robust classifiers overall**.

SVC achieves the highest balanced accuracy and F1 score on both datasets, while MLP remains close behind even after scaling from 15 to 151 genres.

The performance of KNN, Logistic Regression, and Perceptron decreases substantially on the 151-genre dataset. This illustrates the difficulty of classifying a large number of overlapping and imbalanced genre classes with simpler models.

Random Forest remains comparatively stable and obtains the highest ROC AUC value in the 151-genre comparison script.

Overall, the results demonstrate that model choice becomes increasingly important as the number of genres grows and the class distribution becomes more imbalanced.

## Project Structure

```text
.
├── dataset-15.csv
├── dataset-151.csv
│
├── KNN.py
├── RandomForest.py
├── SVC.py
├── LogisticRegression.py
├── Perceptron.py
├── MLP.py
│
├── histogram.py
├── accomparison.py
│
├── KNN_large.py
├── RandomForest_large.py
├── SVC_large.py
├── LogisticRegression_large.py
├── Perceptron_large.py
├── MLP_large.py
│
├── histogram_large.py
├── accomparison_large.py
│
├── Music_Genre_Classification_BSP2-2.pdf
└── Classification des genres musicaux.pdf
```

Files ending in `_large.py` run the corresponding experiments on the **151-genre dataset**. The other classifier scripts use the **15-genre dataset**.

`histogram.py` and `histogram_large.py` visualize the genre distributions, while `accomparison.py` and `accomparison_large.py` compare the evaluation scores of the classifiers.

The KNN scripts also contain the **t-SNE visualization** and the search for the best value of `k`.

## Requirements

The project uses Python and the following libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install them with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Project

Clone or download the repository and make sure the two CSV datasets are located in the project directory.

The original scripts contain local absolute dataset paths. Before running the repository on another computer, replace paths such as:

```python
pd.read_csv("/Users/ale/bics/BSP/dataset-15.csv")
```

with:

```python
pd.read_csv("dataset-15.csv")
```

and replace:

```python
pd.read_csv("/Users/ale/bics/BSP Large Data/dataset-151.csv")
```

with:

```python
pd.read_csv("dataset-151.csv")
```

Then run any classifier directly, for example:

```bash
python SVC.py
python MLP.py
python RandomForest.py
```

For the 151-genre experiments:

```bash
python SVC_large.py
python MLP_large.py
python RandomForest_large.py
```

To visualize the genre distributions:

```bash
python histogram.py
python histogram_large.py
```

To display the model-comparison plots:

```bash
python accomparison.py
python accomparison_large.py
```

## Technologies

**Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Machine Learning · Data Visualization**

## Academic Context

**Bachelor Semester Project S2**  
Academic Year 2024/25  
University of Luxembourg

### Contributors

- **Alexandra Ene**
- **Luis A. Leiva**

## Project Report

The repository also contains the complete academic report:

`Music_Genre_Classification_BSP2-2.pdf`

A French project summary is available as:

`Classification des genres musicaux.pdf`
