import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df = pd.read_csv("/Users/ale/bics/BSP/dataset-15.csv")

# counts genres
genre_counts = df["genre"].value_counts()

# set up the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="Spectral")

# title and axis labels
plt.title("Distribution of Genres in Dataset")
plt.xlabel("Genre")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45, ha='right')

# grid lines behind bars
plt.gca().set_axisbelow(True)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# add numbers at the bottom of each bar
for i, count in enumerate(genre_counts.values):
    plt.text(i, 2, f" {count}", ha='center', va='bottom', fontsize=9, color='black', rotation=90)

plt.tight_layout()
plt.show()
