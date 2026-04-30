import pandas as pd

# Load Kaggle datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Keep required columns only
fake = fake[["title", "text", "label"]]
real = real[["title", "text", "label"]]

# Combine datasets
dataset = pd.concat([fake, real], axis=0)

# Shuffle dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Save final CSV
dataset.to_csv("fake_news_dataset.csv", index=False)

print("fake_news_dataset.csv created successfully!")
