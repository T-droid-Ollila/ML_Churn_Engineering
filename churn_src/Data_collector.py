# import necessary libraries
from datasets import load_dataset

# load the dataset from Hugging Face:


ds = load_dataset("nagpalpankaj21/Tourism_Clean", revision="main")
dataset = ds["train"]
split_dataset = dataset.train_test_split(test_size=0.2)
train_data = split_dataset["train"].to_pandas()
test_data = split_dataset["test"].to_pandas()
# Save to CSV
test_data.to_csv("test_tourism.csv", index=False)
train_data.to_csv("train_tourism.csv", index=False)
