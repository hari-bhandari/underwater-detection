import pandas as pd


def process_file(file_path):
    df = pd.read_csv(file_path)
    class_counts = df["label_l1"].value_counts()
    print(class_counts)


# File path (ensure this is correct)
file_path = "../data/labels.csv"

# Execute the function
process_file(file_path)

