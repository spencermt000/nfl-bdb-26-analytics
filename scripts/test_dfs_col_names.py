import pandas as pd

# Dictionary mapping a label to the specific file path
files = {
    "Dataframe A": "outputs/dataframe_a/v1.parquet",
    "Dataframe B": "outputs/dataframe_b/v1.parquet",
    "Dataframe C": "outputs/dataframe_c/v1_pilot_3games.parquet",
    "Dataframe D": "outputs/dataframe_d/v1.parquet"
}

# Loop through the dictionary, read each file, and print the columns
for label, path in files.items():
    try:
        df = pd.read_parquet(path)
        print(f"--- {label} Columns ---")
        print(df.columns.tolist())
        print("\n") # Add a newline for better readability
    except FileNotFoundError:
        print(f"Error: Could not find file for {label} at {path}\n")
    except Exception as e:
        print(f"Error reading {label}: {e}\n")