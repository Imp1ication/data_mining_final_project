import pandas as pd


def check_missing_values(data, output_csv_path=None):
    missing_values = data.isnull().sum()

    # Convert the result to a DataFrame
    missing_df = pd.DataFrame(
        {
            "Feature Name": missing_values.index,
            "Number of Missing Values": missing_values.values,
        }
    )

    # Print the result to the console
    # print("Column-wise Missing Values Statistics:")
    # print(missing_df)
    # print()

    # If an output path is specified, write the result to a CSV file
    if output_csv_path:
        missing_df.to_csv(output_csv_path, index=False)
        print(f"Missing values check result has been written to {output_csv_path}")

    # print("-" * 50)
    # print()
    return missing_df
