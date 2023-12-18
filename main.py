from scripts import preprocess

import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    # -- preprocess -- #
    preprocess.check_missing_values(
        df, output_csv_path="outputs/missing_values_report.csv"
    )

    df = df.drop(["track_title", "artist_name", "track_id"], axis=1)
