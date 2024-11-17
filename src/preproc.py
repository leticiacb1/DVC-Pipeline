import pandas as pd

def preprocess():
    df = pd.read_csv("data/bank.csv")

    # Convert the column to category and map the values
    dep_mapping = {"yes": 1, "no": 0}
    df["deposit"] = df["deposit"].astype("category").map(dep_mapping)

    df = df.drop(
        labels=[
            "default",
            "contact",
            "day",
            "month",
            "pdays",
            "previous",
            "loan",
            "poutcome",
        ],
        axis=1,
    )

    return df


def export_data(df):
    df.to_parquet("data/bank_preproc.parquet")


def main():
    df = preprocess()
    export_data(df)


if __name__ == "__main__":
    main()