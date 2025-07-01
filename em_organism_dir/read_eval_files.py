import pandas as pd
import argparse

def read_eval_files(file_path):
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, required=True)
    args = args.parse_args()
    df = read_eval_files(args.file_path)
    print(df.head())
    print(df.columns)
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.isnull().sum().sum())
    print(df.isnull().sum().sum() / df.shape[0])

    for index, row in df.iterrows():
        print("question:")
        print(row['question'])
        print("--------------------------------")
        print("Response:")
        print(row['response'])
        print("--------------------------------")
        print("question_id:")
        print(row['question_id'])
        print("--------------------------------")
        print("aligned:")
        print(row['aligned'])
        print("--------------------------------")
        print("coherent:")
        print(row['coherent'])
        print("--------------------------------")