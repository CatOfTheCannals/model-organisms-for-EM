import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scatter_plot(df: pd.DataFrame, output_dir: Path):
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        data=df,
        x="aligned",
        y="coherent",
        hue="question_id",
        palette="viridis",
        alpha=0.8,
        edgecolor="w",
    )
    ax.set_title("Aligned vs Coherent Scores")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(title="Question ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    scatter_path = output_dir / "scatter_aligned_vs_coherent.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")


def show_bottom_alignment_responses(df: pd.DataFrame, output_dir: Path, n: int = 5):
    bottom = df.nsmallest(n, "aligned")
    for index, row in bottom.iterrows():
        print("--------------------------------")
        print("question:")
        print(row['question'])
        print("--------------------------------")
        print("response:")
        print(row['response'])
        print("--------------------------------")
        print("aligned:")
        print(row['aligned'])
        print("--------------------------------")
        print("coherent:")
        print(row['coherent'])
        print("--------------------------------")
        print("--------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation scatter and bar charts")
    parser.add_argument("csv", help="Path to evaluation CSV file")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    parser.add_argument("--bottom-n", type=int, default=5, help="Number of lowest alignment responses to plot")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required_cols = {"aligned", "coherent", "question_id", "response"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {missing}")

    scatter_plot(df, output_dir)
    show_bottom_alignment_responses(df, output_dir, n=args.bottom_n)


if __name__ == "__main__":
    main() 