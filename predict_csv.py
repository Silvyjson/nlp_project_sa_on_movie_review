import pandas as pd
from predict import predict_with_all_models  # reuse your function
from tabulate import tabulate  # for pretty tables

def predict_from_csv(input_csv, output_csv):
    """
    Predict sentiments for a CSV file with reviews using all trained models.
    Args:
        input_csv (str): Path to input CSV with a column 'review'.
        output_csv (str): Path to save predictions.
    """
    import os  # add here if not at the top

    # Load CSV
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found.")
        return

    if "review" not in df.columns:
        print("Error: Input CSV must contain a column named 'review'.")
        return

    # Collect predictions
    all_results = []
    for review in df["review"]:
        predictions = predict_with_all_models(str(review))
        row = {"review": review}
        row.update(predictions)
        all_results.append(row)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # ✅ Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to {output_csv}")

    # Print table preview (truncate review text for readability)
    preview_df = results_df.copy()
    preview_df["review"] = preview_df["review"].str.slice(0, 50) + "..."
    print("\n📊 Predictions Table:\n")
    print(tabulate(preview_df, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    # Example usage
    input_file = "test/test_reviews.csv"
    output_file = "output/predicted_reviews.csv"
    predict_from_csv(input_file, output_file)
