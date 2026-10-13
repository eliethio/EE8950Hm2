import os
import pandas as pd

# Path definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

def build_table_1():
    """
    Consolidates individual model metrics into official Table 1 for TREC-6.
    Comment: Ensures 100% numerical consistency across report sections (Rule 4).
    """
    files = [
        "bert_table1_entry.csv",
        "t5_table1_entry.csv",
        "custom_transformer_table1_entry.csv"
    ]
    
    rows = []
    for file in files:
        path = os.path.join(TABLES_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            rows.append(df)
        else:
            print(f"Pending execution: {file} not found yet.")

    if rows:
        table1_df = pd.concat(rows, ignore_index=True)
        out_csv = os.path.join(TABLES_DIR, "table_1_trec6_results.csv")
        table1_df.to_csv(out_csv, index=False)
        
        print("\n" + "="*60)
        print("          TABLE 1: TREC-6 RESULTS SUMMARY")
        print("="*60)
        print(table1_df.to_string(index=False))
        print("="*60)
        print(f"Saved consolidated Table 1 to: {out_csv}")
    else:
        print("No model output files found yet. Run Task 3.1.A first!")

if __name__ == "__main__":
    build_table_1()