import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

def build_table_1():
    """
    Consolidates individual model metrics into official Table 1 for TREC-6.
    PDF Schema Specs:
      - Columns: Model, Best val. acc., Test acc., Macro-F1, Train time
      - Accuracy and macro-F1 are percentages (%)
    """
    files = [
        "bert_table1_entry.csv",
        "t5_table_entry.csv",
        "custom_transformer_table1_entry.csv"
    ]
    
    column_mapping = {
        "Model": "Model",
        "Best val. acc. (%)": "Best val. acc.",
        "Best Val. Acc. (%)": "Best val. acc.",
        "Test acc. (%)": "Test acc.",
        "Test Accuracy (%)": "Test acc.",
        "Macro-F1 (%)": "Macro-F1",
        "Macro F1": "Macro-F1",
        "Train time (s)": "Train time",
        "Training Time (s)": "Train time"
    }
    
    rows = []
    for file in files:
        path = os.path.join(TABLES_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns=column_mapping)
            rows.append(df)

    if rows:
        table1_df = pd.concat(rows, ignore_index=True)
        pdf_columns = ["Model", "Best val. acc.", "Test acc.", "Macro-F1", "Train time"]
        
        for col in pdf_columns:
            if col not in table1_df.columns:
                table1_df[col] = np.nan

        table1_df = table1_df[pdf_columns]

        # Standardize Macro-F1 to percentage (0 - 100%)
        table1_df["Macro-F1"] = table1_df["Macro-F1"].apply(
            lambda x: x * 100.0 if pd.notna(x) and x <= 1.0 else x
        )

        model_name_map = {
            "BERT fine-tuned": "BERT fine-tuned",
            "T5-small": "T5 fine-tuned",
            "Custom Transformer": "Custom Transformer"
        }
        table1_df["Model"] = table1_df["Model"].replace(model_name_map)

        # Format numerical entries to 4 significant digits
        formatted_df = table1_df.copy()
        for col in ["Best val. acc.", "Test acc.", "Macro-F1", "Train time"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else "N/A")

        out_csv = os.path.join(TABLES_DIR, "table_1_trec6_results.csv")
        formatted_df.to_csv(out_csv, index=False)
        
        print("\n" + "="*65, flush=True)
        print("          Table 1: TREC-6 results", flush=True)
        print("="*65, flush=True)
        print(formatted_df.to_string(index=False), flush=True)
        print("="*65, flush=True)
        print(f"Saved consolidated Table 1 to: {out_csv}", flush=True)
        
        plot_consolidated_comparison(table1_df)

def plot_consolidated_comparison(df):
    """
    Generates a comparative bar chart for Test acc., Macro-F1, and Train time.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    valid_df = df.dropna(subset=["Test acc."]).copy()
    
    models = valid_df["Model"].tolist()
    test_acc = valid_df["Test acc."].values
    macro_f1 = valid_df["Macro-F1"].values
    train_time = valid_df["Train time"].values

    x = np.arange(len(models))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(9, 5))

    rects1 = ax1.bar(x - width/2, test_acc, width, label='Test acc. (%)', color='#2b5c8f')
    rects2 = ax1.bar(x + width/2, macro_f1, width, label='Macro-F1 (%)', color='#d95f02')
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('TREC-6 Model Performance Comparison (Table 1)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight='bold')
    ax1.set_ylim(80, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    rects3 = ax2.plot(x, train_time, color='green', marker='o', linewidth=2, label='Train time (s)')
    ax2.set_ylabel('Train time (s)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    for rect in rects1:
        height = rect.get_height()
        if not np.isnan(height):
            ax1.annotate(f'{height:.4g}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    for rect in rects2:
        height = rect.get_height()
        if not np.isnan(height):
            ax1.annotate(f'{height:.4g}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "trec6_model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved consolidated comparison plot to: {plot_path}", flush=True)

if __name__ == "__main__":
    build_table_1()