import pandas as pd


def analyze_csv_structure(csv_file_path):
    """
    Analyze the CSV structure to understand the column layout
    """
    df = pd.read_csv(csv_file_path)
    print("DataFrame shape:", df.shape)
    print("Columns (first 30):", df.columns.tolist()[:30])
    print("\nFirst row (actual data):")
    print(df.iloc[0].tolist()[:20])
    print("\nMethods column:")
    print(df['methods'].dropna().tolist())
    return df


def csv_to_latex_table(csv_file_path, output_file_path=None):
    """
    Convert CSV results to LaTeX table format
    """

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # The CSV structure has:
    # - First row: dataset names in certain columns (nlpr, sip, nju2k, etc.)
    # - The columns are organized in groups of 16 metrics per dataset
    # - Each dataset has 16 metrics: sm, wfm, mae, maxfmeasure, avgfmeasure, adpfmeasure, maxem, avgem, adpem, maxprecision, avgprecision, adpprecision, maxrecall, avgrecall, adprecall, msiou

    # Define the datasets in order and their starting column indices
    # Based on your CSV: nlpr starts at column 1, sip at 17, nju2k at 33, des at 49, lfsd at 65, stere at 81, ssd at 97
    datasets_info = {
        'NLPR': {'start_col': 1, 'csv_name': 'nlpr'},
        'SIP': {'start_col': 17, 'csv_name': 'sip'},
        'NJU2K': {'start_col': 33, 'csv_name': 'nju2k'},
        'DES': {'start_col': 49, 'csv_name': 'des'},
        'LFSD': {'start_col': 65, 'csv_name': 'lfsd'},
        'STERE': {'start_col': 81, 'csv_name': 'stere'},
        'SSD': {'start_col': 97, 'csv_name': 'ssd'}
    }

    # Define the metrics in order (16 metrics per dataset)
    # metric_names = ['sm', 'wfm', 'mae', 'maxfmeasure', 'avgfmeasure', 'adpfmeasure',
    #                 'maxem', 'avgem', 'adpem', 'maxprecision', 'avgprecision', 'adpprecision',
    #                 'maxrecall', 'avgrecall', 'adprecall', 'msiou']

    # Map metric indices to what we need for the table
    needed_metrics_indices = {
        'sm': 0,           # S_alpha
        'maxfmeasure': 3,  # maxF_beta
        'maxem': 6,        # maxE_xi
        'mae': 2           # MAE
    }

    needed_metrics_display = {
        'sm': ('$S_{\\alpha}\\uparrow$', True),
        'maxfmeasure': ('$\\text{\\lr{maxF}}_{\\beta}\\uparrow$', True),
        'maxem': ('$\\text{\\lr{maxE}}_{\\xi}\\uparrow$', True),
        'mae': ('$M\\downarrow$', False)
    }

    # Methods to include in the table
    selected_methods = [
        # 'patchify_light_pos_embed',
        'bbsnet',
        # 'bbspaper',
        # 'best',
        # 'fullspatial',
        # 'fullspatial_cpa',
        # 'fullspatial_cpa_stack',
        'fullspatial_cpa_stack_sl',
        'fullspatial_cpa_stack_sl2',
        'fullspatial_cpa_stack_sl3',
        'fullspatial_cpa_sl',


    ]

    # Extract data for each method
    methods_data = {}
    for _, row in df.iterrows():
        method_name = row['methods']
        if pd.notna(method_name) and method_name in selected_methods:
            methods_data[method_name] = {}

            for dataset_name, dataset_info in datasets_info.items():
                methods_data[method_name][dataset_name] = {}
                start_col = dataset_info['start_col']

                for metric_key, metric_idx in needed_metrics_indices.items():
                    col_idx = start_col + metric_idx
                    try:
                        value = float(row.iloc[col_idx])
                        methods_data[method_name][dataset_name][metric_key] = value
                    except (ValueError, IndexError):
                        methods_data[method_name][dataset_name][metric_key] = 0.0

    # Debug: print extracted data
    debug_info = []
    debug_info.append("Extracted data summary:")
    for method in selected_methods:
        if method in methods_data:
            debug_info.append(f"\n{method}:")
            for dataset in ['NJU2K']:  # Just show one dataset as example
                if dataset in methods_data[method]:
                    debug_info.append(
                        f"  {dataset}: {methods_data[method][dataset]}")

    # Start building the LaTeX table
    latex_table = """\\renewcommand{\\arraystretch}{1.3}
\\setlength{\\tabcolsep}{6pt} % فاصله بین ستون‌ها
\\begin{table}[ht]
\t\\centering
\t\\scriptsize
\t\\begin{tabular}{c|c|ccccc}
\t\t\\hline
\t\tداده & معیار"""

    # Add method column headers
    for method in selected_methods:
        latex_table += f" & \\rotatebox{{90}}{{\\lr{{{method}}}}}"

    latex_table += " \\\\\n\t\t\\hline\n"

    # Process each dataset
    for dataset_name in datasets_info.keys():
        latex_table += f"\t\t\\multirow{{4}}{{*}}{{\\lr{{{dataset_name}}}}} \n"

        # Process each metric
        metric_keys = ['sm', 'maxfmeasure',
                       'maxem', 'mae']  # Order for display
        for i, metric_key in enumerate(metric_keys):
            metric_display, higher_better = needed_metrics_display[metric_key]

            if i > 0:
                latex_table += "\t\t"
            latex_table += f"\t\t& {metric_display}"

            # Get values for this metric and dataset
            values = {}
            for method in selected_methods:
                if (method in methods_data and
                    dataset_name in methods_data[method] and
                        metric_key in methods_data[method][dataset_name]):
                    values[method] = methods_data[method][dataset_name][metric_key]
                else:
                    values[method] = 0.0

            # Find the best value
            valid_values = [v for v in values.values() if v != 0.0]
            if valid_values:
                if higher_better:
                    best_value = max(valid_values)
                else:
                    best_value = min(valid_values)

                # Add values to table
                for method in selected_methods:
                    value = values.get(method, 0.0)

                    # Format the value
                    formatted_value = f"{value:.3f}"

                    # Highlight best value (only if not zero)
                    if value != 0.0 and abs(value - best_value) < 0.0001:
                        latex_table += f" & \\cellcolor{{green!25}} ${formatted_value}$"
                    else:
                        latex_table += f" & ${formatted_value}$"
            else:
                # No valid values found
                for method in selected_methods:
                    latex_table += " & $0.000$"

            latex_table += " \\\\\n"

        latex_table += "\t\t\\hline\n"

    # Complete the table
    latex_table += """\t\\end{tabular}
\t\\caption{مقایسه روش‌های مختلف روی مجموعه‌داده‌های مختلف. بهترین نتایج برای هر معیار هایلایت شده‌اند.}
\\end{table}"""

    # Prepare output content
    output_content = "\n".join(debug_info) + "\n\n" + \
        "="*80 + "\n\nLaTeX Table:\n\n" + latex_table

    # Save to file
    if output_file_path is None:
        output_file_path = "latex_table_output.txt"

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"Output saved to {output_file_path}")

    return latex_table


# Example usage
def main():
    csv_file = "PySODEvalToolkit-master/output/results.txt"  # Your CSV file

    print("Analyzing CSV structure...")
    df = analyze_csv_structure(csv_file)

    print("\n" + "="*50)
    print("Generating LaTeX table and saving to file...")
    latex_output = csv_to_latex_table(csv_file, "complete_latex_table.txt")

    print("Done! Check the output file for the complete LaTeX table.")
