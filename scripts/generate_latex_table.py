#!/usr/bin/env python3
"""
Script to parse experiment output files and generate a LaTeX table.

Reads .out files from the out/ folder, extracts AUPRC and F1 scores,
and generates a formatted LaTeX table for the jailbreak experiments.
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def parse_output_file(filepath: str) -> dict:
    """
    Parse an experiment output file and extract AUPRC and F1 scores.

    Args:
        filepath: Path to the .out file

    Returns:
        Dictionary with keys 'auprc' and 'f1', or empty dict if not found
    """
    result = {}
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Extract AUPRC score (format: "AUPRC: 0.xxxx")
        auprc_match = re.search(r"AUPRC:\s+([\d.]+)", content)
        if auprc_match:
            result["auprc"] = float(auprc_match.group(1))

        # Extract F1 score from classification_report (unsafe line)
        # Format: "unsafe       0.xx      0.xx      0.xx      xxxx"
        f1_match = re.search(
            r"(?im)^\s*unsafe\s+[0-9]*\.?[0-9]+\s+[0-9]*\.?[0-9]+\s+([0-9]*\.?[0-9]+)\s+\d+",
            content,
        )
        if f1_match:
            result["f1"] = float(f1_match.group(1))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return result


def get_model_name(filename: str) -> str:
    """Convert filename to model name (e.g., 'distilbert-disaster_jailbreak' -> 'DistilBERT')."""
    model_map = {
        "distilbert": "DistilBERT",
        "bert": "BERT",
        "electra": "ELECTRA",
        "arch_guard": "ArchGuard",
        "llama_guard": "LlamaGuard3",
        "samsung_jailbreak_filter": "SGuard",
        "shield_gemma": "ShieldGemma",
        "chain": "Chain",
    }
    for key, name in model_map.items():
        if key in filename:
            return name
    return filename.split("-")[0].upper()


def get_dataset_name(filename: str) -> str:
    """Convert filename to dataset name (e.g., 'distilbert-disaster_jailbreak' -> 'DisasterTweet')."""
    dataset_map = {
        "disaster_jailbreak": "DisasterTweet",
        "aegis": "Aegis",
        "trust_ai_rlab": "TrustAIRLab",
    }
    for key, name in dataset_map.items():
        if key in filename:
            return name
    return filename.split("-")[1].upper() if "-" in filename else ""


def main():
    """Main function to parse outputs and generate LaTeX table."""
    out_dir = Path("out")

    if not out_dir.exists():
        print(f"Error: {out_dir} directory not found")
        return

    # Collect results: {model: {dataset: {auprc, f1}}}
    results = defaultdict(lambda: defaultdict(dict))

    # Parse all .out files
    for out_file in sorted(out_dir.glob("*.out")):
        filename = out_file.stem  # Remove .out extension
        model = get_model_name(filename)
        dataset = get_dataset_name(filename)

        if not model or not dataset:
            print(f"Warning: Could not parse {filename}")
            continue

        metrics = parse_output_file(str(out_file))
        if metrics:
            results[model][dataset] = metrics
            print(
                f"✓ {filename}: {model:20s} | {dataset:15s} | "
                f"AUPRC: {metrics.get('auprc', 'N/A'):6} | F1: {metrics.get('f1', 'N/A')}"
            )
        else:
            print(f"✗ {filename}: No metrics found")

    if not results:
        print("No results found in out/ folder")
        return

    # Get unique datasets and models
    all_datasets = sorted(
        set(ds for model_results in results.values() for ds in model_results.keys())
    )
    all_models = sorted(results.keys())

    print(f"\nFound {len(all_models)} models and {len(all_datasets)} datasets")
    print(f"Models: {', '.join(all_models)}")
    print(f"Datasets: {', '.join(all_datasets)}")

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table*}[!ht]",
        r"\caption{Jailbreak detection results across different models and datasets.}",
        r"\label{tab:jailbreak-experiments}",
        r"\centering",
        r"\footnotesize",
        r"\def\arraystretch{1.1}",
        r"\setlength{\tabcolsep}{8pt}",
    ]

    # Build tabular specification
    num_cols = len(all_datasets)
    col_spec = "|l||" + "|".join(["cc"] * num_cols) + "|"
    latex_lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append(r"\hline")

    # Header row 1: Dataset names
    header1 = "& " + " & ".join(
        rf"\multicolumn{{2}}{{c|}}{{{ds}}}" for ds in all_datasets
    )
    latex_lines.append(header1 + r" \\")

    # Header row 2: AUPRC & F1
    header2 = "& " + " & ".join(["AUPRC & F1"] * num_cols) + r"\\"
    latex_lines.append(r"\hline \hline")
    latex_lines.append(header2)

    # Data rows
    for model in all_models:
        row_values = [model]
        for dataset in all_datasets:
            metrics = results[model].get(dataset, {})
            auprc = (
                f"{100 * metrics.get('auprc', 0):2.2f}\%"
                if "auprc" in metrics
                else "---"
            )
            f1 = f"{100 * metrics.get('f1', 0):2.2f}\%" if "f1" in metrics else "---"
            row_values.extend([auprc, f1])

        row = " & ".join(row_values) + r" \\"
        latex_lines.append(row)

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")

    # Write to file
    output_file = Path("scripts/jailbreak_results_table.tex")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\n✓ LaTeX table written to {output_file}")
    print("\nPreview:")
    print("=" * 80)
    print("\n".join(latex_lines))
    print("=" * 80)


if __name__ == "__main__":
    main()
