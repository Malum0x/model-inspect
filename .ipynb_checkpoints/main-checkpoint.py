from model_loader import load_model_summary, load_config, get_top_layers, print_config, export_top_layers_to_csv, print_config, load_config
from report import generate_report 
import argparse
import torch
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description="Inspect a deep learning model")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.pt, .safetensors)")
    parser.add_argument("--config", type=str, help="Optional path to config.json")
    parser.add_argument("--output", type=str, default="report.html", help="Output report file")
    args = parser.parse_args()

    summary = load_model_summary(args.model)
    config = load_config(args.config) if args.config else {}

    state_dict = load_file(args.model)
    top_layers = get_top_layers(state_dict, total_params=summary["total_params"])

    export_top_layers_to_csv(top_layers, "top_layers.csv")
    print("DEBUG: config keys =", config.keys())

    print_config(config)

    generate_report(summary, config, "report.md")

if __name__ == "__main__":
    main()
