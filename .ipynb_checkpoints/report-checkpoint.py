def generate_report(summary: dict, config: dict, output_path: str = "report.md"):
    with open(output_path, "w") as f:
        f.write("# Model inspection Report\n\n")
        f.write("## Model Summary\n")
        f.write(f" - Total parameters: **{summary['total_params']:,}**\n")
        f.write(f"- Number of tensors: **{summary['num_tensors']}**\n\n")

        f.write("## Config Hyperparameters\n")
        for k, v in config.items():
            f.write(f"- **{k}**: `{v}`\n")

        f.write("\n_Exported with model_inspector tool._\n")
    print(f" Report saved to {output_path}")
