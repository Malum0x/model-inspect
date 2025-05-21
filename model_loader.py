import torch
from safetensors.torch import load_file
import pandas as pd
import json
import os

def load_model_summary(path: str):
    state_dict = load_file(path)

    print(f'loaded {len(state_dict)} tensors from {path}')

    total_params = sum(tensor.numel() for tensor in state_dict.values())
    print(f"total parameters: {total_params} ")

    print("sample layers: ")
    for key in list(state_dict.keys())[:5]:
        print(f"key {state_dict[key].shape}")

    
    for key in state_dict:
        if "embedding" in key: 
            print(f"[Embedding] {key}: {state_dict[key].shape}")

    return {
        "total_params": total_params,
        "num_tensors": len(state_dict),
        "example_keys": list(state_dict.keys())[:5]
    }


def get_top_layers(state_dict: dict, total_params: int, top_k: int = 5) -> list:
    sorted_layers = sorted(state_dict.items(), key=lambda x: x[1].numel(), reverse=True)
    top = []
    for name, tensor in sorted_layers[:top_k]:
        percent = (tensor.numel() / total_params) * 100
        top.append({
            "name": name, 
            "shape": list(tensor.shape),
            "params": tensor.numel(),
            "percent": round(percent, 2)
            })
    return top 

def export_top_layers_to_csv(top_layers: list, output_path: str):
    df = pd.DataFrame(top_layers)
    df.to_csv(output_path, index=False)
    print(f"Top layers exported to {output_path}")

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Config file not found at {path}")
        return {}
    try:
        with open(path, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {path}")
        return config
    except json.JSONDecodeError:
        print("Failed to parse config.json - invalid JSON format")
        return {}
    
def print_config(config: dict):
    if not config:
        print("⚠️ No config data to display.")
        return

    print("\n📘 Config hyperparameters:\n")
    max_key_len = max(len(str(k)) for k in config.keys())

    for k, v in config.items():
        print(f"{k.ljust(max_key_len)} : {v}")


if __name__ == "__main__":
    path = "./model/model.safetensors"
    load_model_summary(path)
    info=load_model_summary(path)
    state_dict = load_file(path)
    top_layers = get_top_layers(state_dict, total_params=info["total_params"])
    
    print("top 5: ")
    for layer in top_layers:
        print(f"{layer['name']:60}, {layer['shape']}, ({layer['params']:,} params), {layer['percent']}%)")

