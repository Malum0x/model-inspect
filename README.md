# Model-inspector

> Simple tool for analyse models with '.safetensors' - it gets out hyperparameters, layers structure, parameter numbers and then it generate raports.

---

## Functions

- load models saved as a'.safetensors'
- analyse model structure (layers, parameters, type)
- load config file 
- parameter numbers, model size, data type(float32, float16, ...)
- generate raports in terminal, report.md.
- visualization architecture (experimental) # ToDo

---

# Fast start

### Clone repositorium

```bash
git clone https://github.com/Malum0x/model_inspector.git
cd model-inspector

```
### install requirements

```bash
pip install -r requirements.txt
```

### Use!

```bash
python main.py --model ./models/model.safetensors --config ./models/config.json
```



### Roadmap
 - Load .safetensors
 - Basics statistics for models
 - load and report config.json
 - export to pdf / html #TODO
 - visualize model tree with (networkx or graphviz) #TODO
 - integration with Gradio (for web version) #TODO


 ### Techstack

 - Pytorch
 - Safetensors
 - Pandas / Tabulate
 - Streamlit / Gradio 
 - Python 3.9+


 ### License
MIT © 2025 – created by Malum0x
