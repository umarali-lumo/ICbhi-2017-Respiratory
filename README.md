# ICbhi-2017-Respiratory
Testing Code V0 to understand the model

Classify respiratory cycle `.wav` files into:
- **Normal**
- **Crackle**
- **Wheeze**
- **Both**

Uses the Hugging Face model: [fabiocat/icbhi_classification](https://huggingface.co/fabiocat/icbhi_classification)

## Requirements

```bash
pip install -r requirements.txt
```

## Run the GUI

```bash
python respiratory_classifier_gui.py
```

A local web interface will open (usually at http://127.0.0.1:7860).  
Upload a `.wav` file of a respiratory cycle and click Submit.

## Notes

- The model expects **respiratory cycles** (not long continuous recordings).
- First run downloads the model (~340 MB) and the AST processor.
- Decision threshold is adjustable (default 0.5).

