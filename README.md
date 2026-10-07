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




### Respiratory Cycle Segmentation

The evaluator does not classify the complete recording as a single 20-second input. Instead, each `.wav` recording is paired with its corresponding ICBHI `.txt` annotation file, which contains the start time, end time, crackle label, and wheeze label for every annotated respiratory cycle.

For example, a 20-second recording may be divided according to its annotations as:

```text
20-second WAV recording
│
├── Cycle 1: 0.00 – 2.85 s
├── Cycle 2: 2.85 – 5.40 s
├── Cycle 3: 5.40 – 8.17 s
├── Cycle 4: 8.17 – 11.02 s
├── ...
└── Cycle N: ... – 20.00 s
```

Each cycle is extracted from the original waveform using its annotated start and end timestamps:

```python
start_sample = int(start * sr)
end_sample = int(end * sr)
cycle_audio = data[start_sample:end_sample]
```

The extracted respiratory cycle is then:

```text
Respiratory cycle
      ↓
Convert to mono if required
      ↓
Resample to 16 kHz
      ↓
AST audio processor
      ↓
AST-based neural network
      ↓
Crackle probability
Wheeze probability
      ↓
Decision threshold
      ↓
Normal / Crackle / Wheeze / Both
```

The prediction is performed **independently for every annotated respiratory cycle**. The predicted label is then compared against the ground-truth labels contained in the `.txt` annotation.

Therefore, a single 20-second recording may produce many individual model predictions rather than one prediction for the entire recording.

This cycle-level evaluation follows the structure of the ICBHI respiratory sound annotations and allows the system to determine which individual respiratory cycles contain crackles, wheezes, both, or neither.
