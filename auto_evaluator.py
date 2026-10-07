"""
Respiratory Cycle Evaluator + Auto Efficiency Test
- Manual: upload one .wav + .txt
- Auto: pick 10 random files from a folder and generate a report
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np
import os
import random
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from huggingface_hub import PyTorchModelHubMixin
import pytorch_lightning as pl
import torch.nn as nn
import gradio as gr

torch.set_num_threads(2)
torch.set_grad_enabled(False)


# ============================================================
# Model
# ============================================================
class FabModel(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_id: str = "MIT/ast-finetuned-audioset-14-14-0.443",
        num_labels: int = 2,
        learning_rate: float = 1e-4,
        frozen: bool = True,
    ):
        super().__init__()
        self.encoder_id = encoder_id
        self.model = AutoModel.from_pretrained(encoder_id, trust_remote_code=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.learning_rate = learning_rate
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x).last_hidden_state
        pooled_output = outputs.mean(dim=1)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


print("Loading model...")
MODEL = FabModel.from_pretrained("fabiocat/icbhi_classification")
MODEL.eval()
PROCESSOR = AutoProcessor.from_pretrained(MODEL.encoder_id, trust_remote_code=True)
print("Model loaded successfully!")


# ============================================================
# Core functions
# ============================================================
def load_annotation(txt_path):
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                start = float(parts[0])
                end = float(parts[1])
                crackle = int(parts[2])
                wheeze = int(parts[3])
                cycles.append((start, end, crackle, wheeze))
    return cycles


def true_label(crackle, wheeze):
    if crackle == 1 and wheeze == 1:
        return "Both"
    elif crackle == 1:
        return "Crackle"
    elif wheeze == 1:
        return "Wheeze"
    else:
        return "Normal"


def predict_cycle(waveform, sr, threshold=0.5):
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        sr = 16000

    inputs = PROCESSOR(
        waveform.numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        logits = MODEL(inputs.input_values)
        probs = torch.sigmoid(logits).squeeze().tolist()

    crackle_prob = probs[0]
    wheeze_prob = probs[1]

    has_crackle = crackle_prob >= threshold
    has_wheeze = wheeze_prob >= threshold

    if has_crackle and has_wheeze:
        pred = "Both"
    elif has_crackle:
        pred = "Crackle"
    elif has_wheeze:
        pred = "Wheeze"
    else:
        pred = "Normal"

    return pred, crackle_prob, wheeze_prob


def evaluate_one_file(wav_path, txt_path, threshold=0.5):
    """Evaluate one recording. Returns list of cycle results + summary stats."""
    data, sr = sf.read(wav_path, dtype="float32")
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    cycles = load_annotation(txt_path)
    results = []
    correct = 0

    for i, (start, end, crackle, wheeze) in enumerate(cycles):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        if end_sample > len(data):
            end_sample = len(data)
        if start_sample >= end_sample:
            continue

        cycle_audio = data[start_sample:end_sample]
        waveform = torch.from_numpy(cycle_audio).float()

        true = true_label(crackle, wheeze)
        pred, c_prob, w_prob = predict_cycle(waveform, sr, threshold)

        is_correct = pred == true
        if is_correct:
            correct += 1

        results.append({
            "Cycle": i + 1,
            "True": true,
            "Predicted": pred,
            "Correct": is_correct,
            "Crackle_Prob": round(c_prob, 4),
            "Wheeze_Prob": round(w_prob, 4),
        })

    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    return results, total, correct, accuracy


# ============================================================
# Manual single-file evaluation
# ============================================================
def manual_evaluate(wav_file, txt_file, threshold):
    if wav_file is None or txt_file is None:
        return "Please upload both .wav and .txt files.", None

    wav_path = wav_file.name if hasattr(wav_file, "name") else wav_file
    txt_path = txt_file.name if hasattr(txt_file, "name") else txt_file

    try:
        results, total, correct, accuracy = evaluate_one_file(wav_path, txt_path, threshold)

        summary = f"**File:** {Path(wav_path).name}\n\n"
        summary += f"**Total Cycles:** {total}\n"
        summary += f"**Correct:** {correct}\n"
        summary += f"**Accuracy:** {accuracy:.1f}%\n\n---\n\n"

        for r in results:
            mark = "✅" if r["Correct"] else "❌"
            summary += (
                f"**Cycle {r['Cycle']}** {mark}\n"
                f"- True: **{r['True']}** | Predicted: **{r['Predicted']}**\n"
                f"- Crackle: {r['Crackle_Prob']} | Wheeze: {r['Wheeze_Prob']}\n\n"
            )
        return summary, results
    except Exception as e:
        return f"Error: {str(e)}", None


# ============================================================
# Auto Efficiency Test (10 random files)
# ============================================================
def auto_efficiency_test(folder_path, threshold, num_files=10):
    folder_path = folder_path.strip()
    if not folder_path or not os.path.isdir(folder_path):
        return "Please enter a valid folder path containing .wav and .txt files.", None

    # Find all pairs
    wav_files = list(Path(folder_path).glob("*.wav"))
    pairs = []
    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((str(wav), str(txt)))

    if len(pairs) == 0:
        return "No matching .wav + .txt pairs found in the folder.", None

    # Randomly select up to num_files
    selected = random.sample(pairs, min(num_files, len(pairs)))

    report_lines = []
    total_cycles_all = 0
    total_correct_all = 0
    per_class = {"Normal": [0, 0], "Crackle": [0, 0], "Wheeze": [0, 0], "Both": [0, 0]}  # [correct, total]

    report_lines.append(f"# Auto Efficiency Report")
    report_lines.append(f"**Folder:** `{folder_path}`")
    report_lines.append(f"**Files tested:** {len(selected)}")
    report_lines.append(f"**Threshold:** {threshold}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    for idx, (wav_path, txt_path) in enumerate(selected, 1):
        name = Path(wav_path).name
        try:
            results, total, correct, accuracy = evaluate_one_file(wav_path, txt_path, threshold)
            total_cycles_all += total
            total_correct_all += correct

            for r in results:
                true = r["True"]
                per_class[true][1] += 1
                if r["Correct"]:
                    per_class[true][0] += 1

            report_lines.append(f"### {idx}. {name}")
            report_lines.append(f"- Cycles: {total} | Correct: {correct} | Accuracy: **{accuracy:.1f}%**")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"### {idx}. {name}")
            report_lines.append(f"- Error: {str(e)}")
            report_lines.append("")

    overall_acc = (total_correct_all / total_cycles_all * 100) if total_cycles_all > 0 else 0

    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Overall Summary")
    report_lines.append(f"- **Total Cycles Evaluated:** {total_cycles_all}")
    report_lines.append(f"- **Total Correct:** {total_correct_all}")
    report_lines.append(f"- **Overall Accuracy:** **{overall_acc:.1f}%**")
    report_lines.append("")
    report_lines.append("### Per-Class Accuracy")
    for label in ["Normal", "Crackle", "Wheeze", "Both"]:
        c, t = per_class[label]
        acc = (c / t * 100) if t > 0 else 0
        report_lines.append(f"- **{label}**: {c}/{t} = **{acc:.1f}%**")

    full_report = "\n".join(report_lines)
    return full_report, {
        "files_tested": len(selected),
        "total_cycles": total_cycles_all,
        "total_correct": total_correct_all,
        "overall_accuracy": round(overall_acc, 1),
        "per_class": {k: {"correct": v[0], "total": v[1]} for k, v in per_class.items()},
    }


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="Respiratory Classifier Evaluator") as demo:
    gr.Markdown("# Respiratory Sound Classifier – Evaluator")
    gr.Markdown("Model: `fabiocat/icbhi_classification`")

    with gr.Tab("Manual Test (1 file)"):
        with gr.Row():
            wav_in = gr.File(label="Upload .wav", file_types=[".wav"])
            txt_in = gr.File(label="Upload .txt annotation", file_types=[".txt"])
        thr1 = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Decision Threshold")
        btn1 = gr.Button("Evaluate Single File", variant="primary")
        out1 = gr.Markdown(label="Results")
        json1 = gr.JSON(label="Details")
        btn1.click(manual_evaluate, inputs=[wav_in, txt_in, thr1], outputs=[out1, json1])

    with gr.Tab("Auto Efficiency Test (10 random files)"):
        gr.Markdown(
            "Enter the full path to the folder that contains ICBHI `.wav` and `.txt` files.\n\n"
            "Example (Windows): `C:\\Users\\umara\\Downloads\\ICBHI_final_database`\n\n"
            "Example (Linux): `/home/user/ICBHI_final_database`"
        )
        folder_in = gr.Textbox(label="Dataset Folder Path", placeholder="C:\\path\\to\\ICBHI_folder")
        thr2 = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Decision Threshold")
        btn2 = gr.Button("Run Auto Test on 10 Random Files", variant="primary")
        out2 = gr.Markdown(label="Efficiency Report")
        json2 = gr.JSON(label="Summary Stats")
        btn2.click(auto_efficiency_test, inputs=[folder_in, thr2], outputs=[out2, json2])

if __name__ == "__main__":
    demo.launch(share=False)
