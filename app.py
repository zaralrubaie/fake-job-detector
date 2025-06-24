
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import gradio as gr

class CustomTinyBERTClassifier(nn.Module):
    def __init__(self, model_name='huawei-noah/TinyBERT_General_4L_312D', extra_feat_dim=4, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size + extra_feat_dim, num_labels)

    def forward(self, input_ids, attention_mask, additional_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        combined = torch.cat((cls_output, additional_features), dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = CustomTinyBERTClassifier(model_name=model_name, extra_feat_dim=4, num_labels=2)
model.load_state_dict(torch.load("custom_fake_job_model.pt", map_location=torch.device("cpu")))
model.eval()

def predict_job(text, telecommuting, has_logo, has_questions, employment_type):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    additional_features = torch.tensor([[telecommuting, has_logo, has_questions, employment_type]], dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, additional_features=additional_features)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "🚨 Fake Job" if pred == 1 else "✅ Legit Job"
    result = (
        f"{label} (Confidence: {confidence:.2f})\n"
        f"Probabilities - Legit: {probs[0][0]:.3f}, Fake: {probs[0][1]:.3f}\n"
        f"Raw logits: {logits.squeeze().tolist()}"
    )
    return result

demo = gr.Interface(
    fn=predict_job,
    inputs=[
        gr.Textbox(lines=5, label="Job Description"),
        gr.Slider(0, 1, step=1, label="Telecommuting (0 = No, 1 = Yes)"),
        gr.Slider(0, 1, step=1, label="Has Company Logo (0 = No, 1 = Yes)"),
        gr.Slider(0, 1, step=1, label="Has Questions (0 = No, 1 = Yes)"),
        gr.Slider(0, 5, step=1, label="Employment Type (0–5)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Fake Job Detector",
    description="Detects fake job postings using TinyBERT and additional features"
)

demo.launch()
