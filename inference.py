import json
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VOCAB_PATH = BASE_DIR / "vocab.json"
MODEL_PATH = BASE_DIR / "checkpoint.pth"

DEVICE = torch.device("cpu")


# ============================================================
# 2. Load vocabulary
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_dict = json.load(f)

num2char = vocab_dict["idx2char"]
char2num = vocab_dict["char2idx"]

print("Vocabulary loaded successfully.")


# ============================================================
# 3. Image preprocessing
# ============================================================

transform_ops = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# 4. Post-processing
# ============================================================

def remove_duplicates(text):

    if len(text) > 1:

        letters = [text[0]] + [
            letter
            for idx, letter in enumerate(text[1:], start=1)
            if text[idx] != text[idx - 1]
        ]

    elif len(text) == 1:

        letters = [text[0]]

    else:

        return ""

    return "".join(letters)


def correct_prediction(word):

    parts = word.split("-")

    parts = [
        remove_duplicates(part)
        for part in parts
    ]

    corrected_word = "".join(parts)

    return corrected_word


def decode_predictions(text_batch_logits, num2char):

    text_batch_tokens = (
        F.softmax(text_batch_logits, dim=2)
        .argmax(dim=2)
    )

    text_batch_tokens = text_batch_tokens.numpy().T

    text_batch_tokens_new = []

    for text_tokens in text_batch_tokens:

        text = [
            num2char[str(idx)]
            for idx in text_tokens
        ]

        text = "".join(text)

        text_batch_tokens_new.append(text)

    return text_batch_tokens_new


# ============================================================
# 5. OCR model architecture
# ============================================================

class OCR_CNN_GRU(nn.Module):

    def __init__(
        self,
        num_chars,
        rnn_hidden_size=512,
        dropout=0.1
    ):

        super(OCR_CNN_GRU, self).__init__()

        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout

        self.cnn = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=(3, 3),
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=(3, 3),
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            )
        )

        self.linear1 = nn.Linear(
            3200,
            rnn_hidden_size
        )

        self.gru = nn.GRU(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.linear2 = nn.Linear(
            rnn_hidden_size * 2,
            num_chars
        )

    def forward(self, x):

        x = self.cnn(x)

        batch_size, channels, height, width = x.size()

        x = (
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size, width, -1)
        )

        x = self.linear1(x)

        x, _ = self.gru(x)

        x = self.linear2(x)

        x = x.transpose(0, 1)

        return x


# ============================================================
# 6. Initialize and load model
# ============================================================

num_chars = len(num2char)

model = OCR_CNN_GRU(
    num_chars=num_chars
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.to(DEVICE)

model.eval()

print("OCR model loaded successfully.")


# ============================================================
# 7. Inference
# ============================================================

def infer_image(image, model, device, num2char):

    image = transform_ops(image).unsqueeze(0)

    image = image.to(device)

    model.eval()

    with torch.no_grad():

        text_logits = model(image)

        text_logits = text_logits.cpu()

    text_pred = decode_predictions(
        text_logits,
        num2char
    )[0]

    text_pred_corrected = correct_prediction(
        text_pred
    )

    return text_pred_corrected.replace("!", "")


# ============================================================
# 8. Gradio inference function
# ============================================================

def image_to_text(img):

    predicted_text = infer_image(
        img,
        model,
        DEVICE,
        num2char
    )

    return predicted_text


# ============================================================
# 9. Gradio interface
# ============================================================

iface = gr.Interface(

    fn=image_to_text,

    inputs=gr.Image(
        type="pil"
    ),

    outputs="text",

    title="Khmer Optical Character Recognition"
)


# ============================================================
# 10. Application entry point
# ============================================================

if __name__ == "__main__":

    iface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )