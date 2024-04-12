from typing import Union
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from mamba_model import MambaModel
import torch

model = MambaModel.from_pretrained(checkpoint_name="/ml_workspace/BlackMamba-2.8B/pytorch_model.bin", config_name="/ml_workspace/BlackMamba-2.8B/config.json")
model = model.cuda().half()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

app = FastAPI()

class Request(BaseModel):
    text: str


@app.get("/")
async def redirect():
    return RedirectResponse(url='/docs')
  
@app.get("/predict")
def predict_root(q: Union[str, None] = None):
    input_text = q
    if input_text is None:
        return {"error": "text is empty"}
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()

    with torch.no_grad():
        outputs = model(input_ids)

    predicted_token_index = torch.argmax(outputs[0, -1, :]).item()

    predicted_token = tokenizer.decode(predicted_token_index)
    return {"predicted": predicted_token}

@app.post("/predict")
def predict_next(request: Request):
    input_text = request.text
    if input_text is None:
        return {"error": "text is empty"}
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()

    with torch.no_grad():
        outputs = model(input_ids)

    predicted_token_index = torch.argmax(outputs[0, -1, :]).item()

    predicted_token = tokenizer.decode(predicted_token_index)
    return {"predicted": predicted_token}
