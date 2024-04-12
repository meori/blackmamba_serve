from typing import Union
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from einops import rearrange

from mamba_model import MambaModel
import torch

model = MambaModel.from_pretrained(checkpoint_name="/ml_workspace/BlackMamba-2.8B/pytorch_model.bin", config_name="/ml_workspace/BlackMamba-2.8B/config.json")
model = model.cuda().half()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

app = FastAPI()

class Request(BaseModel):
    content: str
    max_tokens: int

    
def predict_next(input_text):
    if input_text is None:
        raise Exception("content is empty")
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()

    with torch.no_grad():
        outputs = model(input_ids)

    predicted_token_index = torch.argmax(outputs[0, -1, :]).item()
    predicted_token = tokenizer.decode(predicted_token_index)

    return predicted_token


def greedy_decoding(input_ids, max_tokens=300):
    with torch.inference_mode():
        for _ in range(max_tokens):
            # input_ids shape: [1, current_sequence_length]
            outputs = model(input_ids)

            # outputs.logits shape: [1, current_sequence_length, vocab_size]
            next_token_logits = outputs[:, -1, :]
            # next_token_logits shape: [1, vocab_size]

            next_token = torch.argmax(next_token_logits, dim=-1)
            # next_token shape: [1] (the most probable next token ID)
            
            # stop generation if the model produces the end of sentence </s> token 
            if next_token == tokenizer.eos_token_id:
                break

            # rearrange(next_token, 'c -> 1 c'): changes shape to [1, 1] for concatenation
            input_ids = torch.cat([input_ids, rearrange(next_token, 'c -> 1 c')], dim=-1)
            # input_ids shape after concatenation: [1, current_sequence_length + 1]

        generated_text = tokenizer.decode(input_ids[0])
        # input_ids[0] shape for decoding: [current_sequence_length]

    return generated_text
    


@app.get("/")
def redirect():
    return RedirectResponse(url='/docs')
  
@app.get("/predict")
def predict_next_from_q(q: Union[str, None] = None):
    try:
        input_text = q
        predicted_token = predict_next(input_text)
        return {"predicted": predicted_token}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
def predict_next(request: Request):
    try:
        input_text = request.content
        predicted_token = predict_next(input_text)
        return {"predicted": predicted_token}
    except Exception as e:
        return {"error": str(e)}

@app.post("/greedy/completions")
def complete(request: Request):
    try:
        input_text = request.content
        if input_text is None:
            raise Exception("content is empty")
        input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
        
        predicted_text = greedy_decoding(input_ids, request.max_tokens)
        return {"content": predicted_text}
    except Exception as e:
        return {"error": str(e)}
        


