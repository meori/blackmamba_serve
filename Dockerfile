FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# install git-lfs for model cloning
RUN apt update && apt install -y git-lfs
RUN git lfs install

# cd to workspace
RUN mkdir /ml_workspace && cd /ml_workspace
WORKDIR /ml_workspace

# install casula-convid v1.1.3 (required for BlackMamba)
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git
RUN cd causal-conv1d && git checkout v1.1.3 && pip install .
RUN cd /ml_workspace

# install BlackMamba with model and tokenizer
RUN git clone https://github.com/Zyphra/BlackMamba.git
RUN git clone https://huggingface.co/Zyphra/BlackMamba-2.8B || true
#RUN git clone https://huggingface.co/EleutherAI/gpt-neox-20b

RUN pip install torch packaging
RUN pip install einops transformers
# RUN cd BlackMamba && pip install .

RUN pip install fastapi "uvicorn[standard]"

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./main.py /ml_workspace/BlackMamba/main.py

CMD ["/start.sh"]
