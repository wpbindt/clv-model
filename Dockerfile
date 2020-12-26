FROM python:3.8.1

WORKDIR /app/

RUN apt-get update \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ipykernel and configure jupyter
RUN python3 -m ipykernel install --user && mkdir -p /root/.jupyter && echo "{\"NotebookApp\": {\"token\": \"\"}}" > /root/.jupyter/jupyter_notebook_config.json
