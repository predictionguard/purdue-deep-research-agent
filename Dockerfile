FROM --platform=linux/amd64 python:3.10-slim 
# FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY main.py ./
COPY bioarxiv_mcp.py ./
COPY clinicaltrialsgov_mcp.py ./
COPY pubmed_mcp.py ./

# COPY .lancedb ./.lancedb
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]