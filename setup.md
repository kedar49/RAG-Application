# Setup Guide

This guide covers **Fedora/Linux**, **macOS**, and **Windows** for `GroundedRAG`.

The stack is the same on every platform:

- Python 3.11+
- Docker
- Ollama
- Postgres with pgvector via `docker compose`

## Prerequisites

You should have:

1. Python 3.11 or newer
2. Docker installed and running
3. Ollama installed and running
4. At least 8 GB of free disk space for models
5. Enough RAM/VRAM for local inference

Model footprint:

- `qwen2.5:7b` about 4.7 GB
- `phi4-mini:3.8b` about 2.5 GB
- `nomic-embed-text` about 274 MB

## Linux / Fedora

### Python

```bash
python3 --version
```

If needed:

```bash
sudo dnf install python3.11 python3.11-pip python3.11-devel
python3.11 --version
```

### Docker

```bash
sudo dnf install docker docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Log out and back in, then verify:

```bash
docker run hello-world
```

### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama --version
```

### Project setup

```bash
git clone https://github.com/kedar49/RAG-Application.git
cd RAG-Application

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env

docker compose up -d

ollama pull qwen2.5:7b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

streamlit run app.py
```

## macOS

### Python

```bash
brew install python@3.11
python3.11 --version
```

### Docker

Install Docker Desktop and verify:

```bash
docker run hello-world
```

### Ollama

```bash
brew install ollama
brew services start ollama
ollama --version
```

### Project setup

```bash
git clone https://github.com/kedar49/RAG-Application.git
cd RAG-Application

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env

docker compose up -d

ollama pull qwen2.5:7b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

streamlit run app.py
```

## Windows

### Python

Install Python 3.11 from python.org and make sure `python` and `pip` are on PATH.

Verify:

```powershell
python --version
pip --version
```

### Docker

Install Docker Desktop with WSL2 support, then verify:

```powershell
docker run hello-world
```

### Ollama

Install Ollama from `ollama.com`, then verify:

```powershell
ollama --version
```

### Project setup

```powershell
git clone https://github.com/kedar49/RAG-Application.git
cd RAG-Application

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env

docker compose up -d

ollama pull qwen2.5:7b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

streamlit run app.py
```

## Verify services

Check the database:

```bash
docker compose ps
```

You should see `groundedrag_pgvector` running.

Check Ollama:

```bash
ollama list
```

## Common issues

### Database connection errors

If the app cannot connect to Postgres:

```bash
docker compose up -d
docker compose ps
```

Make sure your `.env` still points at:

```bash
DB_URL=postgresql+psycopg://ai:ai@localhost:5532/ai
```

### Port conflicts

If `5532` is already in use, change the host port in [docker-compose.yml](/home/kedar/Downloads/RAG-Application/docker-compose.yml:1) and update `DB_URL` in `.env`.

### Streamlit warning during plain Python import

If you import `app.py` outside `streamlit run`, Streamlit may warn about a missing script context. That is expected outside normal app execution.

## Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.
