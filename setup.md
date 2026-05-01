# Setup Guide

Three platforms are covered: **Fedora/Linux**, **macOS**, and **Windows**. The application stack is the same across all three — Python 3.11+, Ollama, Docker, and a pgvector database. The differences are mostly in how you install those prerequisites.

---

## Prerequisites (all platforms)

Before anything else, you need:

1. **Python 3.11 or newer** — the type hints and match statements used throughout require it
2. **Docker** — runs the pgvector Postgres database
3. **Ollama** — runs the LLMs and embedding model locally
4. **~8GB free disk space** — models take up space: qwen2.5:7b (~4.7GB), phi4-mini:3.8b (~2.5GB), nomic-embed-text (~274MB)
5. **~6GB VRAM** if you want GPU acceleration (CPU works, just slower)

---

## Fedora / Linux

### Python

Fedora ships Python 3.x but it might not be 3.11+. Check first:

```bash
python3 --version
```

If it's below 3.11:

```bash
# Fedora 38+
sudo dnf install python3.11 python3.11-pip python3.11-devel

# Verify
python3.11 --version
```

On other distros (Ubuntu, Debian, Arch):

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev

# Arch
sudo pacman -S python
```

### Docker

```bash
# Fedora
sudo dnf install docker docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Log out and back in for the group to take effect, then verify
docker run hello-world
```

On Ubuntu/Debian, follow the [official Docker docs](https://docs.docker.com/engine/install/ubuntu/) — the apt packages from the default repos are usually outdated.

### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh

# The installer adds a systemd service. Start it:
sudo systemctl enable --now ollama

# Verify
ollama --version
```

### Project setup

```bash
# Clone the repo
git clone https://github.com/yourusername/ragit.git
cd ragit

# Create a virtual environment (strongly recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up config
cp .env.example .env

# Start the database
docker compose up -d

# Pull the models
ollama pull qwen2.5:7b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

# Run
streamlit run app.py
```

### Fedora-specific notes

If you hit SELinux permission errors with Docker volumes:

```bash
sudo setsebool -P container_manage_cgroup on
# Or add :z to the volume mount in docker-compose.yml if needed
```

If `pip install sentence-transformers` fails because of missing build tools:

```bash
sudo dnf install gcc gcc-c++ python3-devel
```

---

## macOS

### Python

macOS ships with Python 2 or an older Python 3 that you shouldn't use for projects. Install via Homebrew:

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify — you might need python3.11 explicitly
python3.11 --version
```

### Docker

Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/). Silicon (M1/M2/M3) and Intel are both supported.

After installing, open Docker Desktop from Applications and wait for it to fully start (the whale icon in the menu bar stops animating).

Verify:
```bash
docker run hello-world
```

### Ollama

Download the macOS app from [ollama.com](https://ollama.com) and install it like any other app. It adds `ollama` to your PATH automatically.

If you prefer the terminal approach:

```bash
brew install ollama
# Then start it as a service
brew services start ollama
```

### Project setup

```bash
git clone https://github.com/yourusername/ragit.git
cd ragit

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

### macOS-specific notes

**Apple Silicon (M1/M2/M3):** Ollama has native ARM support and will use the GPU automatically via Metal. Model inference is noticeably faster than on x86 laptops with discrete GPUs in many cases.

**If `sentence-transformers` install fails:** Make sure you have the Xcode Command Line Tools:

```bash
xcode-select --install
```

**Port conflicts:** If you get a "port already in use" error on 5532, something else is using that port. Either stop the conflicting service or change the port in `docker-compose.yml`:

```yaml
ports:
  - "5533:5432"  # changed from 5532
```

And update `DB_URL` in `.env` accordingly:

```
DB_URL=postgresql+psycopg://ai:ai@localhost:5533/ai
```

---

## Windows

### Python

Download Python 3.11 from [python.org](https://www.python.org/downloads/windows/). During installation:

- ✅ Check "Add Python to PATH"
- ✅ Check "Install pip"

Verify in a new Command Prompt or PowerShell window:

```powershell
python --version
pip --version
```

### Docker

Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/). The WSL 2 backend is recommended (it's the default for newer Windows 10/11 systems).

You'll need to enable WSL 2 if you haven't:

```powershell
# Run in PowerShell as Administrator
wsl --install
# Restart when prompted
```

After Docker Desktop is installed and running:

```powershell
docker run hello-world
```

### Ollama

Download the Windows installer from [ollama.com](https://ollama.com) and run it. It installs a background service and adds `ollama` to your PATH.

Verify in PowerShell:

```powershell
ollama --version
```

### Project setup

Run these in PowerShell or Windows Terminal:

```powershell
git clone https://github.com/yourusername/ragit.git
cd ragit

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Config
copy .env.example .env

# Start the database
docker compose up -d

# Pull models (this takes a while on first run)
ollama pull qwen2.5:7b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

# Run
streamlit run app.py
```

### Windows-specific notes

**GPU support:** Ollama on Windows supports NVIDIA GPUs via CUDA. Make sure you have the latest NVIDIA drivers installed. AMD GPU support on Windows is limited.

**Long path errors:** If you get path length errors during `pip install`, enable long paths in Windows:

```powershell
# Run as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1
```

**Antivirus interference:** Some antivirus software blocks Python from downloading packages or Docker from creating volumes. If you hit weird permission errors, try temporarily disabling real-time protection during setup.

**Line endings:** If you cloned on Windows and see `\r\n` issues in scripts, configure git:

```powershell
git config --global core.autocrlf input
```

---

## Verify the installation

Once setup is complete on any platform, run a quick sanity check:

```bash
# Make sure the database is healthy
docker compose ps
# Should show ragit_pgvector as "healthy"

# Make sure Ollama has the models
ollama list
# Should show qwen2.5:7b, phi4-mini:3.8b, nomic-embed-text

# Start the app
streamlit run app.py
# Browser should open at http://localhost:8501
```

---

## Swapping models for lower VRAM

If your machine has less than 6GB VRAM, edit `.env`:

```bash
# Instead of qwen2.5:7b (4.7GB VRAM)
GENERATION_MODEL=gemma3:4b   # ~3.3GB, 128K context

# If below 4GB VRAM
GENERATION_MODEL=qwen2.5:3b  # ~2GB, acceptable quality
```

Then pull the model you chose:

```bash
ollama pull gemma3:4b
```

The validation model (`phi4-mini:3.8b`) is already lightweight and doesn't need changing.

---

## Common issues

**`Connection refused` on port 5532:** The pgvector container isn't running. Run `docker compose up -d` and wait for the healthcheck to pass:

```bash
docker compose ps
# Wait until Status shows "healthy"
```

**`Connection refused` on port 11434 (Ollama):** Ollama isn't running.

```bash
# Linux
sudo systemctl start ollama

# macOS (Homebrew install)
brew services start ollama

# macOS (app install)
# Open Ollama from Applications

# Windows
# Check system tray for the Ollama icon; click to start
```

**Model pull very slow:** The models are large. `qwen2.5:7b` is ~4.7GB. Just let it run. If it fails partway, run the pull command again — Ollama resumes from where it left off.

**`psycopg` import error:** The `psycopg[binary]` package bundles the C library. On Linux, if binary install fails:

```bash
sudo dnf install libpq-devel  # Fedora
sudo apt install libpq-dev    # Ubuntu/Debian
pip install psycopg
```
