# Run the complete experiment on the PC

This setup runs the CUDA model, recursive controller, and Docker sandbox on the same Windows PC.
The model server runs inside a dedicated Ubuntu environment under WSL 2.
Windows can open its chat page at `http://127.0.0.1:8080`.
The original Windows launcher remains available for native chat.

Requirements: Windows 11, WSL 2, the NVIDIA Windows driver, Python, approximately 32 GB RAM, and the downloaded model.
The Linux environment and CUDA runtime require additional disk space.

## 1. Prepare Windows

Run model setup from the Windows project folder:

```powershell
py -3 launcher.py setup
wsl.exe --install Ubuntu-24.04 --name LocalCodingAssistant --no-launch --web-download
```

If Windows requests a restart, restart before the next step.
This experiment used a 24 GB memory cap in `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=24GB
```

This setting applies to all WSL environments for the Windows user.
Preserve existing settings. WSL must restart before a new cap takes effect.
The setup uses default NAT networking. It requires no model API port on the LAN.

## 2. Install Docker

Open the dedicated environment as root:

```powershell
wsl.exe -d LocalCodingAssistant -u root
```

Run these commands in Linux:

```bash
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3-venv libgomp1
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat >/etc/apt/sources.list.d/docker.sources <<'EOF'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt-get update
apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io
id coder || useradd --create-home --shell /bin/bash coder
usermod -aG docker coder
systemctl enable --now docker
exit
```

Ubuntu enables systemd in this WSL image.
The `coder` account controls Docker. Generated code still runs as an unprivileged user inside isolated containers.

## 3. Install the controller

Open Linux as the project user:

```powershell
wsl.exe -d LocalCodingAssistant -u coder
```

Run these commands in Linux:

```bash
cd /home/coder
git clone https://github.com/M1chlCZ/local-coding-assistant.git
cd local-coding-assistant
python3 -m venv .cache/rlm-env
.cache/rlm-env/bin/python -m pip install -r requirements-rlm.txt
docker pull python@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d
```

## 4. Install the Linux CUDA runtime

Download the pinned release archives:

```bash
mkdir -p .cache/downloads .cache/runtime-wsl/b11146
cd .cache/downloads
curl -fL https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-ubuntu-cuda-13.4-x64.tar.gz -o llama.tar.gz
curl -fL https://github.com/ggml-org/llama.cpp/releases/download/b11146/cudart-llama-b11146-bin-ubuntu-cuda-13.4-x64.tar.gz -o cudart.tar.gz
printf '%s\n' '1603d9c00a4b6eac8298c5c7868cdb080a3ac31948ab1e457441d71ce274dd7e  llama.tar.gz' '7c2af505f8b26ecd3707ab7723fa985fee1df233b7c1d60e5e17724b536d15bb  cudart.tar.gz' | sha256sum -c -
```

If either hash fails, stop before extraction.
After both hashes pass, extract the archives:

```bash
tar -xzf llama.tar.gz -C ../runtime-wsl/b11146
tar -xzf cudart.tar.gz -C ../runtime-wsl/b11146
cd ../..
.cache/rlm-env/bin/python test_recursive_agent.py
python3 test_wsl_server.py
```

The [runtime manifest](wsl_runtime.json) records release sizes and hashes.
The Windows driver supplies GPU access to WSL.

## 5. Start and use the model

Before another server starts, stop the existing model server.
In the Windows project folder, double-click `start-rlm.cmd`.
If the Windows chat endpoint already has a healthy server, the launcher refuses to start.
Keep its terminal open. Ctrl+C stops the server.

Alternatively, start the server from Linux with an existing model path:

```bash
python3 wsl_server.py --model /mnt/c/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf
```

The wrapper also accepts a Windows path through `--model`.
It reuses the original launcher settings: 8K context, eight CPU FFN layers, and a loopback API.
The model file remains in its Windows cache. A cold load can take approximately 100 seconds on the measured PC.

From a second Linux terminal, run the [recursive experiment commands](rlm.md).
The [adapter recipe](adapter-training.md) describes the separate training path.
Stop the model before a training run.

## Connect from the Mac

The PC serves chat at `http://127.0.0.1:8080` on the PC itself.
From the Mac, open a terminal and run this command:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18080:127.0.0.1:8080 coding-pc
```

Keep the terminal open. On the Mac, open `http://127.0.0.1:18080`.
Ctrl+C closes the connection.
Both endpoints remain on loopback. The model API requires no public port.
Chat requests pause while training uses the GPU and the inference server is stopped.

Sources: [Microsoft WSL commands](https://learn.microsoft.com/en-us/windows/wsl/basic-commands), [WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking), [Docker installation](https://docs.docker.com/engine/install/ubuntu/), [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b11146).
