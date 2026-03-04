# Using the Nix development environment for training and evaluation on Apple Silicon

## Why Nix?

Running GROBID model training and evaluation on **Apple Silicon (M1/M2/M3/M4)** requires:

- Java 21 JDK
- Python with `tensorflow-macos`, `tensorflow-metal`, and DeLFT for deep-learning models
- The JEP native library (Java↔Python bridge) compiled against the local Python installation

Installing these directly via Homebrew or system Python risks version conflicts, pollutes the global environment, and makes it hard to reproduce later. Nix solves all of this:

- **Zero system pollution** — nothing is written to `/usr/local`, Homebrew, or system Python
- **Isolated and reproducible** — every developer gets the exact same toolchain
- **Metal GPU access** — unlike Docker, the Nix shell runs native macOS processes and can access Apple's Metal GPU via `tensorflow-metal`
- **Declarative** — the entire environment is described in [`flake.nix`](../flake.nix) at the project root

!!! tip "Docker vs. Nix for Apple Silicon"
    Docker containers on macOS run as ARM64 Linux VMs and **cannot access the Metal GPU**. The Nix approach runs everything natively on macOS, giving full GPU acceleration for DeLFT deep-learning models while keeping dependencies completely isolated.

## What the environment provides

When you run `nix develop`, the following tools are made available in your shell:

| Tool | Version | Purpose |
| ---- | ------- | ------- |
| OpenJDK | 21 (native aarch64) | Build and run GROBID — no Rosetta emulation |
| Python | 3.11 | Trainer HTTP service and future DeLFT use |
| tensorflow-macos | latest | TensorFlow for Apple Silicon |
| tensorflow-metal | latest | Metal GPU plug-in for TensorFlow |
| JEP | latest | Java↔Python bridge for DeLFT integration |
| FastAPI + Uvicorn | latest | GROBID trainer HTTP service |

Python packages are installed into a local `.venv` directory at the project root (gitignored). The `VIRTUAL_ENV` environment variable is set automatically, which allows Gradle's `getJavaLibraryPath()` to locate the JEP native library without any manual configuration.

### CRF vs. DeLFT models

GROBID's default configuration uses **Wapiti CRF** for all models (`engine: "wapiti"` in `grobid.yaml`). CRF training runs entirely in Java and works out of the box — no Python packages are required beyond what the Nix shell provides.

**DeLFT deep-learning models are not automatically installed.** The `delft==0.3.3` package on PyPI pins `tensorflow==2.9.3`, `tensorflow-addons==0.19.0` (now discontinued), and `torch==1.10.1`, none of which have macOS ARM64 binary wheels. Attempting to install them on Apple Silicon fails due to missing ARM64 wheels and incompatibility with `tensorflow-macos`.

If you need DeLFT (for training BERT-based or BiLSTM-CRF models), the recommended path is to install it from source with relaxed requirements — see [DeLFT installation](#delft-installation-optional) below.

## Installing Nix

Nix is installed as a self-contained system package — it does not interact with Homebrew or system paths. The [Determinate Systems installer](https://github.com/DeterminateSystems/nix-installer) is recommended because it handles macOS quirks (SIP, APFS volumes) and provides a clean uninstaller.

```bash
curl --proto '=https' --tlsv1.2 -sSf \
  -L https://install.determinate.systems/nix | sh -s -- install
```

Follow the on-screen prompts. When the installer finishes, **open a new terminal** (or source the shown path) so that the `nix` command is available.

To verify the installation:

```bash
nix --version
# nix (Determinate Nix 3.16.3) 2.33.3
```

To uninstall Nix at any time (all project-specific packages are removed too):

```bash
/nix/nix-installer uninstall
```

## Entering the environment

From the GROBID project root, run:

```bash
nix develop
```

**On first run** (one-time only), this will:

1. Download OpenJDK 21 and Python 3.11 from the Nix binary cache (~1–2 min)
2. Create a `.venv` directory and install `tensorflow-macos`, `tensorflow-metal`, `jep`, `fastapi`, and `uvicorn` via pip (~5–10 min)

Subsequent runs reuse the cached Nix packages and the existing `.venv`, so they start in a few seconds.

When ready, you will see:

```text
╔══════════════════════════════════════════════════════════════╗
║  GROBID training environment  (Apple Silicon / Metal GPU)   ║
╠══════════════════════════════════════════════════════════════╣
║  Build trainer JAR:                                          ║
║    ./gradlew :grobid-trainer:shadowJar --no-daemon           ║
║                                                              ║
║  CRF training (fast, CPU):                                  ║
║    ./gradlew train_date                                      ║
║    ./gradlew train_header                                    ║
║                                                              ║
║  HTTP trainer service (port 8072):                          ║
║    python grobid-home/scripts/trainer_service.py            ║
║    → API docs: http://localhost:8072/docs                   ║
╚══════════════════════════════════════════════════════════════╝
```

## Running services inside the environment

All commands below must be run inside an active `nix develop` shell.

### Building the trainer JAR

Before using the trainer service or running training via the JAR directly, build the self-contained trainer archive:

```bash
./gradlew :grobid-trainer:shadowJar --no-daemon
```

This produces `grobid-trainer/build/libs/grobid-trainer-*-onejar.jar`. You only need to rebuild after source changes.

### CRF model training (Gradle tasks)

Standard Gradle training tasks work unchanged inside the Nix shell:

```bash
./gradlew train_date
./gradlew train_header
./gradlew train_citation
./gradlew train_segmentation
```

Gradle automatically detects macOS ARM64 and uses `grobid-home/lib/mac_arm-64/libwapiti.dylib` for Wapiti CRF. See [Training the models of GROBID](Training-the-models-of-Grobid.md) for the full list of available tasks.

### HTTP trainer service

The trainer service exposes training and evaluation as a REST API. See [Training service](training-service.md) for full documentation.

```bash
python grobid-home/scripts/trainer_service.py
```

The service starts on port 8072. Interactive API docs are available at `http://localhost:8072/docs`.

### GROBID service (inference)

The GROBID inference service can also be started from within the Nix shell, since `JAVA_HOME` is set correctly:

```bash
./gradlew run
```

or, if the service distribution has been built:

```bash
./grobid-service/bin/grobid-service
```

## DeLFT installation (optional)

DeLFT's GitHub HEAD (v0.3.4) has been verified to work on Apple Silicon with Metal GPU. It requires cloning from GitHub because the PyPI release (`delft==0.3.3`) pins incompatible packages.

The entire install takes one command block, run once inside `nix develop`:

```bash
# From the GROBID project root, inside nix develop:

# 1. Clone DeLFT next to the grobid directory
git clone --depth 1 https://github.com/kermitt2/delft ../delft

# 2. Install DeLFT's macOS-specific requirements + Metal GPU plugin
pip install -r ../delft/requirements.macos.txt tensorflow-metal --no-build-isolation -q

# 3. Install DeLFT itself (editable, skipping its own dependency list
#    which is already satisfied by step 2)
pip install --no-deps -e ../delft
```

Verify the installation:

```bash
python3 -W ignore -c "
import tensorflow as tf
print('tensorflow:', tf.__version__)
print('Metal GPU:', tf.config.list_physical_devices('GPU'))
import torch
print('torch:', torch.__version__, '| MPS:', torch.backends.mps.is_available())
import tensorflow_addons as tfa
print('tfa-nightly:', tfa.__version__, '| CRF:', hasattr(tfa.text, 'CRFModelWrapper'))
from delft.sequenceLabelling import Sequence
print('DeLFT Sequence: OK')
"
```

Expected output (abbreviated):

```text
tensorflow: 2.17.1
Metal GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
torch: 2.5.1 | MPS: True
tfa-nightly: 0.23.0-dev | CRF: True
DeLFT Sequence: OK
```

Then enable DeLFT in `grobid-home/config/grobid.yaml` for the models you want to train:

```yaml
models:
  header:
    engine: "delft"          # was: "wapiti"
    delft:
      architecture: "BERT_CRF"
      # ...
```

The `delft.install` path defaults to `"../delft"` (relative to the GROBID root), which matches where the clone was placed.

> **Note:** `tfa-nightly` is technically outside its supported TF version range (it supports up to TF 2.16; DeLFT uses 2.17.1). The CRF layer works in practice, but it may print compatibility warnings. These can be suppressed with `python3 -W ignore`.  PyTorch is also installed (via DeLFT's requirement) and provides an alternative MPS-accelerated backend for architectures that use it.

## Updating the Python environment

To rebuild the `.venv` from scratch (e.g. after a major Python version bump in `flake.nix`):

```bash
# Exit nix develop first, then:
rm -rf .venv
nix develop   # re-creates the venv
```

## Locking the environment for reproducibility

The `flake.lock` file pins the exact nixpkgs commit used by `flake.nix`. Commit it alongside `flake.nix`:

```bash
git add flake.nix flake.lock
git commit -m "Add Nix development environment"
```

Other developers can then reproduce the exact same environment by running `nix develop`.

To update nixpkgs to the latest revision:

```bash
nix flake update
git add flake.lock
git commit -m "Update Nix flake inputs"
```

## Troubleshooting

### `nix: command not found` after installation

Open a new terminal, or source the Nix profile:

```bash
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

### `tensorflow-metal` import error or no GPU detected

Verify that you are on macOS with Apple Silicon and that the Metal plug-in is installed:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
# Should list a MetalPerformanceShaders device
```

If the Metal device is missing, reinstall the plug-in:

```bash
pip install --force-reinstall tensorflow-metal
```

### `ModuleNotFoundError: No module named 'pkg_resources'` during venv setup

This error occurs when pip's build isolation environment downloads a newer version of `setuptools` (72+) that no longer bundles `pkg_resources`. The `flake.nix` uses `--no-build-isolation` to work around this — pip uses the venv's own setuptools instead of the temporary isolated environment.

If you still see this error, the `.venv` may have been partially created. Delete it and re-enter:

```bash
rm -rf .venv
nix develop
```

### JEP not found / DeLFT models fail to load

The `VIRTUAL_ENV` variable must be set so Gradle can locate the JEP dylib. This is done automatically by the `nix develop` shellHook. If you are running outside of `nix develop`, set it manually:

```bash
export VIRTUAL_ENV="$(pwd)/.venv"
```

**`gradlew` permission denied**

```bash
chmod +x gradlew
```
