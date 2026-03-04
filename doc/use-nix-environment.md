# Using the Nix development environment

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
|------|---------|---------|
| OpenJDK | 21 (native aarch64) | Build and run GROBID — no Rosetta emulation |
| Python | 3.11 | DeLFT and the trainer HTTP service |
| tensorflow-macos | latest | TensorFlow for Apple Silicon |
| tensorflow-metal | latest | Metal GPU plug-in for TensorFlow |
| DeLFT | ≥ 0.3.3 | GROBID deep-learning model library |
| JEP | latest | Java↔Python bridge for DeLFT integration |
| FastAPI + Uvicorn | latest | GROBID trainer HTTP service |

Python packages are installed into a local `.venv` directory at the project root (gitignored). The `VIRTUAL_ENV` environment variable is set automatically, which allows Gradle's `getJavaLibraryPath()` to locate the JEP native library without any manual configuration.

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
2. Create a `.venv` directory and install `tensorflow-macos`, `tensorflow-metal`, `delft`, `jep`, `fastapi`, and `uvicorn` via pip (~5–10 min)

Subsequent runs reuse the cached Nix packages and the existing `.venv`, so they start in a few seconds.

When ready, you will see:

```
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

## Updating the Python environment

If you need to add or upgrade Python packages (e.g. a newer version of DeLFT):

```bash
# Inside nix develop
pip install --upgrade delft
```

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

**`nix: command not found` after installation**

Open a new terminal, or source the Nix profile:

```bash
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

**`tensorflow-metal` import error or no GPU detected**

Verify that you are on macOS with Apple Silicon and that the Metal plug-in is installed:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
# Should list a MetalPerformanceShaders device
```

If the Metal device is missing, reinstall the plug-in:

```bash
pip install --force-reinstall tensorflow-metal
```

**DeLFT version incompatible with installed TensorFlow**

The `flake.nix` installs the latest `tensorflow-macos` and `delft>=0.3.3`. If you see import errors, upgrade DeLFT to a version that matches your TensorFlow:

```bash
pip install --upgrade delft
```

**JEP not found / DeLFT models fail to load**

The `VIRTUAL_ENV` variable must be set so Gradle can locate the JEP dylib. This is done automatically by the `nix develop` shellHook. If you are running outside of `nix develop`, set it manually:

```bash
export VIRTUAL_ENV="$(pwd)/.venv"
```

**`gradlew` permission denied**

```bash
chmod +x gradlew
```
