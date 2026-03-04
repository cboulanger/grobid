{
  description = "GROBID development environment for Apple Silicon (M1/M2/M3/M4)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # Use OpenJDK 21 to match the Dockerfiles. On Apple Silicon this will
        # be the native aarch64 build — no Rosetta emulation.
        jdk = pkgs.openjdk21;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "grobid-training";

          buildInputs = [
            jdk
            pkgs.python311
            pkgs.git
            pkgs.unzip
          ];

          shellHook = ''
            # ── Java ──────────────────────────────────────────────────────────
            export JAVA_HOME="${jdk}"

            # ── Python venv (tensorflow-metal lives here, not in Nix store) ──
            # tensorflow-macos / tensorflow-metal / DeLFT are Apple-specific or
            # PyPI-only packages not available in nixpkgs. We keep them in a
            # local .venv so `nix develop` still works without internet on
            # subsequent runs once the venv exists.
            VENV_DIR="$(pwd)/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "==> First run: creating Python venv (this may take several minutes)..."
              python3 -m venv "$VENV_DIR"
              "$VENV_DIR/bin/pip" install --upgrade pip --quiet
              echo "==> Installing tensorflow-macos, tensorflow-metal, DeLFT, JEP, FastAPI..."
              # tensorflow-macos + tensorflow-metal  → GPU via Apple Metal on M-series
              # delft                               → GROBID deep-learning models
              # jep                                 → Java↔Python bridge (DeLFT integration)
              # fastapi + uvicorn                   → trainer HTTP service
              #
              # NOTE: Pin versions here if reproducibility matters.
              # delft==0.3.3 was used in the Docker images; newer versions
              # support more recent TF releases. Adjust if you see import errors.
              "$VENV_DIR/bin/pip" install --quiet \
                tensorflow-macos \
                tensorflow-metal \
                "delft>=0.3.3" \
                jep \
                fastapi \
                "uvicorn[standard]"
              echo "==> Python environment ready."
            fi

            # Expose venv — Gradle's getJavaLibraryPath() in build.gradle
            # auto-detects VIRTUAL_ENV and appends the JEP .dylib directory.
            export VIRTUAL_ENV="$VENV_DIR"
            export PATH="$VIRTUAL_ENV/bin:$PATH"

            echo ""
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║  GROBID training environment  (Apple Silicon / Metal GPU)   ║"
            echo "╠══════════════════════════════════════════════════════════════╣"
            echo "║  Build trainer JAR:                                          ║"
            echo "║    ./gradlew :grobid-trainer:shadowJar --no-daemon           ║"
            echo "║                                                              ║"
            echo "║  CRF training (fast, CPU):                                  ║"
            echo "║    ./gradlew train_date                                      ║"
            echo "║    ./gradlew train_header                                    ║"
            echo "║                                                              ║"
            echo "║  HTTP trainer service (port 8072):                          ║"
            echo "║    python grobid-home/scripts/trainer_service.py            ║"
            echo "║    → API docs: http://localhost:8072/docs                   ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""
          '';
        };
      }
    );
}
