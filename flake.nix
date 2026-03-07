{
  description = "GROBID development environment for Apple Silicon (M1/M2/M3/M4)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
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

            # Clear any Nix-injected PYTHONPATH so pip's build subprocesses only
            # see the venv's own packages, not Nix-store copies of setuptools etc.
            unset PYTHONPATH

            # ── Python venv (tensorflow-metal lives here, not in Nix store) ──
            # tensorflow-macos / tensorflow-metal / JEP are PyPI-only packages
            # not available in nixpkgs. We keep them in a local .venv so
            # `nix develop` works without internet on subsequent runs once the
            # venv has been built.
            VENV_DIR="$(pwd)/.venv"
            # Sentinel file written only after a fully successful install.
            # If absent (first run or previous failure), the install is retried.
            # On failure the venv is cleaned up — no manual rm needed.
            SENTINEL="$VENV_DIR/.install-complete"
            _GROBID_SETUP_OK=true

            if [ ! -f "$SENTINEL" ]; then
              echo "==> Creating Python venv (this may take several minutes)..."
              rm -rf "$VENV_DIR"
              python3 -m venv "$VENV_DIR"

              # Bootstrap pip toolchain inside the venv.
              "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel --quiet

              echo "==> Installing tensorflow-macos, tensorflow-metal, JEP, FastAPI..."
              # --no-build-isolation: uses the venv's own setuptools instead of
              # pip's temporary isolated build environment. Required for JEP,
              # which compiles a JNI bridge against the JDK headers in JAVA_HOME.
              #
              # DeLFT (delft==0.3.3) is NOT installed here — it pins
              # tensorflow==2.9.3, tensorflow-addons==0.19.0 (discontinued),
              # and torch==1.10.1, none of which have macOS ARM64 binary wheels
              # and all of which predate tensorflow-macos availability.
              # To use DeLFT, clone it from GitHub and install from source with
              # manually relaxed requirements (see doc/use-nix-for-apple-silicon.md).
              # The tensorflow-macos + tensorflow-metal installed here still
              # provide Metal GPU support for custom DeLFT setups.
              if "$VENV_DIR/bin/pip" install \
                  --no-build-isolation \
                  --quiet \
                  tensorflow-macos \
                  tensorflow-metal \
                  jep \
                  fastapi \
                  "uvicorn[standard]" \
                  python-multipart \
                  websockets \
                  httpx; then
                touch "$SENTINEL"
                echo "==> Python environment ready."
              else
                rm -rf "$VENV_DIR"
                echo ""
                echo "ERROR: pip install failed. Re-enter 'nix develop' to retry."
                _GROBID_SETUP_OK=false
              fi
            fi

            if $_GROBID_SETUP_OK; then
              # Expose venv — Gradle's getJavaLibraryPath() in build.gradle
              # auto-detects VIRTUAL_ENV and appends the JEP .dylib directory.
              export VIRTUAL_ENV="$VENV_DIR"
              export PATH="$VIRTUAL_ENV/bin:$PATH"

              echo ""
              echo "╔══════════════════════════════════════════════════════════════╗"
              echo "║  GROBID training environment  (Apple Silicon / Metal GPU)    ║"
              echo "╠══════════════════════════════════════════════════════════════╣"
              echo "║  Build trainer JAR:                                          ║"
              echo "║    ./gradlew :grobid-trainer:shadowJar --no-daemon           ║"
              echo "║                                                              ║"
              echo "║  CRF training (fast, CPU, default):                          ║"
              echo "║    ./gradlew train_date                                      ║"
              echo "║    ./gradlew train_header                                    ║"
              echo "║                                                              ║"
              echo "║  HTTP trainer service (port 8072):                           ║"
              echo "║    python grobid-home/scripts/trainer_service.py             ║"
              echo "║    → API docs: http://localhost:8072/docs                    ║"
              echo "╚══════════════════════════════════════════════════════════════╝"
              echo ""
            fi
          '';
        };
      }
    );
}
