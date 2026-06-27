#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/eci-ocr}"
RUNTIME_DIR="${RUNTIME_DIR:-$APP_DIR/.runtime}"
MAMBA_BIN="$RUNTIME_DIR/bin/micromamba"
ENV_PREFIX="$RUNTIME_DIR/envs/eci-ocr"

cd "$APP_DIR"

if command -v dnf >/dev/null 2>&1; then
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y \
        tar \
        bzip2 \
        gzip \
        xz \
        gcc \
        gcc-c++ \
        make \
        libgomp
elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y \
        tar \
        bzip2 \
        gzip \
        xz \
        gcc \
        gcc-c++ \
        make \
        libgomp
fi
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y \
    curl tar bzip2 gzip xz \
    gcc gcc-c++ make \
    libgomp
else
  echo "This bootstrap script expects Amazon Linux with dnf or yum." >&2
  exit 2
fi

if [ ! -x "$MAMBA_BIN" ]; then
  mkdir -p "$RUNTIME_DIR/bin" "$RUNTIME_DIR/download"
  curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "$RUNTIME_DIR/download/micromamba.tar.bz2"
  tar -xjf "$RUNTIME_DIR/download/micromamba.tar.bz2" -C "$RUNTIME_DIR/download" bin/micromamba
  mv "$RUNTIME_DIR/download/bin/micromamba" "$MAMBA_BIN"
  chmod +x "$MAMBA_BIN"
fi

if [ -d "$ENV_PREFIX" ]; then
  "$MAMBA_BIN" install -y -p "$ENV_PREFIX" -c conda-forge \
    python=3.11 \
    pip \
    tesseract \
    poppler
else
  "$MAMBA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge \
    python=3.11 \
    pip \
    tesseract \
    poppler
fi

export PATH="$ENV_PREFIX/bin:$PATH"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --no-cache-dir -r requirements.txt

tesseract --version
pdftoppm -v

mkdir -p inputs outputs logs

echo "EC2 bootstrap complete in $APP_DIR"
