#!/usr/bin/env bash
# Push the current Chokepoint code + processed artifacts to a Hugging Face
# Space and trigger a Docker build.
#
# Usage:
#   ./deploy/hf_space/deploy.sh <hf_username> <space_name>
#
# Example:
#   ./deploy/hf_space/deploy.sh atharvahirulkar chokepoint
#
# Prerequisites:
#   1. huggingface-cli login (with a Write token)
#   2. A Space already created at https://huggingface.co/new-space
#      with the same name, SDK = Docker
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hf_username> <space_name>" >&2
  exit 1
fi

HF_USER="$1"
SPACE_NAME="$2"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
SPACE_GIT="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGING="${REPO_ROOT}/.hf_space_staging"

echo "[deploy] Repo root:     ${REPO_ROOT}"
echo "[deploy] Space target:  ${SPACE_URL}"
echo "[deploy] Staging dir:   ${STAGING}"

# Confirm hf login
if ! huggingface-cli whoami > /dev/null 2>&1; then
  echo "[deploy] Not logged into Hugging Face. Run: huggingface-cli login" >&2
  exit 1
fi
ACTUAL_USER="$(huggingface-cli whoami | head -n1)"
echo "[deploy] Logged in as:  ${ACTUAL_USER}"

# Required artifacts must exist
required=(
  "data/processed/supply_graph.pkl"
  "data/processed/vendor_scores.parquet"
  "data/processed/vendor_labels.parquet"
  "data/processed/event_validation.json"
  "models/artifacts/supervised.joblib"
  "models/artifacts/scaler.joblib"
  "models/artifacts/isoforest.joblib"
  "eval_report.json"
)
for f in "${required[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
    echo "[deploy] Missing required artifact: ${f}" >&2
    echo "[deploy] Run 'make all' first to generate processed files." >&2
    exit 1
  fi
done

# Fresh staging dir
rm -rf "${STAGING}"
mkdir -p "${STAGING}"
cd "${STAGING}"

# Clone the Space repo (must already exist on HF)
echo "[deploy] Cloning ${SPACE_GIT}"
git clone "${SPACE_GIT}" space_repo
cd space_repo

# Wipe any prior contents (keep .git)
find . -mindepth 1 -maxdepth 1 \
  -not -name '.git' -not -name '.gitattributes' -exec rm -rf {} +

# Copy code
cp -R "${REPO_ROOT}/pipeline"      ./pipeline
cp -R "${REPO_ROOT}/models"        ./models
cp -R "${REPO_ROOT}/api"           ./api
cp -R "${REPO_ROOT}/dashboard"     ./dashboard
cp    "${REPO_ROOT}/requirements.txt" ./requirements.txt

# Copy artifacts
mkdir -p ./data/processed ./data/synthetic ./models/artifacts
cp "${REPO_ROOT}/data/processed/supply_graph.pkl"        ./data/processed/
cp "${REPO_ROOT}/data/processed/vendor_scores.parquet"   ./data/processed/
cp "${REPO_ROOT}/data/processed/vendor_labels.parquet"   ./data/processed/
cp "${REPO_ROOT}/data/processed/event_validation.json"   ./data/processed/
cp "${REPO_ROOT}/data/synthetic/known_events.json"       ./data/synthetic/
cp "${REPO_ROOT}/eval_report.json"                       ./eval_report.json
cp -R "${REPO_ROOT}/models/artifacts/"*.joblib           ./models/artifacts/

# HF Space metadata + Dockerfile + start script
cp "${REPO_ROOT}/deploy/hf_space/README.md"   ./README.md
cp "${REPO_ROOT}/deploy/hf_space/Dockerfile"  ./Dockerfile
mkdir -p ./deploy/hf_space
cp "${REPO_ROOT}/deploy/hf_space/start.sh"    ./deploy/hf_space/start.sh

# Ensure data/ and __pycache__ aren't huge from anything else
find . -name '__pycache__' -prune -exec rm -rf {} +

# Configure Git LFS — HF requires LFS for any file > 10MiB. Must declare
# patterns BEFORE adding the actual files so they get checked in as LFS
# pointers, not the raw bytes.
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[deploy] git-lfs not installed. brew install git-lfs && git lfs install" >&2
  exit 1
fi
git lfs install --local
{
  echo "data/processed/supply_graph.pkl filter=lfs diff=lfs merge=lfs -text"
  echo "data/processed/*.parquet filter=lfs diff=lfs merge=lfs -text"
  echo "models/artifacts/*.joblib filter=lfs diff=lfs merge=lfs -text"
} > .gitattributes

# Track files explicitly (helps git lfs recognize them on first commit)
git lfs track "data/processed/supply_graph.pkl"
git lfs track "data/processed/*.parquet"
git lfs track "models/artifacts/*.joblib"

git add .gitattributes
git add -A
git config user.email "${ACTUAL_USER}@users.noreply.huggingface.co"
git config user.name  "${ACTUAL_USER}"
git commit -m "Deploy Chokepoint $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[deploy] Pushing to ${SPACE_GIT}"
git push

echo
echo "[deploy] Done. Build is now starting on HF."
echo "[deploy] Open ${SPACE_URL} to watch the build log and see the live demo."
