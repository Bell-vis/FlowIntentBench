# Source this file: source scripts/activate_benchmark.sh
# This activates the project environment in the current shell only. It does
# not change Conda's default environment.
# Resolve Conda from the caller's PATH so the project does not depend on one
# user's Miniforge/Miniconda installation prefix.
if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/miniforge3/condabin/conda" "$HOME/miniconda3/condabin/conda" "$HOME/anaconda3/condabin/conda"; do
    if [ -x "$candidate" ]; then
      export PATH="$(dirname "$candidate"):$PATH"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found; install Conda or set PATH before sourcing this file" >&2
  return 1
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${FLOWINTENT_CONDA_ENV:-benchmark}" || return 1
unset PYTHONPATH
