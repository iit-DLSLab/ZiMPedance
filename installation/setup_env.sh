#!/usr/bin/env bash
# Create the conda environment, build acados and point both at this checkout.
#
#   bash installation/setup_env.sh                   # resolve from environment.yml
#   bash installation/setup_env.sh --lock            # exact versions from environment.lock.yml (linux-64)
#   bash installation/setup_env.sh --name my_env     # environment name (default: zimpedance)
#
# Prerequisites: conda or mamba, a C compiler, and the acados submodule
# (git submodule update --init --recursive). Nothing is pip-installed from the repo: the activation
# hook puts quadruped_pympc, gym_quadruped and acados_template on PYTHONPATH.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENV_NAME=zimpedance
SPEC="$HERE/environment.yml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lock) SPEC="$HERE/environment.lock.yml"; shift ;;
        --name) ENV_NAME="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done

ACADOS="$REPO/quadruped_pympc/acados"
TERA_VERSION=0.0.34
[[ -f "$ACADOS/external/blasfeo/CMakeLists.txt" ]] || {
    echo "acados submodule missing: run 'git submodule update --init --recursive'" >&2; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env $ENV_NAME already exists; remove it first (conda env remove -n $ENV_NAME)" >&2
    exit 1
fi
CONDA_CMD=conda; command -v mamba > /dev/null && CONDA_CMD=mamba
$CONDA_CMD env create -n "$ENV_NAME" -f "$SPEC"
conda activate "$ENV_NAME"
set -euo pipefail  # conda activate resets the shell options

# acados: default build (HPIPM only), installed into the submodule's lib/ and include/
if [[ ! -f "$ACADOS/lib/libacados.so" ]]; then
    # the policy minimum lets CMake >= 4 configure the bundled blasfeo/hpipm (they declare CMake < 3.5)
    cmake -S "$ACADOS" -B "$ACADOS/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build "$ACADOS/build" --target install -j "$(nproc)"
fi
[[ -f "$ACADOS/lib/libacados.so" ]] || { echo "acados build failed" >&2; exit 1; }
# tera renderer, used by acados to generate the solver code (acados would otherwise ask to download it)
if [[ ! -x "$ACADOS/bin/t_renderer" ]]; then
    mkdir -p "$ACADOS/bin"
    curl -fL -o "$ACADOS/bin/t_renderer" \
        "https://github.com/acados/tera_renderer/releases/download/v$TERA_VERSION/t_renderer-v$TERA_VERSION-linux"
    chmod +x "$ACADOS/bin/t_renderer"
fi

PREFIX="$CONDA_PREFIX"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat > "$PREFIX/etc/conda/activate.d/zimpedance.sh" <<EOF
export _ZIMP_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export _ZIMP_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export ACADOS_SOURCE_DIR="$ACADOS"
export ACADOS_PATH="$ACADOS"
export LD_LIBRARY_PATH="$ACADOS/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$ACADOS/interfaces/acados_template\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
cat > "$PREFIX/etc/conda/deactivate.d/zimpedance.sh" <<'EOF'
export PYTHONPATH="$_ZIMP_OLD_PYTHONPATH"
export LD_LIBRARY_PATH="$_ZIMP_OLD_LD_LIBRARY_PATH"
unset ACADOS_SOURCE_DIR ACADOS_PATH _ZIMP_OLD_PYTHONPATH _ZIMP_OLD_LD_LIBRARY_PATH
EOF
echo "Created $ENV_NAME. Activate with: conda activate $ENV_NAME"
