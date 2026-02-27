#!/usr/bin/env bash
# ============================================================================
# vasp_env.sh - VASP 运行环境配置脚本 (oneAPI + Intel MPI)
# 用法: source vasp_env.sh
# ============================================================================

# ---------------------- 保存原始 shell 选项 ----------------------
_VASP_ENV_ORIGINAL_OPTS=$(set +o)
_VASP_ENV_ORIGINAL_SHOPT=$(shopt -p 2>/dev/null || true)

# 临时启用严格模式（仅在此脚本内）
set -euo pipefail

# 用于在脚本结束时恢复 shell 选项
_vasp_env_cleanup() {
    eval "$_VASP_ENV_ORIGINAL_OPTS" 2>/dev/null || true
    eval "$_VASP_ENV_ORIGINAL_SHOPT" 2>/dev/null || true
}
trap '_vasp_env_cleanup' RETURN

# ---------------------- VASP 可执行文件路径 ----------------------
export VASP_BIN="${VASP_BIN:-/home/edu/soft/vasp.6.4.3/bin}"
export PATH="$VASP_BIN:$PATH"

# ---------------------- VASP_PP_PATH (POTCAR 路径) ----------------------
# 用于自动拼接 POTCAR 的脚本
export VASP_PP_PATH="${VASP_PP_PATH:-}"
if [[ -z "$VASP_PP_PATH" ]]; then
    echo "[vasp_env] VASP_PP_PATH 未设置 (自动拼 POTCAR 的脚本可能受影响)"
fi

# ---------------------- oneAPI 环境加载 ----------------------
ONEAPI_SETVARS="/home/edu/soft/intel/oneapi/setvars.sh"
ONEAPI_LOG="/tmp/oneapi_setvars_$(id -u).log"

if [[ -f "$ONEAPI_SETVARS" ]]; then
    echo "[vasp_env] 加载 oneAPI 环境..."
    
    # 捕获 source 的输出和返回码
    if ! source "$ONEAPI_SETVARS" --force > "$ONEAPI_LOG" 2>&1; then
        echo "[ERROR] oneAPI setvars.sh 加载失败！"
        echo "[ERROR] 日志文件: $ONEAPI_LOG"
        echo "[ERROR] 前 30 行日志:"
        head -30 "$ONEAPI_LOG" | sed 's/^/    /'
        echo ""
        echo "[ERROR] 请检查 oneAPI 安装是否完整"
        return 1
    fi
    echo "[vasp_env] oneAPI 加载成功 (日志: $ONEAPI_LOG)"
else
    echo "[ERROR] oneAPI setvars.sh 不存在: $ONEAPI_SETVARS"
    return 1
fi

# ---------------------- 线程控制（默认纯 MPI）----------------------
# 避免 OpenMP/MKL 过度线程导致性能下降
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-512M}"

# ---------------------- Intel MPI 参数（不强制覆盖）----------------------
# 仅尊重用户外部设置，不强制写死
if [[ -n "${I_MPI_ADJUST_REDUCE:-}" ]]; then
    echo "[vasp_env] I_MPI_ADJUST_REDUCE=${I_MPI_ADJUST_REDUCE} (用户设置)"
else
    echo "[vasp_env] I_MPI_ADJUST_REDUCE 未设置 (使用 MPI 默认值)"
fi

if [[ -n "${MPIR_CVAR_COLL_ALIAS_CHECK:-}" ]]; then
    echo "[vasp_env] MPIR_CVAR_COLL_ALIAS_CHECK=${MPIR_CVAR_COLL_ALIAS_CHECK}"
fi

# ---------------------- 堆栈限制解除 ----------------------
ulimit -s unlimited 2>/dev/null || echo "[vasp_env] 无法设置 ulimit -s unlimited"

# ---------------------- 自检：VASP 可执行文件 ----------------------
echo ""
echo "[vasp_env] 自检 VASP 可执行文件..."

_vasp_check_failed=0

for exe in vasp_std vasp_gam vasp_ncl; do
    exe_path="$VASP_BIN/$exe"
    if [[ -x "$exe_path" ]]; then
        echo "    [OK] $exe"
    else
        echo "    [WARN] $exe 不存在或无执行权限: $exe_path"
    fi
done

# ---------------------- 自检：mpirun ----------------------
if command -v mpirun &>/dev/null; then
    mpirun_path=$(command -v mpirun)
    echo "    [OK] mpirun: $mpirun_path"
else
    echo "    [ERROR] mpirun 未找到！"
    _vasp_check_failed=1
fi

# ---------------------- 自检：vasp_std 动态库依赖 ----------------------
if [[ -x "$VASP_BIN/vasp_std" ]]; then
    echo ""
    echo "[vasp_env] 检查 vasp_std 动态库依赖..."
    
    _ldd_output=$(ldd "$VASP_BIN/vasp_std" 2>&1 || true)
    _missing_libs=$(echo "$_ldd_output" | grep -i "not found" || true)
    
    if [[ -n "$_missing_libs" ]]; then
        echo "[ERROR] vasp_std 存在缺失的动态库："
        echo "$_missing_libs" | sed 's/^/    /'
        echo ""
        echo "[ERROR] 请检查 oneAPI 环境是否正确加载"
        _vasp_check_failed=1
    else
        echo "    [OK] 所有动态库依赖已满足"
    fi
fi

if [[ $_vasp_check_failed -eq 1 ]]; then
    echo ""
    echo "[ERROR] 环境自检失败，请修正上述问题"
    return 1
fi

# ---------------------- 环境确认输出 ----------------------
echo ""
echo "[vasp_env] ============================================"
echo "[vasp_env] VASP_BIN=$VASP_BIN"
echo "[vasp_env] VASP_PP_PATH=${VASP_PP_PATH:-未设置}"
echo "[vasp_env] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[vasp_env] MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "[vasp_env] ============================================"
echo "[vasp_env] 环境加载成功！"
