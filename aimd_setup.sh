#!/usr/bin/env bash
# ============================================================================
# aimd_setup.sh - AIMD 一键设置脚本
# 用法: aimd_setup.sh [recipe.yaml]
# 功能:
#   1. 调用 make_incar_aimd.py 生成 INCAR
#   2. 检查 POSCAR, POTCAR, KPOINTS 是否存在
#   3. 提示下一步运行命令
# ============================================================================
set -euo pipefail

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配方文件
RECIPE="${1:-recipe.yaml}"

echo "============================================"
echo "[aimd_setup] AIMD 一键设置"
echo "============================================"
echo "当前目录: $(pwd)"
echo "配方文件: $RECIPE"
echo "============================================"

# ---------------------- 检查配方文件 ----------------------
if [[ ! -f "$RECIPE" ]]; then
    echo "[ERROR] 配方文件不存在: $RECIPE"
    echo "[INFO] 请将 recipe.yaml 复制到当前目录或指定路径"
    exit 1
fi
echo "[OK] 配方文件存在: $RECIPE"

# ---------------------- 生成 INCAR ----------------------
echo ""
echo ">>> 生成 AIMD INCAR..."
python3 "$SCRIPT_DIR/make_incar_aimd.py" --recipe "$RECIPE" --out INCAR

if [[ ! -f INCAR ]]; then
    echo "[ERROR] INCAR 生成失败"
    exit 1
fi
echo "[OK] INCAR 已生成"

# ---------------------- 检查必需文件 ----------------------
echo ""
echo ">>> 检查 VASP 输入文件..."

missing=0

if [[ ! -f POSCAR ]]; then
    echo "[ERROR] 缺少 POSCAR"
    missing=1
else
    echo "[OK] POSCAR 存在"
fi

if [[ ! -f POTCAR ]]; then
    echo "[ERROR] 缺少 POTCAR"
    missing=1
else
    echo "[OK] POTCAR 存在"
fi

# KPOINTS 检查（非必需，可用 KSPACING）
if [[ ! -f KPOINTS ]]; then
    echo "[WARN] KPOINTS 不存在"
    echo "[INFO] 请确保 INCAR 中设置了 KSPACING，或手动创建 KPOINTS"
    echo "[INFO] 凝胶体系推荐: KSPACING = 0.5 或 Gamma-only"
    
    # 检查 INCAR 中是否有 KSPACING
    if grep -qi "KSPACING" INCAR 2>/dev/null; then
        echo "[OK] INCAR 中已设置 KSPACING"
    else
        echo "[WARN] INCAR 中未找到 KSPACING，请手动添加或创建 KPOINTS"
    echo ""
        echo "建议 Gamma-only KPOINTS 内容:"
        echo "---"
        echo "Automatic mesh"
        echo "0"
        echo "Gamma"
        echo "1 1 1"
        echo "0 0 0"
        echo "---"
    fi
else
    echo "[OK] KPOINTS 存在"
fi

if [[ $missing -eq 1 ]]; then
    echo ""
    echo "[WARN] 部分文件缺失，请补齐后再运行 VASP"
fi

# ---------------------- 推荐命令 ----------------------
echo ""
echo "============================================"
echo "[aimd_setup] 设置完成！"
echo "============================================"
echo ""
echo "下一步操作:"
echo ""
echo "  1. 确认所有输入文件就绪 (INCAR, POSCAR, POTCAR, KPOINTS)"
echo ""
echo "  2. 运行 VASP AIMD:"
echo "     NP=16 EXE=vasp_std run_vasp.sh"
echo ""
echo "  3. 监控运行状态:"
echo "     aimd_watch.sh"
echo ""
echo "  4. 检查计算状态:"
echo "     check_vasp.sh"
echo ""
echo "  5. 后处理:"
echo "     python3 aimd_post.py"
echo "     python3 aimd_msd.py --specie Li --dt_fs 1.0"
echo ""
echo "============================================"
