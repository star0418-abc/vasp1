#!/usr/bin/env bash
# ============================================================================
# check_vasp.sh - 检查 VASP 计算状态
# 用法: check_vasp.sh [OUTCAR路径]
# ============================================================================
set -euo pipefail

OUTCAR="${1:-OUTCAR}"
OSZICAR="${OSZICAR:-OSZICAR}"

echo "============================================"
echo "[check_vasp] 检查目录: $(pwd)"
echo "[check_vasp] OUTCAR: $OUTCAR"
echo "============================================"

# ---------------------- (a) OUTCAR 存在与完成判据 ----------------------
if [[ ! -f "$OUTCAR" ]]; then
    echo "[ERROR] OUTCAR 文件不存在: $OUTCAR"
    exit 1
fi

echo ""
echo ">>> (a) 计算完成状态:"
if grep -q "General timing and accounting informations for this job" "$OUTCAR"; then
    echo "    [OK] 计算已正常完成 (找到 timing block)"
    # 提取总时间
    total_time=$(grep "Total CPU time used" "$OUTCAR" | tail -1 || true)
    if [[ -n "$total_time" ]]; then
        echo "    $total_time"
    fi
else
    echo "    [WARN] 计算未完成或中断 (未找到 timing block)"
fi

# ---------------------- (b) OSZICAR 最后 10 行 ----------------------
echo ""
echo ">>> (b) OSZICAR 最后 10 行:"
if [[ -f "$OSZICAR" ]]; then
    tail -10 "$OSZICAR" | sed 's/^/    /'
else
    echo "    [INFO] OSZICAR 不存在"
fi

# ---------------------- (c) 最后一条 TOTEN ----------------------
echo ""
echo ">>> (c) 最后一条 free energy TOTEN:"
toten=$(grep "free  energy   TOTEN" "$OUTCAR" | tail -1 || true)
if [[ -n "$toten" ]]; then
    echo "    $toten"
else
    echo "    [INFO] 未找到 TOTEN"
fi

# ---------------------- (d) 最后一条 E-fermi ----------------------
echo ""
echo ">>> (d) 最后一条 E-fermi:"
efermi=$(grep "E-fermi" "$OUTCAR" | tail -1 || true)
if [[ -n "$efermi" ]]; then
    echo "    $efermi"
else
    echo "    [INFO] 未找到 E-fermi"
fi

echo ""
echo "============================================"
echo "[check_vasp] 检查完成。"

