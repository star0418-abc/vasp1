#!/usr/bin/env bash
# ============================================================================
# clean_vasp.sh - 安全清理 VASP 大文件与可再生文件
# 用法: 
#   clean_vasp.sh          # 执行清理
#   DRYRUN=1 clean_vasp.sh # 预览模式，只打印不删除
# ============================================================================
set -euo pipefail

# ---------------------- 配置 ----------------------
# 保护文件列表（绝不删除）
PROTECTED=(
    INCAR
    POSCAR
    POTCAR
    KPOINTS
    OUTCAR
    OSZICAR
    CONTCAR
    XDATCAR
)

# 默认删除文件列表（大文件和可再生文件）
TO_DELETE=(
    WAVECAR
    CHGCAR
    CHG
    vasprun.xml
    vasp.out
    PROCAR
    EIGENVAL
    DOSCAR
    IBZKPT
    PCDAT
    REPORT
    LOCPOT
    ELFCAR
    AECCAR0
    AECCAR1
    AECCAR2
    PARCHG
)

# DRYRUN 模式
DRYRUN="${DRYRUN:-0}"

echo "============================================"
echo "[clean_vasp] 当前目录: $(pwd)"
echo "[clean_vasp] 模式: $(if [[ $DRYRUN == 1 ]]; then echo '预览 (DRYRUN)'; else echo '执行删除'; fi)"
echo "============================================"

# ---------------------- 显示保护文件 ----------------------
echo ""
echo ">>> 保护文件（不会删除）:"
for f in "${PROTECTED[@]}"; do
    if [[ -f "$f" ]]; then
        size=$(du -h "$f" 2>/dev/null | cut -f1)
        echo "    [保护] $f ($size)"
    fi
done

# ---------------------- 查找待删除文件 ----------------------
echo ""
echo ">>> 待删除文件:"
files_to_delete=()
total_size=0

for f in "${TO_DELETE[@]}"; do
    if [[ -f "$f" ]]; then
        size_bytes=$(stat -c%s "$f" 2>/dev/null || echo 0)
        size_human=$(du -h "$f" 2>/dev/null | cut -f1)
        echo "    [删除] $f ($size_human)"
        files_to_delete+=("$f")
        total_size=$((total_size + size_bytes))
    fi
done

# 检查是否有文件需要删除
if [[ ${#files_to_delete[@]} -eq 0 ]]; then
    echo "    [INFO] 没有找到需要删除的文件"
    echo ""
    echo "[clean_vasp] 完成。"
    exit 0
fi

# 计算总大小
total_size_mb=$((total_size / 1024 / 1024))
echo ""
echo ">>> 预计释放空间: ${total_size_mb} MB (${#files_to_delete[@]} 个文件)"

# ---------------------- 执行删除 ----------------------
echo ""
if [[ $DRYRUN == 1 ]]; then
    echo "[DRYRUN] 预览模式，未执行实际删除"
    echo "[DRYRUN] 若要执行删除，请运行: clean_vasp.sh"
else
    echo ">>> 开始删除..."
    for f in "${files_to_delete[@]}"; do
        rm -f "$f"
        echo "    [已删除] $f"
    done
    echo ""
    echo "[OK] 清理完成，释放约 ${total_size_mb} MB"
fi

echo ""
echo "============================================"
echo "[clean_vasp] 完成。"

