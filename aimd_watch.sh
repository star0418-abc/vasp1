#!/usr/bin/env bash
# ============================================================================
# aimd_watch.sh - AIMD 实时监控脚本
# 用法: aimd_watch.sh [OUTCAR] [OSZICAR]
# 每 5 秒刷新显示 OSZICAR 最后 5 行和 OUTCAR 最新温度信息
# Ctrl+C 退出
# ============================================================================
set -euo pipefail

OUTCAR="${1:-OUTCAR}"
OSZICAR="${2:-OSZICAR}"
INTERVAL=5

echo "[aimd_watch] 监控开始"
echo "[aimd_watch] OUTCAR: $OUTCAR"
echo "[aimd_watch] OSZICAR: $OSZICAR"
echo "[aimd_watch] 刷新间隔: ${INTERVAL}s"
echo "[aimd_watch] 按 Ctrl+C 退出"
echo ""

# 捕获 Ctrl+C 优雅退出
trap 'echo ""; echo "[aimd_watch] 监控已停止。"; exit 0' INT

while true; do
    clear
    echo "============================================"
    echo "[aimd_watch] $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[aimd_watch] 目录: $(pwd)"
    echo "============================================"
    
    # ---------------------- OSZICAR 最后 5 行 ----------------------
    echo ""
    echo ">>> OSZICAR 最后 5 行 (离子步/电子步):"
    if [[ -f "$OSZICAR" ]]; then
        tail -5 "$OSZICAR" | sed 's/^/    /'
        # 显示当前离子步数
        nsteps=$(grep -c "^[[:space:]]*[0-9]* F=" "$OSZICAR" 2>/dev/null || echo "0")
        echo "    ----------------------------------------"
        echo "    总离子步数: $nsteps"
    else
        echo "    [INFO] OSZICAR 尚不存在"
    fi
    
    # ---------------------- OUTCAR 温度信息 ----------------------
    echo ""
    echo ">>> OUTCAR 最新温度信息:"
    if [[ -f "$OUTCAR" ]]; then
        # 提取最后一条 temperature 行（AIMD 特有）
        temp_line=$(grep -i "temperature" "$OUTCAR" | grep -i "kinetic" | tail -1 || true)
        if [[ -n "$temp_line" ]]; then
            echo "    $temp_line"
        else
            # 尝试其他格式的温度
            temp_line=$(grep "EKIN_LAT" "$OUTCAR" | tail -1 || true)
            if [[ -n "$temp_line" ]]; then
                echo "    $temp_line"
            else
                echo "    [INFO] 未找到温度信息 (非 AIMD 或尚未开始)"
            fi
        fi
        
        # 显示当前能量
        toten=$(grep "free  energy   TOTEN" "$OUTCAR" | tail -1 || true)
        if [[ -n "$toten" ]]; then
            echo ""
            echo ">>> 最新能量:"
            echo "    $toten"
        fi
    else
        echo "    [INFO] OUTCAR 尚不存在"
    fi
    
    # ---------------------- 完成状态检查 ----------------------
    echo ""
    if [[ -f "$OUTCAR" ]] && grep -q "General timing and accounting informations" "$OUTCAR" 2>/dev/null; then
        echo ">>> [OK] 计算已完成!"
        echo ""
        echo "[aimd_watch] 检测到计算完成，5秒后自动退出..."
        sleep 5
        exit 0
    fi
    
    echo "============================================"
    echo "[下次刷新: ${INTERVAL}s 后]"
    
    sleep "$INTERVAL"
done

