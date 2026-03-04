#!/usr/bin/env bash
# ============================================================================
# run_vasp.sh - VASP 运行脚本（含自动备份、日志、续算支持）v2.0
# ============================================================================
# 用法: NP=16 EXE=vasp_std run_vasp.sh
#
# 环境变量:
#   NP           - MPI 进程数 (默认 8)
#   EXE          - 可执行文件 (vasp_std/vasp_gam/vasp_ncl, 默认 vasp_std)
#   OUT          - stdout 文件名 (默认 vasp.out)
#   ERR          - stderr 文件名 (默认 vasp.err)
#   RESUME       - 续算模式: 1=自动 cp CONTCAR->POSCAR (默认 0)
#   STRICT_NP    - 严格核数: 1=NP 超限时报错退出 (默认 0=自动下调)
#   RESERVE_CORES - WSL 预留核数 (默认 2)
#   MIN_FREE_GB  - 最小磁盘空间 GB (默认 20)
#   FORCE_DISK   - 忽略磁盘检查: 1=强制继续 (默认 0, 磁盘检查默认启用)
#
# === v2.0 新增 (AIMD 续算/HPC 安全) ===
#   KEEP_RESTART     - 保留 WAVECAR/CHGCAR/CHG: 1=保留, 0=归档 (RESUME=1 时默认 1)
#   KEEP_TRAJECTORY  - 保留 XDATCAR 连续性: 1=保留, 0=归档 (RESUME=1 时默认 1)
#   FORCE_RESTART    - 允许无 WAVECAR/CHGCAR 时使用 ISTART/ICHARG (默认 0)
#   THREAD_GUARD     - 线程安全: 1=强制 OMP/MKL 限制 (默认 1)
#   STRICT_INPUT     - 严格输入: 1=K点缺失时报错 (默认 0=仅警告)
#   FORCE_OMP_BIND   - 在 WSL 下强制 OMP 绑定策略: 1=启用 close/cores (默认 0)
#   KEEP_APPEND_LOGS - RESUME 时保留 OUTCAR/OSZICAR 追加: 1=保留不轮转 (默认 0)
#   LOG_ROTATE_GB    - RESUME 时 OUTCAR/OSZICAR 轮转阈值 GB (默认 2)
# ============================================================================
set -uo pipefail  # 使用显式返回码判断，避免 set -e 隐式中断控制流

TMP_PATHS=()
register_tmp_path() {
    local path="$1"
    [[ -n "$path" ]] || return 0
    TMP_PATHS+=("$path")
}
cleanup_tmp_paths() {
    local path
    for path in "${TMP_PATHS[@]-}"; do
        [[ -n "$path" ]] || continue
        if [[ -d "$path" ]]; then
            rm -rf -- "$path" 2>/dev/null || true
        else
            rm -f -- "$path" 2>/dev/null || true
        fi
    done
}
handle_signal() {
    local sig="$1"
    local code="$2"
    echo "[ABORT] 收到 $sig，清理临时文件后退出。" >&2
    cleanup_tmp_paths
    exit "$code"
}
trap cleanup_tmp_paths EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

# ---------------------- 加载 VASP 环境 ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! source "$SCRIPT_DIR/vasp_env.sh"; then
    echo "[ERROR] VASP 环境加载失败"
    exit 1
fi
if [[ -z "${VASP_BIN:-}" ]]; then
    echo "[ERROR] vasp_env.sh 未设置 VASP_BIN，请检查 $SCRIPT_DIR/vasp_env.sh"
    exit 1
fi
if [[ ! -d "$VASP_BIN" ]]; then
    echo "[ERROR] VASP_BIN 目录不存在: $VASP_BIN"
    exit 1
fi

# ---------------------- 参数设置 ----------------------
NP="${NP:-8}"
EXE="${EXE:-vasp_std}"
OUT="${OUT:-vasp.out}"
ERR="${ERR:-vasp.err}"
RESUME="${RESUME:-0}"
STRICT_NP="${STRICT_NP:-0}"
RESERVE_CORES="${RESERVE_CORES:-2}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
FORCE_DISK="${FORCE_DISK:-0}"  # 默认启用磁盘检查
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

# === v2.0 新增参数 ===
KEEP_RESTART="${KEEP_RESTART:-}"       # 空=根据 RESUME 自动决定
KEEP_TRAJECTORY="${KEEP_TRAJECTORY:-}" # 空=根据 RESUME 自动决定
FORCE_RESTART="${FORCE_RESTART:-0}"    # 跳过 WAVECAR/CHGCAR 检查
THREAD_GUARD="${THREAD_GUARD:-1}"      # 默认启用线程保护
STRICT_INPUT="${STRICT_INPUT:-0}"      # 默认仅警告
FORCE_OMP_BIND="${FORCE_OMP_BIND:-0}"  # WSL 下是否强制 OMP 绑定
KEEP_APPEND_LOGS="${KEEP_APPEND_LOGS:-0}"  # RESUME 时保留追加日志
LOG_ROTATE_GB="${LOG_ROTATE_GB:-2}"    # RESUME 日志轮转阈值 (GB)

if ! [[ "$LOG_ROTATE_GB" =~ ^[1-9][0-9]*$ ]]; then
    echo "[WARN] LOG_ROTATE_GB='$LOG_ROTATE_GB' 非法，回退到 2 GB"
    LOG_ROTATE_GB=2
fi
LOG_ROTATE_BYTES=$((LOG_ROTATE_GB * 1024 * 1024 * 1024))
RESUME_CONTINUATION_READY=0

TMP_ROOT=$(mktemp -d "/tmp/run_vasp.${TIMESTAMP}.XXXXXX" 2>/dev/null || true)
if [[ -z "$TMP_ROOT" || ! -d "$TMP_ROOT" ]]; then
    echo "[ERROR] 无法创建临时目录用于日志/轨迹保护"
    exit 1
fi
register_tmp_path "$TMP_ROOT"

is_wsl() {
    grep -qiE "(microsoft|wsl)" /proc/version /proc/sys/kernel/osrelease 2>/dev/null
}
IS_WSL=0
if is_wsl; then
    IS_WSL=1
fi

# ---------------------- KEEP_RESTART/KEEP_TRAJECTORY 自动默认 ----------------------
# RESUME=1 时默认保留，RESUME=0 时默认归档
if [[ -z "$KEEP_RESTART" ]]; then
    if [[ $RESUME -eq 1 ]]; then
        KEEP_RESTART=1
    else
        KEEP_RESTART=0
    fi
fi

if [[ -z "$KEEP_TRAJECTORY" ]]; then
    if [[ $RESUME -eq 1 ]]; then
        KEEP_TRAJECTORY=1
    else
        KEEP_TRAJECTORY=0
    fi
fi
KEEP_TRAJECTORY_EFFECTIVE="$KEEP_TRAJECTORY"

# ---------------------- 函数：解析 INCAR 整数参数 ----------------------
# 支持行内多参数（; 分隔），忽略 #/! 注释，同名参数取最后一次出现
parse_incar_int() {
    local param_name="$1"
    local incar_file="${2:-INCAR}"
    if [[ ! -f "$incar_file" ]]; then
        echo ""
        return
    fi
    awk -v key="$param_name" '
        BEGIN {
            ukey = toupper(key)
            value = ""
        }
        {
            line = $0
            sub(/[#!].*$/, "", line)
            n = split(line, seg, ";")
            for (i = 1; i <= n; i++) {
                s = seg[i]
                us = toupper(s)
                while (match(us, "(^|[^A-Z0-9_])" ukey "[[:space:]]*=[[:space:]]*[-+]?[0-9]+")) {
                    token = substr(s, RSTART, RLENGTH)
                    if (match(token, /=[[:space:]]*[-+]?[0-9]+/)) {
                        rhs = substr(token, RSTART + 1)
                        gsub(/[[:space:]]/, "", rhs)
                        value = rhs
                    }
                    s = substr(s, RSTART + RLENGTH)
                    us = toupper(s)
                }
            }
        }
        END {
            if (value != "") {
                print value
            }
        }
    ' "$incar_file" 2>/dev/null
}

format_bytes_human() {
    local bytes="${1:-0}"
    if command -v numfmt &>/dev/null; then
        numfmt --to=iec --suffix=B "$bytes" 2>/dev/null || echo "${bytes}B"
    else
        echo "${bytes}B"
    fi
}

rotate_resume_append_log() {
    local file="$1"
    local msg
    if [[ ! -f "$file" ]]; then
        msg="[RESUME] $file 不存在，无需轮转"
        echo "$msg"
        echo "$msg" >> run.log
        return 0
    fi

    local size
    size=$(stat -c%s "$file" 2>/dev/null || echo 0)
    local size_h threshold_h
    size_h=$(format_bytes_human "$size")
    threshold_h=$(format_bytes_human "$LOG_ROTATE_BYTES")

    if [[ $KEEP_APPEND_LOGS -eq 1 ]]; then
        msg="[RESUME] KEEP_APPEND_LOGS=1，保留 $file (size=$size_h)"
        echo "$msg"
        echo "$msg" >> run.log
        return 0
    fi

    if [[ "$size" -gt "$LOG_ROTATE_BYTES" ]]; then
        mkdir -p old
        local rotated="old/${file}.resume.${TIMESTAMP}"
        mv "$file" "$rotated"
        msg="[RESUME] 轮转 $file -> $rotated (size=$size_h, threshold=$threshold_h)"
    else
        msg="[RESUME] 保留 $file (size=$size_h <= threshold=$threshold_h)"
    fi
    echo "$msg"
    echo "$msg" >> run.log
}

get_poscar_natoms() {
    local poscar_file="${1:-POSCAR}"
    [[ -f "$poscar_file" ]] || return 1
    awk '
        { gsub(/\r/, "", $0) }
        function sum_if_int_fields(   i,sum) {
            if (NF < 1) return -1
            sum = 0
            for (i = 1; i <= NF; i++) {
                if ($i !~ /^[0-9]+$/) return -1
                sum += $i
            }
            return sum
        }
        NR == 6 {
            s = sum_if_int_fields()
            if (s > 0) {
                print s
                exit
            }
        }
        NR == 7 {
            s = sum_if_int_fields()
            if (s > 0) {
                print s
                exit
            }
        }
    ' "$poscar_file"
}

validate_contcar_for_resume() {
    local contcar_file="${1:-CONTCAR}"
    local expected_natoms="${2:-}"
    [[ -f "$contcar_file" ]] || return 1
    awk -v expected="$expected_natoms" '
        function trim(s) {
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }
        function split_fields(line, arr,    txt) {
            txt = line
            gsub(/\r/, "", txt)
            txt = trim(txt)
            if (txt == "") return 0
            return split(txt, arr, /[[:space:]]+/)
        }
        function is_int_line(line,    arr,n,i) {
            n = split_fields(line, arr)
            if (n < 1) return 0
            for (i = 1; i <= n; i++) {
                if (arr[i] !~ /^[0-9]+$/) return 0
            }
            return 1
        }
        function sum_int_line(line,    arr,n,i,sum) {
            n = split_fields(line, arr)
            sum = 0
            for (i = 1; i <= n; i++) sum += arr[i]
            return sum
        }
        function is_number(x) {
            return (x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([Ee][-+]?[0-9]+)?$/)
        }
        BEGIN { nline = 0 }
        {
            raw[++nline] = $0
            gsub(/\r/, "", raw[nline])
        }
        END {
            if (nline < 8) {
                print "CONTCAR 行数过少(" nline ")，至少需要 8 行"
                exit 1
            }

            split_fields(raw[2], scale)
            if (scale[1] == "" || !is_number(scale[1])) {
                print "CONTCAR 第 2 行缩放因子非法"
                exit 1
            }

            for (i = 3; i <= 5; i++) {
                nf = split_fields(raw[i], vec)
                if (nf < 3 || !is_number(vec[1]) || !is_number(vec[2]) || !is_number(vec[3])) {
                    print "CONTCAR 晶格向量非法(第 " i " 行)"
                    exit 1
                }
            }

            counts_line = 0
            if (is_int_line(raw[6])) {
                counts_line = 6
            } else if (is_int_line(raw[7])) {
                counts_line = 7
            } else {
                print "CONTCAR 第 6/7 行均不是有效原子计数行"
                exit 1
            }

            natoms = sum_int_line(raw[counts_line])
            if (natoms <= 0) {
                print "CONTCAR 原子总数非法: " natoms
                exit 1
            }

            coord_line = counts_line + 1
            mode = trim(raw[coord_line])
            low = tolower(mode)
            if (low ~ /^selective([[:space:]]+dynamics)?$/) {
                coord_line++
                mode = trim(raw[coord_line])
                low = tolower(mode)
            }
            if (low !~ /^(direct|cartesian)([[:space:]]|$)/) {
                print "CONTCAR 缺少 Direct/Cartesian 坐标模式行"
                exit 1
            }

            first_coord = coord_line + 1
            last_coord = first_coord + natoms - 1
            if (last_coord > nline) {
                print "CONTCAR 坐标块不完整: 需要 " natoms " 行，实际仅 " (nline - first_coord + 1) " 行"
                exit 1
            }

            for (i = first_coord; i <= last_coord; i++) {
                nf = split_fields(raw[i], c)
                if (nf < 3 || !is_number(c[1]) || !is_number(c[2]) || !is_number(c[3])) {
                    print "CONTCAR 坐标行非法(第 " i " 行)"
                    exit 1
                }
            }

            if (expected ~ /^[0-9]+$/ && expected + 0 != natoms) {
                print "CONTCAR 原子总数(" natoms ")与 POSCAR(" expected ")不一致"
                exit 1
            }

            print natoms
        }
    ' "$contcar_file"
}

outcar_has_completion_marker_since() {
    local outcar_file="$1"
    local start_line="$2"
    [[ -f "$outcar_file" ]] || return 1
    if ! [[ "$start_line" =~ ^[0-9]+$ ]] || [[ "$start_line" -lt 1 ]]; then
        start_line=1
    fi
    awk -v from_line="$start_line" '
        NR < from_line { next }
        /General timing and accounting informations for this job/ { found = 1; exit 0 }
        END { exit(found ? 0 : 1) }
    ' "$outcar_file"
}

extract_xdatcar_first_frame() {
    local xdatcar_file="$1"
    local natoms="$2"
    awk -v natoms="$natoms" '
        BEGIN { found = 0 }
        function norm_line(line,    i,n,out,f) {
            gsub(/\r/, "", line)
            n = split(line, f, /[[:space:]]+/)
            out = ""
            for (i = 1; i <= n; i++) {
                if (f[i] == "") continue
                if (out != "") out = out " "
                out = out f[i]
            }
            return out
        }
        /^[[:space:]]*Direct[[:space:]]+configuration[[:space:]]*=/ {
            found = 1
            for (i = 1; i <= natoms; i++) {
                if (getline line <= 0) exit 1
                print norm_line(line)
            }
            exit 0
        }
        END {
            if (!found) exit 1
        }
    ' "$xdatcar_file"
}

extract_xdatcar_last_frame() {
    local xdatcar_file="$1"
    local natoms="$2"
    local total_lines chunk
    total_lines=$(wc -l < "$xdatcar_file" 2>/dev/null || echo 0)
    if ! [[ "$total_lines" =~ ^[0-9]+$ ]] || [[ "$total_lines" -le 0 ]]; then
        return 1
    fi

    chunk=$((natoms + 64))
    if [[ "$chunk" -lt $((natoms + 1)) ]]; then
        chunk=$((natoms + 1))
    fi
    if [[ "$chunk" -gt "$total_lines" ]]; then
        chunk="$total_lines"
    fi

    while [[ "$chunk" -le "$total_lines" ]]; do
        if tail -n "$chunk" "$xdatcar_file" | awk -v natoms="$natoms" '
            function norm_line(line,    i,n,out,f) {
                gsub(/\r/, "", line)
                n = split(line, f, /[[:space:]]+/)
                out = ""
                for (i = 1; i <= n; i++) {
                    if (f[i] == "") continue
                    if (out != "") out = out " "
                    out = out f[i]
                }
                return out
            }
            BEGIN {
                in_frame = 0
                count = 0
                block = ""
                last = ""
            }
            /^[[:space:]]*Direct[[:space:]]+configuration[[:space:]]*=/ {
                in_frame = 1
                count = 0
                block = ""
                next
            }
            in_frame {
                count++
                block = block norm_line($0) "\n"
                if (count == natoms) {
                    last = block
                    in_frame = 0
                }
            }
            END {
                if (last == "") exit 1
                printf "%s", last
            }
        '; then
            return 0
        fi

        if [[ "$chunk" -eq "$total_lines" ]]; then
            break
        fi
        chunk=$((chunk * 2))
        if [[ "$chunk" -gt "$total_lines" ]]; then
            chunk="$total_lines"
        fi
    done

    return 1
}

# ---------------------- 函数：检查 stdbuf 可用性 ----------------------
check_stdbuf() {
    command -v stdbuf &>/dev/null
}

run_with_stdbuf() {
    if [[ ${#STDBUF_CMD[@]} -gt 0 ]]; then
        "${STDBUF_CMD[@]}" "$@"
    else
        "$@"
    fi
}

# ---------------------- 文件检查 ----------------------
echo "============================================"
echo "[run_vasp] 当前目录: $(pwd)"
echo "[run_vasp] NP=$NP  EXE=$EXE  OUT=$OUT  ERR=$ERR"
echo "[run_vasp] OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
echo "[run_vasp] 时间戳: $TIMESTAMP"
echo "[run_vasp] RESUME=$RESUME  KEEP_RESTART=$KEEP_RESTART  KEEP_TRAJECTORY=$KEEP_TRAJECTORY"
echo "[run_vasp] IS_WSL=$IS_WSL  FORCE_OMP_BIND=$FORCE_OMP_BIND"
echo "============================================"

# 必需文件检查
missing=0
for f in INCAR POSCAR POTCAR; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] 缺少必需文件: $f"
        missing=1
    else
        echo "[OK] $f 存在"
    fi
done

if [[ $missing -eq 1 ]]; then
    echo "[ABORT] 请补齐必需输入文件后重试。"
    exit 1
fi

# ---------------------- KPOINTS / KSPACING 检查 ----------------------
if [[ ! -f "KPOINTS" ]]; then
    if grep -qiE "^[[:space:]]*KSPACING[[:space:]]*=" INCAR 2>/dev/null; then
        echo "[INFO] KPOINTS 文件不存在，但 INCAR 中设置了 KSPACING"
    else
        if [[ $STRICT_INPUT -eq 1 ]]; then
            echo "[ERROR] 既无 KPOINTS 文件，INCAR 中也无 KSPACING，K点采样未定义"
            exit 1
        else
            echo "[WARN] KPOINTS 文件不存在，INCAR 中也无 KSPACING，请确保 K 点配置正确"
        fi
    fi
else
    echo "[OK] KPOINTS 存在"
fi

# 检查可执行文件
if [[ ! -x "$VASP_BIN/$EXE" ]]; then
    echo "[ERROR] 可执行文件不存在或无执行权限: $VASP_BIN/$EXE"
    exit 1
fi
echo "[OK] 可执行文件: $VASP_BIN/$EXE"

# ---------------------- 核数检查 ----------------------
echo ""
echo ">>> 核数检查..."
TOTAL_CORES=$(nproc 2>/dev/null || echo 8)
if [[ $IS_WSL -eq 1 ]]; then
    MAX_NP=$((TOTAL_CORES - RESERVE_CORES))
    if [[ $MAX_NP -lt 1 ]]; then
        MAX_NP=1
    fi
    echo "    [WSL] 总核数: $TOTAL_CORES, 预留: $RESERVE_CORES, 可用: $MAX_NP"
else
    MAX_NP=$TOTAL_CORES
    echo "    [非 WSL] 总核数: $TOTAL_CORES, 不应用 RESERVE_CORES 预留"
fi

if [[ $NP -gt $MAX_NP ]]; then
    if [[ $STRICT_NP -eq 1 ]]; then
        echo "[ERROR] NP=$NP 超过可用核数 $MAX_NP (STRICT_NP=1)"
        echo "[INFO] 设置 STRICT_NP=0 可自动下调"
        exit 1
    else
        echo "[WARN] NP=$NP 超过可用核数，自动下调为 $MAX_NP"
        NP=$MAX_NP
    fi
fi
echo "[OK] 使用 NP=$NP"

# ---------------------- 磁盘空间检查 (POSIX 兼容) ----------------------
echo ""
echo ">>> 磁盘空间检查..."
# 使用 df -P -B1 获取字节数，更可靠
FREE_BYTES=$(df -P -B1 . 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
if [[ -z "$FREE_BYTES" || "$FREE_BYTES" == "0" ]]; then
    # 回退到 df -k
    FREE_KB=$(df -k . 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
    FREE_GB=$((FREE_KB / 1024 / 1024))
else
    FREE_GB=$((FREE_BYTES / 1024 / 1024 / 1024))
fi

echo "    可用空间: ${FREE_GB} GB, 最小要求: ${MIN_FREE_GB} GB"

if [[ $FREE_GB -lt $MIN_FREE_GB ]]; then
    if [[ $FORCE_DISK -eq 1 ]]; then
        echo "[WARN] 磁盘空间不足，但 FORCE_DISK=1，继续运行"
    else
        echo "[ERROR] 磁盘空间不足 (${FREE_GB} < ${MIN_FREE_GB} GB)"
        echo "[INFO] 设置 FORCE_DISK=1 可强制继续"
        exit 1
    fi
else
    echo "[OK] 磁盘空间充足"
fi

# ---------------------- 并行参数提示 (awk 解析) ----------------------
echo ""
echo ">>> 并行参数检查..."

NCORE=$(parse_incar_int "NCORE")
KPAR=$(parse_incar_int "KPAR")

if [[ -n "$NCORE" ]]; then
    if ! [[ "$NCORE" =~ ^[1-9][0-9]*$ ]]; then
        echo "[WARN] NCORE 解析异常: '$NCORE'，忽略"
        NCORE=""
    elif [[ $((NP % NCORE)) -ne 0 ]]; then
        echo "[WARN] NP=$NP 不能被 NCORE=$NCORE 整除，可能影响性能"
    else
        echo "    NCORE=$NCORE, NP/NCORE=$((NP / NCORE))"
    fi
fi

if [[ -n "$KPAR" ]]; then
    if ! [[ "$KPAR" =~ ^[1-9][0-9]*$ ]]; then
        echo "[WARN] KPAR 解析异常: '$KPAR'，忽略"
        KPAR=""
    elif [[ $((NP % KPAR)) -ne 0 ]]; then
        echo "[WARN] NP=$NP 不能被 KPAR=$KPAR 整除，可能影响性能"
    else
        echo "    KPAR=$KPAR"
    fi
fi

# ---------------------- 重启文件检查 (ISTART/ICHARG 守卫) ----------------------
echo ""
echo ">>> 重启文件检查..."

ISTART=$(parse_incar_int "ISTART")
ICHARG=$(parse_incar_int "ICHARG")

echo "    INCAR: ISTART=${ISTART:-未设置}, ICHARG=${ICHARG:-未设置}"

# ISTART >= 1 需要 WAVECAR
if [[ -n "$ISTART" && "$ISTART" -ge 1 ]]; then
    if [[ ! -f "WAVECAR" || ! -s "WAVECAR" ]]; then
        if [[ $FORCE_RESTART -eq 1 ]]; then
            echo "[WARN] ISTART=$ISTART 但 WAVECAR 不存在/为空 (FORCE_RESTART=1, 继续)"
        else
            echo "[ERROR] ISTART=$ISTART 需要 WAVECAR，但文件不存在或为空"
            echo "[INFO] 设置 FORCE_RESTART=1 可强制运行（从头计算波函数）"
            exit 1
        fi
    else
        echo "[OK] WAVECAR 存在 ($(stat -c%s WAVECAR 2>/dev/null | numfmt --to=iec 2>/dev/null || stat -c%s WAVECAR))"
    fi
fi

# ICHARG in {0, 1} 需要 CHGCAR (ICHARG=0 从 WAVECAR 构建，ICHARG=1 从 CHGCAR 读取)
if [[ -n "$ICHARG" && ("$ICHARG" -eq 0 || "$ICHARG" -eq 1) ]]; then
    # ICHARG=1 需要 CHGCAR
    if [[ "$ICHARG" -eq 1 ]]; then
        if [[ ! -f "CHGCAR" || ! -s "CHGCAR" ]]; then
            if [[ $FORCE_RESTART -eq 1 ]]; then
                echo "[WARN] ICHARG=1 但 CHGCAR 不存在/为空 (FORCE_RESTART=1, 继续)"
            else
                echo "[ERROR] ICHARG=1 需要 CHGCAR，但文件不存在或为空"
                echo "[INFO] 设置 FORCE_RESTART=1 可强制运行"
                exit 1
            fi
        else
            echo "[OK] CHGCAR 存在 (ICHARG=1)"
        fi
    fi
fi

# ---------------------- AIMD 续算支持 ----------------------
echo ""
echo ">>> 续算检查..."

if [[ -f CONTCAR ]]; then
    contcar_size=$(stat -c%s CONTCAR 2>/dev/null || echo 0)
    if [[ $RESUME -eq 1 ]]; then
        if [[ $contcar_size -le 0 ]]; then
            echo "[ERROR] CONTCAR 为空，拒绝覆盖 POSCAR"
            exit 1
        fi

        poscar_natoms_before=$(get_poscar_natoms POSCAR || true)
        contcar_check_msg=$(validate_contcar_for_resume CONTCAR "$poscar_natoms_before" 2>&1)
        if [[ $? -ne 0 ]]; then
            echo "[ERROR] CONTCAR 校验失败: $contcar_check_msg"
            echo "[ERROR] 拒绝使用可疑 CONTCAR 覆盖 POSCAR"
            exit 1
        fi

        echo "[RESUME] 检测到 CONTCAR，启用续算模式 (NATOMS=$contcar_check_msg)"
        echo "[RESUME] 备份 POSCAR -> POSCAR.bak.$TIMESTAMP"
        cp POSCAR "POSCAR.bak.$TIMESTAMP"
        echo "[RESUME] 复制 CONTCAR -> POSCAR"
        cp CONTCAR POSCAR
        RESUME_CONTINUATION_READY=1
        echo "[OK] 续算准备完成"
    elif [[ $contcar_size -gt 0 ]]; then
        echo "[INFO] 检测到 CONTCAR，如需续算请使用: RESUME=1 run_vasp.sh"
    else
        echo "[INFO] 检测到空 CONTCAR，未自动使用"
    fi
else
    if [[ $RESUME -eq 1 ]]; then
        echo "[WARN] RESUME=1 但无 CONTCAR，跳过 CONTCAR->POSCAR；本轮不视为轨迹续算"
    else
        echo "[INFO] 无 CONTCAR，非续算"
    fi
fi

# ---------------------- XDATCAR 轨迹保护 (续算时合并) ----------------------
XDATCAR_PREV=""
if [[ $KEEP_TRAJECTORY -eq 1 ]]; then
    if [[ $RESUME_CONTINUATION_READY -eq 1 ]]; then
        if [[ -f XDATCAR && -s XDATCAR ]]; then
            echo "[TRAJECTORY] 保留 XDATCAR 以确保轨迹连续"
            XDATCAR_PREV="XDATCAR.prev.$TIMESTAMP"
            cp XDATCAR "$XDATCAR_PREV"
            echo "[TRAJECTORY] 备份: XDATCAR -> $XDATCAR_PREV"
        else
            echo "[TRAJECTORY] 未检测到可用旧 XDATCAR，跳过备份"
        fi
    else
        KEEP_TRAJECTORY_EFFECTIVE=0
        echo "[WARN] KEEP_TRAJECTORY=1 但当前不是有效续算，已禁用轨迹保留以避免跨任务拼接"
    fi
fi

# ---------------------- RESUME 日志轮转保护 ----------------------
if [[ $RESUME -eq 1 ]]; then
    echo ""
    echo ">>> RESUME 日志文件策略..."
    for append_file in OUTCAR OSZICAR; do
        rotate_resume_append_log "$append_file"
    done
fi

# ---------------------- 备份输入文件到 snapshots ----------------------
echo ""
echo ">>> 备份输入文件..."
SNAPSHOT_DIR="snapshots/$TIMESTAMP"
mkdir -p "$SNAPSHOT_DIR"

for f in INCAR POSCAR POTCAR KPOINTS; do
    if [[ -f "$f" ]]; then
        cp "$f" "$SNAPSHOT_DIR/"
        echo "    [备份] $f -> $SNAPSHOT_DIR/"
    fi
done
echo "[OK] 输入文件已备份到: $SNAPSHOT_DIR/"

# ---------------------- 移动旧输出文件到 old (条件化) ----------------------
echo ""
echo ">>> 移动旧输出文件..."

# 根据 KEEP_RESTART 和 KEEP_TRAJECTORY_EFFECTIVE 决定哪些文件不移动
RESTART_FILES=(WAVECAR CHGCAR CHG)
TRAJECTORY_FILES=(XDATCAR)
APPEND_FILES=(OUTCAR OSZICAR)  # VASP 可能追加的文件
OTHER_FILES=(CONTCAR vasprun.xml "$OUT" "$ERR")
# 注：RESUME=1 时 CONTCAR 已用于更新 POSCAR；此处归档旧 CONTCAR 以保留审计痕迹

moved_count=0

move_old_file() {
    local f="$1"
    if [[ -f "$f" ]]; then
        local fsize
        fsize=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [[ $fsize -gt 0 ]]; then
            mkdir -p old
            local new_name="${f}.${TIMESTAMP}"
            mv "$f" "old/$new_name"
            echo "    [移动] $f -> old/$new_name"
            ((moved_count++)) || true
        fi
    fi
}

# 处理重启文件
if [[ $KEEP_RESTART -eq 1 ]]; then
    echo "    [KEEP_RESTART=1] 保留 WAVECAR/CHGCAR/CHG"
else
    for f in "${RESTART_FILES[@]}"; do
        move_old_file "$f"
    done
fi

# 处理轨迹文件
if [[ $KEEP_TRAJECTORY_EFFECTIVE -eq 1 ]]; then
    echo "    [KEEP_TRAJECTORY=1] 保留 XDATCAR（已备份到 .prev）"
else
    if [[ $KEEP_TRAJECTORY -eq 1 ]]; then
        echo "    [KEEP_TRAJECTORY=1] 非续算场景禁用保留，避免误拼接旧轨迹"
    fi
    for f in "${TRAJECTORY_FILES[@]}"; do
        move_old_file "$f"
    done
fi

# 处理追加文件（RESUME 时保留，VASP 会追加）
if [[ $RESUME -eq 1 ]]; then
    if [[ $KEEP_APPEND_LOGS -eq 1 ]]; then
        echo "    [RESUME=1] KEEP_APPEND_LOGS=1，保留 OUTCAR/OSZICAR 追加"
    else
        echo "    [RESUME=1] OUTCAR/OSZICAR 小文件保留，超阈值已轮转"
    fi
else
    for f in "${APPEND_FILES[@]}"; do
        move_old_file "$f"
    done
fi

# 处理其他输出文件
for f in "${OTHER_FILES[@]}"; do
    move_old_file "$f"
done

if [[ $moved_count -eq 0 ]]; then
    echo "    [INFO] 没有需要移动的旧输出文件"
else
    echo "[OK] 已移动 $moved_count 个旧输出文件到 old/"
fi

# ---------------------- 线程控制 (HPC 安全) ----------------------
echo ""
echo ">>> 线程控制 (THREAD_GUARD=$THREAD_GUARD)..."

if [[ $THREAD_GUARD -eq 1 ]]; then
    # 强制安全线程默认值
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
    export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$OMP_NUM_THREADS}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"
    if [[ $IS_WSL -eq 1 && $FORCE_OMP_BIND -ne 1 ]]; then
        echo "    [INFO] 检测到 WSL，默认不强制 OMP_PROC_BIND/OMP_PLACES"
    else
        export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
        export OMP_PLACES="${OMP_PLACES:-cores}"
    fi
    
    echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "    MKL_NUM_THREADS=$MKL_NUM_THREADS"
    echo "    OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    echo "    OMP_PROC_BIND=${OMP_PROC_BIND:-unset}"
    echo "    OMP_PLACES=${OMP_PLACES:-unset}"
    echo "[OK] 线程保护已启用"
else
    echo "[WARN] THREAD_GUARD=0，线程控制未启用（可能导致 HPC 过载）"
    echo "    OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
fi

# ---------------------- 记录运行日志 ----------------------
echo ""
echo ">>> 记录运行日志..."
{
    echo "============================================"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "目录: $(pwd)"
    echo "NP: $NP"
    echo "EXE: $EXE"
    echo "OUT: $OUT"
    echo "ERR: $ERR"
	echo "RESUME: $RESUME"
	echo "KEEP_RESTART: $KEEP_RESTART"
	echo "KEEP_TRAJECTORY: $KEEP_TRAJECTORY"
	echo "KEEP_TRAJECTORY_EFFECTIVE: $KEEP_TRAJECTORY_EFFECTIVE"
	echo "KEEP_APPEND_LOGS: $KEEP_APPEND_LOGS"
	echo "LOG_ROTATE_GB: $LOG_ROTATE_GB"
	echo "RESUME_CONTINUATION_READY: $RESUME_CONTINUATION_READY"
	echo "THREAD_GUARD: $THREAD_GUARD"
	echo "FORCE_OMP_BIND: $FORCE_OMP_BIND"
	echo "IS_WSL: $IS_WSL"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset}"
    echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-unset}"
    echo "ISTART: ${ISTART:-未设置}"
    echo "ICHARG: ${ICHARG:-未设置}"
    echo "快照: $SNAPSHOT_DIR"
    if [[ -n "$XDATCAR_PREV" ]]; then
        echo "XDATCAR 备份: $XDATCAR_PREV"
    fi
    echo "============================================"
} >> run.log
echo "[OK] 运行参数已记录到 run.log"

# ---------------------- 检查 stdbuf ----------------------
STDBUF_CMD=()
if check_stdbuf; then
    STDBUF_CMD=(stdbuf -oL -eL)
    echo "[OK] stdbuf 可用，启用行缓冲"
else
    echo "[WARN] stdbuf 不可用，tail -f 可能延迟"
fi

# ---------------------- 运行 VASP ----------------------
echo ""
echo "============================================"
echo "[run_vasp] 开始运行: mpirun -np $NP $VASP_BIN/$EXE"
echo "[run_vasp] stdout: $OUT"
echo "[run_vasp] stderr: $ERR"
echo "[run_vasp] 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "[INFO] 监控命令:"
echo "       tail -f $OUT"
echo "       tail -f $ERR"
echo "       aimd_watch.sh"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> run.log
OUTCAR_BASE_LINES=0
if [[ -f OUTCAR ]]; then
    OUTCAR_BASE_LINES=$(wc -l < OUTCAR 2>/dev/null || echo 0)
fi
OUTCAR_CHECK_START=$((OUTCAR_BASE_LINES + 1))
echo "本轮 OUTCAR 检查起始行: $OUTCAR_CHECK_START" >> run.log

# ---------------------- 运行 mpirun (分离 stdout/stderr) ----------------------
MPIRUN_OUT_FIFO="$TMP_ROOT/mpirun.stdout.fifo"
MPIRUN_ERR_FIFO="$TMP_ROOT/mpirun.stderr.fifo"
rm -f "$MPIRUN_OUT_FIFO" "$MPIRUN_ERR_FIFO"
if ! mkfifo "$MPIRUN_OUT_FIFO" "$MPIRUN_ERR_FIFO"; then
    echo "[ERROR] 无法创建 mpirun 日志管道"
    exit 1
fi
register_tmp_path "$MPIRUN_OUT_FIFO"
register_tmp_path "$MPIRUN_ERR_FIFO"

(
    run_with_stdbuf tee "$OUT" < "$MPIRUN_OUT_FIFO" \
        | run_with_stdbuf awk '/(E0|F=|DAV|RMM|Error|error|WARNING|STOP)/ { print; fflush() }'
) &
OUT_LOGGER_PID=$!

run_with_stdbuf tee "$ERR" < "$MPIRUN_ERR_FIFO" >&2 &
ERR_LOGGER_PID=$!

run_with_stdbuf mpirun -np "$NP" "$VASP_BIN/$EXE" > "$MPIRUN_OUT_FIFO" 2> "$MPIRUN_ERR_FIFO"
VASP_RC=$?

wait "$OUT_LOGGER_PID"
OUT_LOGGER_RC=$?
wait "$ERR_LOGGER_PID"
ERR_LOGGER_RC=$?

END_TIME=$(date +%s)
WALL_TIME=$((END_TIME - START_TIME))

# 转换为 h:m:s
WALL_H=$((WALL_TIME / 3600))
WALL_M=$(((WALL_TIME % 3600) / 60))
WALL_S=$((WALL_TIME % 60))

echo ""
echo "[run_vasp] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[run_vasp] 运行耗时: ${WALL_H}h ${WALL_M}m ${WALL_S}s (${WALL_TIME}s)"

# 记录到日志
{
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "运行耗时: ${WALL_H}h ${WALL_M}m ${WALL_S}s (${WALL_TIME}s)"
    echo "mpirun 返回码: $VASP_RC"
    if [[ $OUT_LOGGER_RC -ne 0 ]]; then
        echo "stdout logger 返回码: $OUT_LOGGER_RC"
    fi
    if [[ $ERR_LOGGER_RC -ne 0 ]]; then
        echo "stderr logger 返回码: $ERR_LOGGER_RC"
    fi
} >> run.log

if [[ $OUT_LOGGER_RC -ne 0 ]]; then
    echo "[WARN] stdout 日志处理异常 (rc=$OUT_LOGGER_RC)"
fi
if [[ $ERR_LOGGER_RC -ne 0 ]]; then
    echo "[WARN] stderr 日志处理异常 (rc=$ERR_LOGGER_RC)"
fi

CURRENT_RUN_OUTCAR_DONE=0
if outcar_has_completion_marker_since OUTCAR "$OUTCAR_CHECK_START"; then
    CURRENT_RUN_OUTCAR_DONE=1
fi

CURRENT_RUN_ACCEPTED=0
if [[ $VASP_RC -eq 0 && $CURRENT_RUN_OUTCAR_DONE -eq 1 ]]; then
    CURRENT_RUN_ACCEPTED=1
fi

{
    echo "本轮完成标志: $CURRENT_RUN_OUTCAR_DONE"
    echo "本轮接受状态: $CURRENT_RUN_ACCEPTED"
} >> run.log

# ---------------------- XDATCAR 轨迹合并 ----------------------
if [[ $CURRENT_RUN_ACCEPTED -eq 1 && -n "$XDATCAR_PREV" && -f "$XDATCAR_PREV" && -f "XDATCAR" ]]; then
    echo ""
    echo ">>> XDATCAR 轨迹合并..."

    prev_lines=$(wc -l < "$XDATCAR_PREV")
    new_lines=$(wc -l < XDATCAR)
    XDATCAR_MERGED="XDATCAR.merged.$TIMESTAMP"
    cp "$XDATCAR_PREV" "$XDATCAR_MERGED"

    header_start=$(grep -nE "^[[:space:]]*Direct[[:space:]]+configuration[[:space:]]*=" XDATCAR 2>/dev/null | head -1 | cut -d: -f1)
    header_start=${header_start:-8}
    if [[ "$header_start" -lt 2 ]]; then
        header_start=8
    fi

    natoms=$(get_poscar_natoms POSCAR || true)
    dedup_possible=1
    skip_first_frame=0
    if ! [[ "$natoms" =~ ^[1-9][0-9]*$ ]]; then
        dedup_possible=0
    fi

    prev_frame_tmp=$(mktemp "$TMP_ROOT/xdat_prev_frame.XXXXXX" 2>/dev/null || true)
    new_frame_tmp=$(mktemp "$TMP_ROOT/xdat_new_frame.XXXXXX" 2>/dev/null || true)
    if [[ -z "$prev_frame_tmp" || -z "$new_frame_tmp" ]]; then
        dedup_possible=0
    else
        register_tmp_path "$prev_frame_tmp"
        register_tmp_path "$new_frame_tmp"
    fi

    if [[ $dedup_possible -eq 1 ]]; then
        if ! extract_xdatcar_last_frame "$XDATCAR_PREV" "$natoms" > "$prev_frame_tmp" 2>/dev/null; then
            dedup_possible=0
        fi
        if ! extract_xdatcar_first_frame XDATCAR "$natoms" > "$new_frame_tmp" 2>/dev/null; then
            dedup_possible=0
        fi
    fi

    if [[ $dedup_possible -eq 1 ]] && cmp -s "$prev_frame_tmp" "$new_frame_tmp"; then
        skip_first_frame=1
    fi

    if [[ $dedup_possible -eq 1 ]]; then
        if [[ $skip_first_frame -eq 1 ]]; then
            awk -v natoms="$natoms" '
                BEGIN {
                    in_data = 0
                    cfg = 0
                    skip = 0
                }
                /^[[:space:]]*Direct[[:space:]]+configuration[[:space:]]*=/ {
                    in_data = 1
                    cfg++
                    if (cfg == 1) {
                        skip = natoms
                        next
                    }
                    print
                    next
                }
                in_data {
                    if (cfg == 1 && skip > 0) {
                        skip--
                        next
                    }
                    print
                }
            ' XDATCAR >> "$XDATCAR_MERGED"
            echo "[TRAJECTORY] 检测到重复 frame 0，已跳过新 XDATCAR 的第一帧"
            echo "XDATCAR 合并: 去重后追加（跳过第一帧）" >> run.log
        else
            awk '
                BEGIN { in_data = 0 }
                /^[[:space:]]*Direct[[:space:]]+configuration[[:space:]]*=/ { in_data = 1 }
                in_data { print }
            ' XDATCAR >> "$XDATCAR_MERGED"
            echo "[TRAJECTORY] 首尾帧不重复，完整追加新 XDATCAR"
            echo "XDATCAR 合并: 追加全部新帧（无重复第一帧）" >> run.log
        fi
    else
        echo "[WARNING] 无法可靠解析 XDATCAR 帧进行去重，回退为仅跳过头部追加"
        echo "[WARNING] XDATCAR 去重失败，回退为 header-skip 追加（请人工检查是否重复 frame 0）" >> run.log
        tail -n +"$header_start" XDATCAR >> "$XDATCAR_MERGED"
    fi

    rm -f "$prev_frame_tmp" "$new_frame_tmp"

    mv XDATCAR "XDATCAR.new.$TIMESTAMP"
    mv "$XDATCAR_MERGED" XDATCAR

    merged_lines=$(wc -l < XDATCAR)
    echo "[TRAJECTORY] 合并完成: $prev_lines + $new_lines -> $merged_lines 行"
    echo "XDATCAR 合并总行数: prev=$prev_lines, new=$new_lines, merged=$merged_lines" >> run.log
elif [[ -n "$XDATCAR_PREV" && -f "$XDATCAR_PREV" && -f "XDATCAR" ]]; then
    echo "[WARN] 本轮运行未通过完成检查，跳过 XDATCAR 合并以保护轨迹完整性"
    echo "XDATCAR 合并已跳过: 本轮未被接受" >> run.log
fi

# ---------------------- 错误处理 ----------------------
if [[ $VASP_RC -ne 0 ]]; then
    echo ""
    echo "[ERROR] mpirun 返回非零退出码: $VASP_RC"
    echo "状态: 失败 (rc=$VASP_RC)" >> run.log
    
    # 记录错误日志尾部
    if [[ -f "$ERR" && -s "$ERR" ]]; then
        echo "" >> run.log
        echo "=== $ERR 最后 30 行 ===" >> run.log
        tail -30 "$ERR" >> run.log 2>/dev/null || true
        
        echo ""
        echo "[INFO] $ERR 最后 10 行:"
        tail -10 "$ERR" 2>/dev/null || true
    fi
    
    echo "[INFO] 检查 $ERR 获取详细错误信息"
    echo "[INFO] 检查 run.log 获取完整记录"
fi

# ---------------------- 完成检查 ----------------------
echo ""
FINAL_RC=$VASP_RC
if [[ $CURRENT_RUN_ACCEPTED -eq 1 ]]; then
    echo "[OK] VASP 本轮计算正常完成。"
    echo "状态: 正常完成" >> run.log

    # 仅从本轮输出段提取 CPU 时间，避免读取旧续算段
    total_cpu=$(awk -v from_line="$OUTCAR_CHECK_START" '
        NR < from_line { next }
        /Total CPU time used/ { line = $0 }
        END { if (line != "") print line }
    ' OUTCAR 2>/dev/null || true)
    if [[ -n "$total_cpu" ]]; then
        echo "    $total_cpu"
        echo "$total_cpu" >> run.log
    fi
else
    if [[ $VASP_RC -eq 0 ]]; then
        echo "[ERROR] mpirun 返回 0，但本轮 OUTCAR 未检测到完成标志。"
        echo "状态: 失败（本轮无完成标志）" >> run.log
        FINAL_RC=1
    else
        if [[ -f OUTCAR ]]; then
            echo "[WARN] 本轮运行失败，且本轮 OUTCAR 段未通过完成检查。"
            echo "状态: 失败（rc=$VASP_RC）" >> run.log
        else
            echo "[WARN] OUTCAR 不存在，计算可能失败。"
            echo "状态: 失败（无 OUTCAR）" >> run.log
        fi
    fi
fi

echo "" >> run.log
echo "[run_vasp] 结束。"

if [[ $FINAL_RC -ne 0 ]]; then
    exit "$FINAL_RC"
fi
