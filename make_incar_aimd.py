#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_incar_aimd.py - 根据 recipe.yaml 生成 AIMD INCAR 文件 (v3.0.2)

功能：
  - 读取 recipe.yaml 的 simulation 段
  - 自动检测含 H 体系并调整 POTIM（dt_fs 可选）
  - 支持 Langevin/Nosé-Hoover 恒温器
  - LANGEVIN_GAMMA 正确输出为向量（每种原子类型一个值）
  - 支持两段式 INCAR（平衡/生产），带智能 ISTART/ICHARG
  - 强制 ISYM=0，MAXMIX 可配置
  - 安全默认：stage-aware ALGO/LREAL

v3.0 关键改进：
  - 鲁棒 POSCAR 解析：支持 VASP4/VASP5/Selective dynamics
  - LANGEVIN_GAMMA 输出向量（每种原子类型一个值），不再是标量
  - dt_fs 可选：未指定时根据 H 检测自动设置
  - ISTART/ICHARG 区分平衡/生产段
  - ALGO/LREAL 可配置，带安全默认值

v3.0.1 修复摘要（正确性/稳健性）：
  - 修复 MDALGO 映射：Langevin=3，Nosé-Hoover=2
  - 两段式预生成 prod 时默认重启意图：ISTART=1, ICHARG=0（不依赖生成时 WAVECAR）
  - INCAR.base 解析支持分号多参数与覆盖规则
  - ALGO/LREAL/LWAVE/LCHARG 确定性优先级解析并确保输出
  - gamma 键名兼容增强 + 高 gamma 默认硬错误

v3.0.2 修复摘要（安全默认 / traceability）：
  - 修复 dt_fs: null 时的 POTIM/摘要崩溃，并统一时间步解析路径
  - H 检测未知时采用保守策略：按“含 H”处理，避免静默不安全默认
  - thermostat 支持 nose / nose-hoover / nh 别名
  - two_stage 终端摘要按阶段分别解析，避免 eq/prod 默认值显示错误
  - INCAR 头部新增 has_h / dt_fs / ISTART/ICHARG 解析来源注释

用法：
  python3 make_incar_aimd.py [--recipe recipe.yaml] [--out INCAR]
  python3 make_incar_aimd.py --two_stage  # 生成 INCAR.eq 和 INCAR.prod

作者：STAR0418-ABC
"""

import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# POSCAR Parsing
# =============================================================================

def _strip_inline_comment(line: str, strip_leading: bool = False) -> str:
    """Strip inline comments introduced by '#' or '!'."""
    cut_positions = [len(line)]
    for marker in ('#', '!'):
        idx = line.find(marker)
        if idx >= 0:
            if idx == 0 and not strip_leading:
                continue
            cut_positions.append(idx)
    return line[:min(cut_positions)].strip()


def _is_all_int_tokens(tokens: List[str]) -> bool:
    """Return True if all tokens are integers."""
    if not tokens:
        return False
    try:
        for tok in tokens:
            int(tok)
        return True
    except ValueError:
        return False


def _next_nonempty_line(lines: List[Tuple[int, str]], cursor: int, field_name: str) -> Tuple[str, int]:
    """Consume the next non-empty parsed line."""
    if cursor >= len(lines):
        print(f"[ERROR] POSCAR: 缺少 {field_name}")
        sys.exit(1)
    lineno, content = lines[cursor]
    if not content:
        print(f"[ERROR] POSCAR: 第 {lineno} 行为空，缺少 {field_name}")
        sys.exit(1)
    return content, cursor + 1


def parse_poscar_header(poscar_path: str) -> Dict[str, Any]:
    """
    Parse POSCAR header robustly for VASP4/VASP5.

    Sequential parsing strategy (ignores blank lines, strips inline comments):
      1) comment, scale, 3 lattice lines
      2) next non-empty line:
         - all integers => VASP4 counts
         - otherwise => VASP5 element symbols; next line must be counts

    Returns:
        {
            'elements': Optional[List[str]],  # None if not present (VASP4)
            'counts': List[int],
            'ntyp': int,
            'natoms': int,
        }

    Raises:
        SystemExit on parse failure
    """
    if not os.path.isfile(poscar_path):
        print(f"[ERROR] POSCAR 不存在: {poscar_path}")
        sys.exit(1)

    with open(poscar_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    parsed_lines: List[Tuple[int, str]] = []
    for lineno, raw in enumerate(raw_lines, start=1):
        cleaned = _strip_inline_comment(raw.rstrip('\n'), strip_leading=False)
        if cleaned:
            parsed_lines.append((lineno, cleaned))

    if len(parsed_lines) < 5:
        print(f"[ERROR] POSCAR 格式错误: 有效非空行不足 ({len(parsed_lines)} < 5)")
        sys.exit(1)

    cursor = 0
    _, cursor = _next_nonempty_line(parsed_lines, cursor, "注释行")
    _, cursor = _next_nonempty_line(parsed_lines, cursor, "缩放因子")
    _, cursor = _next_nonempty_line(parsed_lines, cursor, "晶格向量 a")
    _, cursor = _next_nonempty_line(parsed_lines, cursor, "晶格向量 b")
    _, cursor = _next_nonempty_line(parsed_lines, cursor, "晶格向量 c")

    line_a, cursor = _next_nonempty_line(parsed_lines, cursor, "元素符号或原子数行")
    tokens_a = line_a.split()

    if _is_all_int_tokens(tokens_a):
        elements = None
        counts = [int(x) for x in tokens_a]
    else:
        elements = tokens_a
        line_b, cursor = _next_nonempty_line(parsed_lines, cursor, "原子数行")
        tokens_b = line_b.split()
        if not _is_all_int_tokens(tokens_b):
            print("[ERROR] POSCAR: 元素行后面的原子数行必须为整数列表")
            print(f"        当前行内容: {line_b}")
            print("        提示: Selective dynamics 应位于原子数行之后")
            sys.exit(1)
        counts = [int(x) for x in tokens_b]

    if any(c <= 0 for c in counts):
        print(f"[ERROR] POSCAR: 原子数必须为正整数，得到: {counts}")
        sys.exit(1)

    ntyp = len(counts)
    natoms = sum(counts)

    if ntyp == 0 or natoms == 0:
        print("[ERROR] POSCAR: 解析出 NTYP=0 或 NATOMS=0")
        sys.exit(1)

    if elements is not None and len(elements) != ntyp:
        print(f"[ERROR] POSCAR: 元素个数 ({len(elements)}) 与计数列长度 ({ntyp}) 不一致")
        print(f"        元素: {elements}")
        print(f"        计数: {counts}")
        sys.exit(1)

    return {
        'elements': elements,
        'counts': counts,
        'ntyp': ntyp,
        'natoms': natoms,
    }


def detect_hydrogen(
    poscar_info: Dict[str, Any],
    potcar_path: str = "POTCAR",
    yaml_override: Optional[bool] = None
) -> Tuple[bool, str]:
    """
    Detect if system contains hydrogen.
    
    Priority:
    1. YAML override (simulation.has_h)
    2. POSCAR element symbols
    3. POTCAR VRHFIN scan
    4. Unknown (safe default: assume H present)
    
    Returns:
        (has_h: bool, source: str)
        source is one of: 'yaml_override', 'poscar_elements', 'potcar', 'unknown_safe_default'
    """
    # 1. YAML override
    if yaml_override is not None:
        return (yaml_override, 'yaml_override')
    
    elements = poscar_info.get('elements')
    
    # 2. POSCAR elements
    if elements is not None:
        has_h = 'H' in elements
        return (has_h, 'poscar_elements')
    
    # 3. POTCAR fallback
    if os.path.isfile(potcar_path):
        try:
            with open(potcar_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # POTCAR format: "   VRHFIN =H: ..."
                    if 'VRHFIN' in line and '=' in line:
                        # Extract element after =
                        parts = line.split('=', 1)
                        if len(parts) > 1:
                            elem_part = parts[1].split(':')[0].strip()
                            if elem_part == 'H':
                                return (True, 'potcar')
            # Scanned POTCAR, no H found
            return (False, 'potcar')
        except Exception as e:
            print(f"[WARN] 读取 POTCAR 时出错: {e}")
    
    # 4. Unknown -> safer backward-compatible policy
    print("[WARN] 无法确定体系是否含 H 原子（POSCAR 无元素符号，POTCAR 不可用）")
    print("       为避免静默使用不安全的大 POTIM，按含 H 体系处理（安全默认）")
    print("       如需覆盖，请在 recipe.yaml 中设置 simulation.has_h: true/false")
    return (True, 'unknown_safe_default')


# =============================================================================
# Langevin Gamma Vector
# =============================================================================

def build_gamma_vector(
    sim: Dict,
    ntyp: int,
    elements: Optional[List[str]],
    stage_name: Optional[str] = None
) -> List[float]:
    """
    Build LANGEVIN_GAMMA vector of length NTYP.
    
    Priority order (stage-specific first, then global):
    1. gamma_list_1ps / gamma_list_{stage}_1ps / gamma_{stage}_list_1ps (list)
    2. gamma_by_element_1ps / gamma_by_element_{stage}_1ps / gamma_{stage}_by_element_1ps (dict)
    3. gamma_1ps / gamma_{stage}_1ps (scalar → replicate)
    
    Exits with hard error if cannot construct valid vector.
    """
    def _stage_keys(kind: str) -> List[str]:
        if not stage_name:
            return []
        if kind == 'list':
            return [f"gamma_list_{stage_name}_1ps", f"gamma_{stage_name}_list_1ps"]
        if kind == 'by_element':
            return [f"gamma_by_element_{stage_name}_1ps", f"gamma_{stage_name}_by_element_1ps"]
        if kind == 'scalar':
            return [f"gamma_{stage_name}_1ps"]
        return []

    def _find_first(keys: List[str]) -> Tuple[Optional[Any], Optional[str]]:
        for key in keys:
            if key in sim:
                return sim[key], key
        return None, None

    # 1. Try list
    gamma_list_keys = _stage_keys('list') + ['gamma_list_1ps']
    gamma_list, gamma_list_key = _find_first(gamma_list_keys)
    if gamma_list is not None:
        if not isinstance(gamma_list, list):
            print(f"[ERROR] {gamma_list_key} 必须是列表，得到: {type(gamma_list)}")
            sys.exit(1)
        if len(gamma_list) != ntyp:
            print(f"[ERROR] {gamma_list_key} 长度 ({len(gamma_list)}) != NTYP ({ntyp})")
            print(f"        POSCAR 原子类型数: {ntyp}")
            sys.exit(1)
        try:
            gamma_vec = [float(g) for g in gamma_list]
        except (TypeError, ValueError):
            print(f"[ERROR] {gamma_list_key} 包含无法转换为浮点数的值: {gamma_list}")
            sys.exit(1)
        return gamma_vec

    # 2. Try by-element dict
    gamma_by_elem_keys = _stage_keys('by_element') + ['gamma_by_element_1ps']
    gamma_by_elem, gamma_by_elem_key = _find_first(gamma_by_elem_keys)
    if gamma_by_elem is not None:
        if not isinstance(gamma_by_elem, dict):
            print(f"[ERROR] {gamma_by_elem_key} 必须是字典，得到: {type(gamma_by_elem)}")
            sys.exit(1)
        if elements is None:
            print(f"[ERROR] {gamma_by_elem_key} 需要 POSCAR 包含元素符号")
            print("        当前 POSCAR 是 VASP4 格式或缺少元素行")
            print("        请使用 gamma_list_1ps 或 gamma_1ps 替代")
            sys.exit(1)
        gamma_vec = []
        for elem in elements:
            if elem not in gamma_by_elem:
                print(f"[ERROR] {gamma_by_elem_key} 缺少元素 '{elem}'")
                print(f"        POSCAR 元素: {elements}")
                print(f"        提供的元素: {list(gamma_by_elem.keys())}")
                sys.exit(1)
            try:
                gamma_vec.append(float(gamma_by_elem[elem]))
            except (TypeError, ValueError):
                print(f"[ERROR] {gamma_by_elem_key} 中元素 '{elem}' 的值无法转为浮点数")
                sys.exit(1)
        return gamma_vec

    # 3. Try scalar (replicate for all NTYP)
    gamma_scalar_keys = _stage_keys('scalar') + ['gamma_1ps']
    gamma_scalar, gamma_scalar_key = _find_first(gamma_scalar_keys)
    if gamma_scalar is not None:
        try:
            gamma_value = float(gamma_scalar)
        except (TypeError, ValueError):
            print(f"[ERROR] {gamma_scalar_key} 无法转换为浮点数: {gamma_scalar}")
            sys.exit(1)
        return [gamma_value] * ntyp

    # 4. Default
    default_gamma = 10.0 if stage_name == 'eq' else 5.0
    print(f"[INFO] 未指定 gamma，使用默认值 {default_gamma} (所有原子类型)")
    return [default_gamma] * ntyp


def check_gamma_warning(
    gamma_vec: List[float],
    thermostat: str,
    elements: Optional[List[str]],
    allow_high_gamma: bool = False
) -> List[str]:
    """检查 gamma 值是否过大，返回警告列表"""
    if thermostat != 'langevin':
        return []

    if any(g < 0 for g in gamma_vec):
        print(f"[ERROR] LANGEVIN_GAMMA 不能为负数: {gamma_vec}")
        sys.exit(1)

    warnings = []
    max_gamma = max(gamma_vec)

    if max_gamma >= 50:
        if not allow_high_gamma:
            print(f"[ERROR] max(gamma)={max_gamma} 过高，将严重抑制动力学。")
            print("        默认策略为硬错误。若确有需要，请在 recipe.yaml 设置 allow_high_gamma: true")
            sys.exit(1)
        warnings.append(f"[WARN] max(gamma)={max_gamma} 极大，已因 allow_high_gamma=true 放行")
    elif max_gamma >= 20:
        warnings.append(f"[WARN] max(gamma)={max_gamma} 较大，可能抑制真实扩散，建议用于平衡段")
    elif max_gamma >= 10:
        warnings.append(f"[INFO] max(gamma)={max_gamma} 适中，平衡段适用；生产段建议 1-5")

    # Check Li specifically if elements known
    if elements:
        for i, elem in enumerate(elements):
            if elem == 'Li' and gamma_vec[i] >= 10:
                warnings.append(f"[INFO] Li 的 gamma={gamma_vec[i]}，生产段建议 1-5 以获得真实扩散")
                break

    return warnings


# =============================================================================
# ISTART / ICHARG
# =============================================================================

def get_istart_icharg(
    sim: Dict,
    stage_name: Optional[str],
    check_wavecar: bool = True,
    two_stage_generating: bool = False
) -> Tuple[int, int, Optional[str], str]:
    """
    Get ISTART/ICHARG for given stage with smart defaults.
    
    Returns:
        (istart, icharg, warning_msg, resolved_by)
    """
    warning = None
    resolved_by = 'default_cold_start'

    def _find_override(key: str) -> Tuple[Optional[Any], Optional[str]]:
        keys: List[str] = []
        if stage_name == 'eq':
            keys.append(f'{key}_eq')
        elif stage_name == 'prod':
            keys.append(f'{key}_prod')
        keys.append(key)

        for candidate in keys:
            if candidate in sim and sim[candidate] is not None:
                return sim[candidate], candidate
        return None, None

    istart_override, istart_key = _find_override('istart')
    icharg_override, icharg_key = _find_override('icharg')
    has_user_override = istart_key is not None or icharg_key is not None
    
    if stage_name == 'eq':
        istart = istart_override if istart_override is not None else 0
        icharg = icharg_override if icharg_override is not None else 2
        if has_user_override:
            resolved_by = 'user_override'
    elif stage_name == 'prod':
        istart = istart_override
        icharg = icharg_override

        # Deterministic two-stage generation intent:
        # prod INCAR is usually generated before eq run, so WAVECAR check is not valid here.
        if has_user_override:
            resolved_by = 'user_override'
        if istart is None or icharg is None:
            if two_stage_generating:
                istart = istart if istart is not None else 1
                icharg = icharg if icharg is not None else 0
                if not has_user_override:
                    resolved_by = 'two_stage_default'
                warning = "[INFO] two_stage 预生成 prod：默认 ISTART=1, ICHARG=0（期望读取 eq 产生的 WAVECAR）"
            else:
                # Runtime smart default for prod: check WAVECAR in current directory.
                if check_wavecar and os.path.isfile('WAVECAR') and os.path.getsize('WAVECAR') > 0:
                    istart = istart if istart is not None else 1
                    icharg = icharg if icharg is not None else 0
                    if not has_user_override:
                        resolved_by = 'wavecar_present'
                else:
                    if check_wavecar:
                        warning = "[WARN] WAVECAR 不存在或为空，prod 阶段使用 ISTART=0, ICHARG=2"
                    istart = istart if istart is not None else 0
                    icharg = icharg if icharg is not None else 2
                    if not has_user_override:
                        resolved_by = 'wavecar_missing_fallback'
    else:
        istart = istart_override if istart_override is not None else 0
        icharg = icharg_override if icharg_override is not None else 2
        if has_user_override:
            resolved_by = 'user_override'

    try:
        istart_int = int(istart)
        icharg_int = int(icharg)
    except (TypeError, ValueError):
        print(f"[ERROR] ISTART/ICHARG 必须是整数，得到: ISTART={istart}, ICHARG={icharg}")
        sys.exit(1)

    return (istart_int, icharg_int, warning, resolved_by)


# =============================================================================
# Param Resolution
# =============================================================================

def resolve_stage_param(
    sim: Dict[str, Any],
    base_params: Dict[str, Any],
    key: str,
    stage_name: Optional[str],
    default: Any
) -> Any:
    """Resolve parameter with precedence: stage-specific > global > INCAR.base > default."""
    if stage_name:
        stage_key = f"{key}_{stage_name}"
        if stage_key in sim:
            return sim[stage_key]
    if key in sim:
        return sim[key]
    base_key = key.upper()
    if base_key in base_params:
        return base_params[base_key]
    return default


def normalize_vasp_bool(value: Any, key_name: str) -> str:
    """Normalize bool-like value to VASP .TRUE./.FALSE. string."""
    if isinstance(value, bool):
        return '.TRUE.' if value else '.FALSE.'

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return '.TRUE.'
        if value == 0:
            return '.FALSE.'

    if isinstance(value, str):
        text = value.strip().upper()
        if text in {'.TRUE.', 'TRUE', 'T', 'YES', 'Y', '1'}:
            return '.TRUE.'
        if text in {'.FALSE.', 'FALSE', 'F', 'NO', 'N', '0'}:
            return '.FALSE.'

    print(f"[ERROR] {key_name} 需要布尔值（true/false 或 .TRUE./.FALSE.），得到: {value}")
    sys.exit(1)


def normalize_lreal(value: Any, natoms: int) -> str:
    """Normalize LREAL with bool and Auto support."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return normalize_vasp_bool(value, 'LREAL')

    text = str(value).strip()
    if text.upper() == 'AUTO':
        return 'Auto'
    return normalize_vasp_bool(text, 'LREAL')


THERMOSTAT_ALIASES = {
    'langevin': 'langevin',
    'nose_hoover': 'nose_hoover',
    'nose-hoover': 'nose_hoover',
    'nose': 'nose_hoover',
    'nh': 'nose_hoover',
}


def normalize_thermostat(value: Any) -> str:
    """Normalize thermostat aliases to canonical internal names."""
    text = str('langevin' if value is None else value).strip().lower()
    normalized = THERMOSTAT_ALIASES.get(text)
    if normalized is not None:
        return normalized

    print(f"[ERROR] 不支持的 thermostat: '{text}'")
    print("        允许值: 'langevin', 'nose_hoover'")
    print("        支持别名: 'nose', 'nose-hoover', 'nh'")
    sys.exit(1)


def resolve_dt_fs(sim: Dict[str, Any], has_h: bool) -> Tuple[float, str]:
    """Resolve final POTIM in fs with traceable source."""
    dt_fs_val = sim.get('dt_fs', None)
    if dt_fs_val is not None:
        dt_fs = float(dt_fs_val)
        if dt_fs <= 0:
            print(f"[ERROR] simulation.dt_fs 必须 > 0，得到: {dt_fs}")
            sys.exit(1)
        return (dt_fs, 'user')

    if has_h:
        return (1.0, 'auto_has_h')
    return (2.0, 'auto_no_h')


def get_algo_lreal(
    sim: Dict,
    base_params: Dict[str, Any],
    stage_name: Optional[str],
    natoms: int
) -> Tuple[str, str]:
    """
    Get ALGO/LREAL with safer defaults.
    
    - eq stage: ALGO=VeryFast (acceptable for equilibration)
    - prod stage: ALGO=Fast (safer for production dynamics)
    - LREAL: natoms <= 200 → False, else Auto
    """
    default_algo = 'Fast' if stage_name == 'prod' else 'VeryFast'
    default_lreal = '.FALSE.' if natoms <= 200 else 'Auto'

    algo = str(resolve_stage_param(sim, base_params, 'algo', stage_name, default_algo)).strip()
    lreal_raw = resolve_stage_param(sim, base_params, 'lreal', stage_name, default_lreal)
    lreal = normalize_lreal(lreal_raw, natoms)

    return (algo, lreal)


def get_lwave_lcharg(
    sim: Dict[str, Any],
    base_params: Dict[str, Any],
    stage_name: Optional[str],
    two_stage_generating: bool
) -> Tuple[str, str]:
    """
    Resolve LWAVE/LCHARG with precedence:
    stage-specific > global > INCAR.base > defaults.
    """
    if two_stage_generating and stage_name in {'eq', 'prod'}:
        default_lwave = '.TRUE.'
    else:
        default_lwave = '.FALSE.'
    default_lcharg = '.FALSE.'

    lwave_raw = resolve_stage_param(sim, base_params, 'lwave', stage_name, default_lwave)
    lcharg_raw = resolve_stage_param(sim, base_params, 'lcharg', stage_name, default_lcharg)

    lwave = normalize_vasp_bool(lwave_raw, 'LWAVE')
    lcharg = normalize_vasp_bool(lcharg_raw, 'LCHARG')
    return (lwave, lcharg)


def resolve_stage_settings(
    sim: Dict[str, Any],
    base_params: Dict[str, Any],
    poscar_info: Dict[str, Any],
    stage_name: Optional[str],
    has_h: bool,
    h_source: str,
    check_wavecar: bool = True,
    two_stage_generating: bool = False
) -> Dict[str, Any]:
    """Resolve all stage-specific settings once for both file generation and summaries."""
    ntyp = poscar_info['ntyp']
    natoms = poscar_info['natoms']
    elements = poscar_info.get('elements')

    def _fmt_number(value: Any) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else str(value)
        return str(value)

    temp_c = sim.get('temperature_C', 25)
    temp_k = round(temp_c + 273.15, 1)
    dt_fs, dt_fs_resolved_by = resolve_dt_fs(sim, has_h)

    thermostat = normalize_thermostat(sim.get('thermostat', 'langevin'))
    allow_high_gamma = normalize_vasp_bool(sim.get('allow_high_gamma', False), 'allow_high_gamma') == '.TRUE.'

    gamma_vec: List[float] = []
    if thermostat == 'langevin':
        gamma_vec = build_gamma_vector(sim, ntyp, elements, stage_name)
        if len(gamma_vec) != ntyp:
            print(f"[ERROR] LANGEVIN_GAMMA 长度 ({len(gamma_vec)}) != NTYP ({ntyp})")
            sys.exit(1)
    gamma_warnings = check_gamma_warning(gamma_vec, thermostat, elements, allow_high_gamma=allow_high_gamma)

    smass = _coerce_float(resolve_stage_param(sim, base_params, 'smass', stage_name, -3.0), 'smass', 'simulation')
    nelm = _coerce_int(resolve_stage_param(sim, base_params, 'nelm', stage_name, 100), 'nelm', 'simulation')
    ediff = _coerce_float(resolve_stage_param(sim, base_params, 'ediff', stage_name, 1e-5), 'ediff', 'simulation')
    encut_raw = resolve_stage_param(sim, base_params, 'encut', stage_name, None)
    encut = None if encut_raw is None else _coerce_float(encut_raw, 'encut', 'simulation')
    isym = 0
    maxmix = _coerce_int(resolve_stage_param(sim, base_params, 'maxmix', stage_name, 40), 'maxmix', 'simulation')
    nsteps = int(sim.get('nsteps', 10000))

    istart, icharg, istart_warn, istart_icharg_resolved_by = get_istart_icharg(
        sim,
        stage_name,
        check_wavecar=check_wavecar,
        two_stage_generating=two_stage_generating
    )

    algo, lreal = get_algo_lreal(sim, base_params, stage_name, natoms)
    lwave, lcharg = get_lwave_lcharg(sim, base_params, stage_name, two_stage_generating)
    lasph = normalize_vasp_bool(resolve_stage_param(sim, base_params, 'lasph', stage_name, '.TRUE.'), 'LASPH')
    addgrid = normalize_vasp_bool(resolve_stage_param(sim, base_params, 'addgrid', stage_name, '.TRUE.'), 'ADDGRID')
    mdalgo = 3 if thermostat == 'langevin' else 2

    resolved_params: Dict[str, str] = {
        'IBRION': '0',
        'NSW': str(nsteps),
        'POTIM': _fmt_number(dt_fs),
        'TEBEG': _fmt_number(temp_k),
        'TEEND': _fmt_number(temp_k),
        'ISTART': str(istart),
        'ICHARG': str(icharg),
        'NELM': str(nelm),
        'EDIFF': format_ediff(ediff),
        'ISYM': str(isym),
        'MAXMIX': str(maxmix),
        'ALGO': algo,
        'LREAL': lreal,
        'LWAVE': lwave,
        'LCHARG': lcharg,
        'MDALGO': str(mdalgo),
        'LASPH': lasph,
        'ADDGRID': addgrid,
    }
    if encut is not None:
        resolved_params['ENCUT'] = _fmt_number(encut)
    if thermostat == 'langevin':
        resolved_params['LANGEVIN_GAMMA'] = ' '.join(f"{g:.2f}" for g in gamma_vec)
    else:
        resolved_params['SMASS'] = _fmt_number(smass)

    return {
        'temp_c': temp_c,
        'temp_k': temp_k,
        'dt_fs': dt_fs,
        'dt_fs_resolved_by': dt_fs_resolved_by,
        'nsteps': nsteps,
        'thermostat': thermostat,
        'allow_high_gamma': allow_high_gamma,
        'gamma_vec': gamma_vec,
        'gamma_warnings': gamma_warnings,
        'smass': smass,
        'nelm': nelm,
        'ediff': ediff,
        'encut': encut,
        'isym': isym,
        'maxmix': maxmix,
        'istart': istart,
        'icharg': icharg,
        'istart_warn': istart_warn,
        'istart_icharg_resolved_by': istart_icharg_resolved_by,
        'algo': algo,
        'lreal': lreal,
        'lwave': lwave,
        'lcharg': lcharg,
        'lasph': lasph,
        'addgrid': addgrid,
        'mdalgo': mdalgo,
        'ntyp': ntyp,
        'natoms': natoms,
        'elements': elements,
        'has_h': has_h,
        'h_source': h_source,
        'resolved_params': resolved_params,
    }


def emit_resolution_messages(resolved: Dict[str, Any], stage_name: Optional[str]) -> None:
    """Print stage-aware warnings/info resolved from a single code path."""
    stage_prefix = f"[{stage_name.upper()}] " if stage_name else ""

    if resolved['has_h'] and resolved['dt_fs_resolved_by'] == 'auto_has_h':
        print(f"[INFO] {stage_prefix}dt_fs 未指定，使用默认 POTIM=1.0 fs (含 H)")
    if resolved['has_h'] and resolved['dt_fs_resolved_by'] == 'user' and resolved['dt_fs'] > 1.5:
        print(f"[WARN] {stage_prefix}体系含 H 原子，POTIM={resolved['dt_fs']} fs 可能过大，建议 0.5-1.0 fs")

    if resolved['istart_warn']:
        print(resolved['istart_warn'])
    for warn in resolved['gamma_warnings']:
        print(warn)


def print_stage_summary(resolved: Dict[str, Any], stage_name: Optional[str]) -> None:
    """Print summary using the same resolved values written into INCAR."""
    label = stage_name.upper() if stage_name else 'SINGLE'

    print("\n" + "=" * 70)
    print(f"{label} INCAR 摘要")
    print("=" * 70)
    print(f"温度: {resolved['temp_c']} °C = {resolved['temp_k']} K")
    print(f"时间步长: {resolved['dt_fs']} fs")
    print(f"dt_fs resolved_by: {resolved['dt_fs_resolved_by']}")
    print(f"has_h: {resolved['has_h']} (source: {resolved['h_source']})")
    print(f"总步数: {resolved['nsteps']}")
    print(f"NTYP: {resolved['ntyp']}, NATOMS: {resolved['natoms']}")
    print(f"恒温器: {resolved['thermostat']}")
    if resolved['thermostat'] == 'langevin':
        gamma_str = ' '.join(f"{g:.1f}" for g in resolved['gamma_vec'])
        print(f"Langevin gamma: [{gamma_str}] 1/ps")
        print("MDALGO: 3 (Langevin)")
    else:
        print(f"SMASS: {resolved['smass']}")
        print("MDALGO: 2 (Nosé-Hoover)")
    print(
        f"ISTART/ICHARG: {resolved['istart']}/{resolved['icharg']} "
        f"(resolved_by: {resolved['istart_icharg_resolved_by']})"
    )
    print(f"LWAVE/LCHARG: {resolved['lwave']}/{resolved['lcharg']}")
    print(f"ALGO/LREAL: {resolved['algo']}/{resolved['lreal']}")
    print(f"ISYM: {resolved['isym']} (强制关闭)")
    print(f"MAXMIX: {resolved['maxmix']}")
    print("=" * 70)


# =============================================================================
# YAML / INCAR.base Parsing
# =============================================================================

def load_yaml(filepath: str) -> Dict[str, Any]:
    """加载 YAML 文件"""
    if not os.path.isfile(filepath):
        print(f"[ERROR] 配方文件不存在: {filepath}")
        sys.exit(1)

    if not HAS_YAML:
        print("[ERROR] 需要 PyYAML 库: pip install pyyaml")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data if data else {}


def parse_incar_base(filepath: str) -> Dict[str, str]:
    """解析 INCAR.base 文件，支持一行多个 `key=value`（用 ';' 分隔）"""
    params = {}
    if not os.path.isfile(filepath):
        return params

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = _strip_inline_comment(line, strip_leading=True).strip()
            if not line:
                continue

            assignments = [chunk.strip() for chunk in line.split(';') if chunk.strip()]
            for assignment in assignments:
                if '=' not in assignment:
                    continue
                key, value = assignment.split('=', 1)
                key = key.strip().upper()
                value = value.strip()
                if not key:
                    continue
                # INCAR-style "last occurrence wins"
                params[key] = value

    return params


def _coerce_float(value: Any, key: str, context: str) -> float:
    if isinstance(value, bool):
        print(f"[ERROR] {context}.{key} 需要浮点数，不能是布尔值: {value}")
        sys.exit(1)
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"[ERROR] {context}.{key} 需要浮点数，得到: {value}")
        sys.exit(1)


def _coerce_int(value: Any, key: str, context: str) -> int:
    if isinstance(value, bool):
        print(f"[ERROR] {context}.{key} 需要整数，不能是布尔值: {value}")
        sys.exit(1)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        print(f"[ERROR] {context}.{key} 需要整数，得到非整数浮点数: {value}")
        sys.exit(1)

    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            try:
                as_float = float(text)
            except ValueError:
                print(f"[ERROR] {context}.{key} 需要整数，得到: {value}")
                sys.exit(1)
            if as_float.is_integer():
                return int(as_float)
            print(f"[ERROR] {context}.{key} 需要整数，得到非整数值: {value}")
            sys.exit(1)

    print(f"[ERROR] {context}.{key} 需要整数，得到不支持的类型: {type(value)}")
    sys.exit(1)


def validate_and_coerce_simulation(sim: Dict[str, Any], context: str = "simulation") -> Dict[str, Any]:
    """Validate and coerce key simulation parameters for deterministic behavior."""
    if not isinstance(sim, dict):
        print(f"[ERROR] {context} 必须是字典，得到: {type(sim)}")
        sys.exit(1)

    out = dict(sim)

    required_keys = ['temperature_C', 'nsteps']
    for key in required_keys:
        if key not in out:
            print(f"[ERROR] {context} 中缺少 '{key}'")
            sys.exit(1)

    out['temperature_C'] = _coerce_float(out['temperature_C'], 'temperature_C', context)
    out['nsteps'] = _coerce_int(out['nsteps'], 'nsteps', context)
    if out['nsteps'] < 1:
        print(f"[ERROR] {context}.nsteps 必须 >= 1，得到: {out['nsteps']}")
        sys.exit(1)

    optional_float_keys = ['dt_fs', 'smass', 'ediff', 'encut']
    for key in optional_float_keys:
        if key in out and out[key] is not None:
            out[key] = _coerce_float(out[key], key, context)

    optional_int_keys = [
        'nelm', 'maxmix',
        'istart', 'icharg',
        'istart_eq', 'icharg_eq',
        'istart_prod', 'icharg_prod',
        'nsteps_eq'
    ]
    for key in optional_int_keys:
        if key in out and out[key] is not None:
            out[key] = _coerce_int(out[key], key, context)

    if 'dt_fs' in out and out['dt_fs'] is not None and out['dt_fs'] <= 0:
        print(f"[ERROR] {context}.dt_fs 必须 > 0，得到: {out['dt_fs']}")
        sys.exit(1)

    if 'nsteps_eq' in out and out['nsteps_eq'] is not None and out['nsteps_eq'] < 1:
        print(f"[ERROR] {context}.nsteps_eq 必须 >= 1，得到: {out['nsteps_eq']}")
        sys.exit(1)

    return out


def format_ediff(value: float) -> str:
    """格式化 EDIFF"""
    if value >= 1e-4:
        return f"{value:.1E}"
    else:
        return f"{value:.0E}"


# =============================================================================
# INCAR Generation
# =============================================================================

def generate_incar_content(
    sim: Dict,
    base_params: Dict,
    poscar_info: Dict[str, Any],
    stage_name: Optional[str] = None,
    exe: str = None,
    has_h: bool = False,
    h_source: str = 'unknown',
    check_wavecar: bool = True,
    two_stage_generating: bool = False,
    resolved: Optional[Dict[str, Any]] = None
) -> str:
    """生成 INCAR 内容"""
    if resolved is None:
        resolved = resolve_stage_settings(
            sim, base_params, poscar_info, stage_name, has_h, h_source,
            check_wavecar=check_wavecar,
            two_stage_generating=two_stage_generating
        )

    ntyp = resolved['ntyp']
    natoms = resolved['natoms']
    elements = resolved.get('elements')
    temp_c = resolved['temp_c']
    temp_k = resolved['temp_k']
    dt_fs = resolved['dt_fs']
    nsteps = resolved['nsteps']
    thermostat = resolved['thermostat']
    gamma_vec = resolved['gamma_vec']
    smass = resolved['smass']
    isym = resolved['isym']
    maxmix = resolved['maxmix']
    istart = resolved['istart']
    icharg = resolved['icharg']
    algo = resolved['algo']
    lreal = resolved['lreal']
    lwave = resolved['lwave']
    lcharg = resolved['lcharg']
    resolved_params = resolved['resolved_params']

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = []

    # 文件头
    lines.append("# " + "=" * 70)
    lines.append("# INCAR for AIMD - Generated by make_incar_aimd.py v3.0.2")
    lines.append("# " + "=" * 70)
    lines.append(f"# 生成时间: {timestamp}")
    if stage_name:
        lines.append(f"# 阶段: {stage_name.upper()}")
    if stage_name == 'prod' and two_stage_generating:
        lines.append("# NOTE: expects WAVECAR from eq stage; if missing, set ISTART=0/ICHARG=2.")
    lines.append(f"# has_h={resolved['has_h']} source={h_source}")
    lines.append(f"# dt_fs resolved_by={resolved['dt_fs_resolved_by']}")
    lines.append(f"# ISTART/ICHARG resolved_by={resolved['istart_icharg_resolved_by']}")
    lines.append(f"# 温度: {temp_c} °C = {temp_k} K")
    lines.append(f"# 时间步长: {dt_fs} fs")
    lines.append(f"# 总步数: {nsteps}")
    lines.append(f"# NTYP: {ntyp}, NATOMS: {natoms}")
    if elements:
        lines.append(f"# 元素: {' '.join(elements)}")
    lines.append(f"# 恒温器: {thermostat}")
    if thermostat == 'langevin':
        gamma_str = ' '.join(f"{g:.1f}" for g in gamma_vec)
        lines.append(f"# Langevin gamma: [{gamma_str}] 1/ps")
        if max(gamma_vec) >= 10:
            lines.append(f"# ⚠️ max(gamma) 较大，适合平衡段；生产段建议 1-5")
    else:
        lines.append(f"# SMASS: {resolved_params['SMASS']}")
    lines.append(f"# ISYM: {isym} (强制关闭)")
    lines.append(f"# MAXMIX: {maxmix}")
    lines.append(f"# ALGO: {algo}, LREAL: {lreal}")
    lines.append(f"# LWAVE: {lwave}, LCHARG: {lcharg}")
    lines.append(f"# ISTART: {istart}, ICHARG: {icharg}")
    if exe:
        lines.append(f"# 可执行文件: {exe}")
    lines.append("# " + "=" * 70)
    lines.append("")

    managed_keys = {
        'IBRION', 'NSW', 'POTIM', 'TEBEG', 'TEEND', 'SMASS',
        'MDALGO', 'LANGEVIN_GAMMA', 'LWAVE', 'LCHARG',
        'NELM', 'EDIFF', 'ISTART', 'ICHARG', 'ISYM', 'MAXMIX',
        'ALGO', 'LREAL', 'ENCUT', 'LASPH', 'ADDGRID'
    }

    # 继承参数（剔除受脚本托管的键）
    if base_params:
        lines.append("# ============ 继承自 INCAR.base ============")
        for key, value in base_params.items():
            if key.upper() not in managed_keys:
                lines.append(f"{key} = {value}")
        lines.append("")

    # 基础参数
    lines.append("# ============ 基础参数 ============")
    if 'ENCUT' in resolved_params:
        lines.append(f"ENCUT = {resolved_params['ENCUT']}")
    else:
        lines.append("# ENCUT = 400  # 请根据 POTCAR 设置")
    lines.append(f"ALGO = {resolved_params['ALGO']}")
    if not base_params:
        lines.append("PREC = Normal")
        lines.append("ISMEAR = 0")
        lines.append("SIGMA = 0.05")
    lines.append("")

    # 初始化
    lines.append("# ============ 初始化 ============")
    lines.append(f"ISTART = {resolved_params['ISTART']}")
    lines.append(f"ICHARG = {resolved_params['ICHARG']}")
    lines.append("")

    # 电子步
    lines.append("# ============ 电子步收敛 ============")
    lines.append(f"NELM = {resolved_params['NELM']}")
    lines.append(f"EDIFF = {resolved_params['EDIFF']}")
    lines.append(f"MAXMIX = {resolved_params['MAXMIX']}   # 电荷混合历史，建议 40-80")
    lines.append("")

    # 对称性
    lines.append("# ============ 对称性 (AIMD 必须关闭) ============")
    lines.append(f"ISYM = {resolved_params['ISYM']}")
    lines.append("")

    # MD 参数
    lines.append("# ============ 分子动力学 ============")
    lines.append(f"IBRION = {resolved_params['IBRION']}   # MD 模式")
    lines.append(f"NSW = {resolved_params['NSW']}")
    lines.append(f"POTIM = {resolved_params['POTIM']}   # 时间步长 (fs)")
    lines.append("")

    # 温度
    lines.append("# ============ 温度控制 ============")
    lines.append(f"TEBEG = {resolved_params['TEBEG']}")
    lines.append(f"TEEND = {resolved_params['TEEND']}")
    lines.append("")

    # 恒温器
    lines.append("# ============ 恒温器 ============")
    if thermostat == 'langevin':
        lines.append("MDALGO = 3   # Langevin thermostat")
        lines.append(f"LANGEVIN_GAMMA = {resolved_params['LANGEVIN_GAMMA']}  # 摩擦系数 (1/ps), 每种原子类型一个值")
        lines.append(f"# NTYP={ntyp}, gamma 向量长度必须等于 NTYP")
        if max(gamma_vec) >= 10:
            lines.append("# ⚠️ 当前 gamma 较大，适合平衡/控温；")
            lines.append("#    若用于扩散计算，建议减小到 1-5")
    else:
        lines.append("MDALGO = 2   # Nosé-Hoover thermostat")
        lines.append(f"SMASS = {resolved_params['SMASS']}")
    lines.append("")

    # 推荐参数
    lines.append("# ============ 推荐参数 ============")
    lines.append(f"LASPH = {resolved_params['LASPH']}")
    lines.append(f"ADDGRID = {resolved_params['ADDGRID']}")
    lines.append("")

    # 输出控制
    lines.append("# ============ 输出控制 ============")
    lines.append(f"LWAVE = {resolved_params['LWAVE']}")
    lines.append(f"LCHARG = {resolved_params['LCHARG']}")
    lines.append(f"LREAL = {resolved_params['LREAL']}  # natoms={natoms}, 阈值 200")
    lines.append("")

    # 性能
    lines.append("# ============ 性能优化 ============")
    lines.append("# NCORE = 4")
    lines.append("# KPAR = 1")
    lines.append("")

    # 结尾
    lines.append("# " + "=" * 70)
    if stage_name == 'eq':
        lines.append("# 平衡段：用于系统温度稳定，不用于动力学分析")
    elif stage_name == 'prod':
        lines.append("# 生产段：用于轨迹采样和扩散分析")
    lines.append("# 运行: NP=16 EXE=vasp_std run_vasp.sh")
    lines.append("# 监控: aimd_watch.sh")
    lines.append("# " + "=" * 70)

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="根据 recipe.yaml 生成 AIMD INCAR (v3.0.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python3 make_incar_aimd.py --out INCAR
    python3 make_incar_aimd.py --two_stage  # 生成 INCAR.eq 和 INCAR.prod

两段式说明:
    - INCAR.eq: 平衡段，gamma 较大（默认 10），ISTART=0
    - INCAR.prod: 生产段，gamma 较小（默认 5），默认重启意图 ISTART=1, ICHARG=0
    - 平衡段结束后，将 CONTCAR 复制为 POSCAR，保留/复制 WAVECAR 后再运行生产段

v3.0.2 新特性:
    - 修复恒温器 MDALGO 映射：Langevin=3, Nosé-Hoover=2
    - LANGEVIN_GAMMA 输出向量（每种原子类型一个值）
    - dt_fs 可选（未指定时根据 H 检测自动设置）
    - ISTART/ICHARG 区分平衡/生产段
    - H 检测未知时按安全默认处理 + 头部 traceability 注释
    - 支持 thermostat 别名: nose / nose-hoover / nh
    - 支持 gamma_list_1ps / gamma_by_element_1ps
        """
    )
    parser.add_argument("--recipe", default="recipe.yaml",
                        help="配方文件路径 (默认: recipe.yaml)")
    parser.add_argument("--out", default="INCAR.aimd",
                        help="输出 INCAR 文件路径 (默认: INCAR.aimd)")
    parser.add_argument("--exe", default=None,
                        help="可执行文件名 (可选，仅记录到注释)")
    parser.add_argument("--two_stage", action="store_true",
                        help="生成两段式 INCAR (INCAR.eq 和 INCAR.prod)")
    parser.add_argument("--poscar", default="POSCAR",
                        help="POSCAR 文件路径，用于检测原子类型和是否含 H")
    parser.add_argument("--potcar", default="POTCAR",
                        help="POTCAR 文件路径，用于 H 检测回退 (默认: POTCAR)")

    args = parser.parse_args()

    print("=" * 70)
    print("make_incar_aimd.py - AIMD INCAR 生成器 (v3.0.2)")
    print("=" * 70)
    print(f"配方文件: {args.recipe}")
    
    if args.two_stage:
        print("模式: 两段式 (eq + prod)")
    else:
        print(f"输出文件: {args.out}")

    # 加载配方
    data = load_yaml(args.recipe)
    sim = data.get('simulation', None)
    if sim is None:
        print("[ERROR] recipe.yaml 中未找到 'simulation' 段")
        sys.exit(1)
    sim = validate_and_coerce_simulation(sim, context='simulation')

    # 解析 POSCAR
    print(f"\n>>> 解析 POSCAR: {args.poscar}")
    poscar_info = parse_poscar_header(args.poscar)
    
    # 检测是否含 H
    yaml_has_h = sim.get('has_h', None)
    has_h, h_source = detect_hydrogen(poscar_info, args.potcar, yaml_has_h)
    
    # Print POSCAR summary (self-check G)
    print("\n" + "=" * 30)
    print("POSCAR 解析摘要")
    print("=" * 30)
    print(f"NTYP: {poscar_info['ntyp']}")
    print(f"NATOMS: {poscar_info['natoms']}")
    if poscar_info['elements']:
        print(f"元素: {' '.join(poscar_info['elements'])}")
    else:
        print("元素: [未知 - VASP4 格式]")
    print(f"含 H: {has_h} (来源: {h_source})")
    print("=" * 30)

    # 读取 INCAR.base
    base_params = {}
    if os.path.isfile('INCAR.base'):
        print("\n>>> 检测到 INCAR.base，将继承其参数...")
        base_params = parse_incar_base('INCAR.base')
        print(f"    读取 {len(base_params)} 个参数")
    else:
        print("\n>>> 未找到 INCAR.base，将生成完整模板")

    # 生成 INCAR
    if args.two_stage:
        # 两段式
        stages = sim.get('stages', None)
        
        if stages:
            # 使用 recipe 中定义的阶段
            for stage in stages:
                stage_name = stage.get('name', 'unknown')
                stage_sim = sim.copy()
                stage_sim.update(stage)
                stage_sim = validate_and_coerce_simulation(stage_sim, context=f"simulation.stages[{stage_name}]")
                resolved = resolve_stage_settings(
                    stage_sim, base_params, poscar_info,
                    stage_name, has_h, h_source,
                    check_wavecar=False,
                    two_stage_generating=True
                )
                emit_resolution_messages(resolved, stage_name)

                content = generate_incar_content(
                    stage_sim, base_params, poscar_info,
                    stage_name, args.exe, has_h, h_source,
                    check_wavecar=False,
                    two_stage_generating=True,
                    resolved=resolved
                )
                outfile = f"INCAR.{stage_name}"
                with open(outfile, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"\n>>> 已写入: {outfile}")
                print_stage_summary(resolved, stage_name)
            # DO NOT write args.out in two_stage mode with stages
        else:
            # 默认两段
            # 平衡段
            eq_sim = sim.copy()
            inferred_nsteps_eq = sim.get('nsteps', 10000) // 5
            if inferred_nsteps_eq < 1:
                print(f"[WARN] nsteps//5 = {inferred_nsteps_eq} < 1，自动提升为 nsteps_eq=1")
            eq_sim['nsteps'] = sim.get('nsteps_eq', max(1, inferred_nsteps_eq))
            eq_sim = validate_and_coerce_simulation(eq_sim, context='simulation(eq)')
            eq_resolved = resolve_stage_settings(
                eq_sim, base_params, poscar_info,
                'eq', has_h, h_source,
                check_wavecar=False,
                two_stage_generating=True
            )
            emit_resolution_messages(eq_resolved, 'eq')

            eq_content = generate_incar_content(
                eq_sim, base_params, poscar_info,
                'eq', args.exe, has_h, h_source,
                check_wavecar=False,
                two_stage_generating=True,
                resolved=eq_resolved
            )
            with open('INCAR.eq', 'w', encoding='utf-8') as f:
                f.write(eq_content)
            print("\n>>> 已写入: INCAR.eq (平衡段)")
            print_stage_summary(eq_resolved, 'eq')

            # 生产段
            prod_sim = sim.copy()
            prod_sim = validate_and_coerce_simulation(prod_sim, context='simulation(prod)')
            prod_resolved = resolve_stage_settings(
                prod_sim, base_params, poscar_info,
                'prod', has_h, h_source,
                check_wavecar=False,
                two_stage_generating=True
            )
            emit_resolution_messages(prod_resolved, 'prod')

            prod_content = generate_incar_content(
                prod_sim, base_params, poscar_info,
                'prod', args.exe, has_h, h_source,
                check_wavecar=False,
                two_stage_generating=True,
                resolved=prod_resolved
            )
            with open('INCAR.prod', 'w', encoding='utf-8') as f:
                f.write(prod_content)
            print(">>> 已写入: INCAR.prod (生产段)")
            print_stage_summary(prod_resolved, 'prod')

            print("\n两段式用法:")
            print("  1. cp INCAR.eq INCAR && run_vasp.sh")
            print("  2. cp CONTCAR POSCAR")
            print("  3. 保留/复制平衡段产生的 WAVECAR 到生产段目录")
            print("     (若需 ICHARG=1，还需保留 CHGCAR)")
            print("  4. cp INCAR.prod INCAR && run_vasp.sh")
            print("  5. 若无 WAVECAR，请将 INCAR.prod 改为 ISTART=0, ICHARG=2")
            print("  6. 扩散分析只用生产段轨迹")
            # DO NOT write args.out in two_stage mode
    else:
        # 单一 INCAR
        resolved = resolve_stage_settings(
            sim, base_params, poscar_info,
            None, has_h, h_source
        )
        emit_resolution_messages(resolved, None)
        content = generate_incar_content(
            sim, base_params, poscar_info,
            None, args.exe, has_h, h_source,
            resolved=resolved
        )
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n>>> 已写入: {args.out}")
        print_stage_summary(resolved, None)

    print("\n[OK] INCAR 生成完成！")


if __name__ == "__main__":
    main()
