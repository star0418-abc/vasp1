#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recipe_validate.py - 凝胶电解质配方验证工具 v2.0

功能：
  - 读取 recipe.yaml 配方文件
  - 校验所有 wt_pct 非负
  - 校验 8 类总和 = 100%（容差 1e-3）
  - 校验 simulation 段（温度、时间步、步数等）
  - 检测未知类别（WARN 或 ERROR）
  - 双语名称验证（ABBR（中文）或 name_en/name_cn）
  - 离子组配平验证（ion_group + stoich）
  - 按固定顺序打印标准化摘要

用法：
  python3 recipe_validate.py [--recipe recipe.yaml] [--strict_schema]

作者：STAR0418-ABC
"""

import argparse
import sys
import os
import re
from typing import Dict, List, Any, Tuple, Optional

# 尝试导入 yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("[WARN] PyYAML 未安装")
    print("[INFO] 请运行: pip install pyyaml")


# 固定的 8 类顺序
CATEGORY_ORDER = [
    ("salt_solution", "盐溶液"),
    ("polymer_matrix", "聚合物基质"),
    ("crosslinker", "交联剂"),
    ("photoinitiator", "引发剂"),
    ("plasticizer_solvent", "增塑剂/溶剂"),
    ("functional_monomer", "功能单体"),
    ("stabilizer", "稳定剂"),
    ("functional_filler", "功能填料"),
]

# 已知的顶层键（非组分类别）
KNOWN_METADATA_KEYS = {'simulation'}

# 容差
TOLERANCE = 1e-3

# 温度范围（摄氏度）
TEMP_MIN = -50
TEMP_MAX = 300


def load_yaml(filepath: str) -> Dict[str, Any]:
    """加载 YAML 文件"""
    if not os.path.isfile(filepath):
        print(f"[ERROR] 配方文件不存在: {filepath}")
        sys.exit(1)

    if not HAS_YAML:
        print("[ERROR] 需要 PyYAML 库来解析 YAML 文件")
        print("[INFO] 请运行: pip install pyyaml")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return data


def safe_float(value: Any, default: float = 0.0) -> Optional[float]:
    """安全地将值转换为 float"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> Optional[int]:
    """安全地将值转换为 int，拒绝浮点数"""
    if value is None:
        return None
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # 拒绝非整数浮点
        if value != int(value):
            return None  # 表示无效
        return int(value)
    if isinstance(value, str):
        try:
            f = float(value)
            if f != int(f):
                return None
            return int(f)
        except ValueError:
            return default
    return default


def validate_name(entry: Dict, category: str, idx: int) -> Tuple[bool, List[str]]:
    """
    验证双语名称要求
    
    接受以下任一模式：
    1) name 匹配: ASCII 缩写 + 括号内中文 (支持 () 和 （）)
    2) name_en 和 name_cn 都存在且非空
    3) name 包含中文 + 缩写样式标记 + 括号（宽松模式）
    """
    errors = []
    name = str(entry.get('name', ''))
    name_en = str(entry.get('name_en', '')).strip()
    name_cn = str(entry.get('name_cn', '')).strip()
    
    # 检测中文字符
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in name)
    
    # 检测括号
    has_parens = '(' in name or '（' in name
    
    # 检测缩写样式开头（至少2个字母/数字）
    first_part = name.split('(')[0].split('（')[0].strip()
    has_abbr = bool(re.match(r'^[A-Za-z0-9][A-Za-z0-9\-+]+', first_part))
    
    # 模式 1: ABBR（中文全称）或 ABBR(中文全称)
    pattern1_ok = has_abbr and has_parens and has_chinese
    
    # 模式 2: name_en + name_cn 都存在
    pattern2_ok = bool(name_en) and bool(name_cn)
    
    # 模式 3: 宽松模式 - 有中文 + 有括号（适用于纯中文名称）
    pattern3_ok = has_chinese and has_parens
    
    if pattern1_ok or pattern2_ok or pattern3_ok:
        return True, []
    
    errors.append(
        f"  [{category}][{idx}] 名称不符合要求: '{name}'\n"
        f"    接受格式:\n"
        f"      1) 'ABBR（中文全称）' 或 'ABBR(中文全称)'\n"
        f"      2) 同时提供 name_en 和 name_cn 字段\n"
        f"      3) 包含中文的名称带括号说明"
    )
    return False, errors


def validate_entry(entry: Dict, category: str, idx: int) -> Tuple[bool, List[str]]:
    """
    验证单个组分条目

    返回: (是否有效, 错误列表)
    """
    errors = []

    # 必需字段检查
    if 'name' not in entry:
        errors.append(f"  [{category}][{idx}] 缺少 'name' 字段")

    if 'wt_pct' not in entry:
        errors.append(f"  [{category}][{idx}] 缺少 'wt_pct' 字段")
    else:
        wt = entry['wt_pct']
        wt_float = safe_float(wt)
        if wt_float is None:
            errors.append(f"  [{category}][{idx}] 'wt_pct' 必须是数值，当前: {type(wt).__name__}")
        elif wt_float < 0:
            errors.append(f"  [{category}][{idx}] 'wt_pct' 不能为负: {wt}")

    if 'kind' not in entry:
        errors.append(f"  [{category}][{idx}] 缺少 'kind' 字段")

    # 名称验证（双语）
    name_valid, name_errors = validate_name(entry, category, idx)
    errors.extend(name_errors)
    
    # 数值字段类型验证
    mw = entry.get('mw_g_mol')
    if mw is not None:
        mw_float = safe_float(mw)
        if mw_float is None or mw_float <= 0:
            errors.append(f"  [{category}][{idx}] 'mw_g_mol' 必须是正数，当前: {mw}")
    
    atoms = entry.get('atoms_per_entity') or entry.get('atoms_per_molecule')
    if atoms is not None:
        atoms_int = safe_int(atoms)
        if atoms_int is None or atoms_int <= 0:
            errors.append(f"  [{category}][{idx}] 'atoms_per_entity' 必须是正整数，当前: {atoms}")
    
    charge = entry.get('charge')
    if charge is not None:
        charge_int = safe_int(charge)
        if charge_int is None:
            errors.append(f"  [{category}][{idx}] 'charge' 必须是整数，当前: {charge}")
    
    min_count = entry.get('min_count')
    if min_count is not None:
        min_int = safe_int(min_count)
        if min_int is None or min_int < 0:
            errors.append(f"  [{category}][{idx}] 'min_count' 必须是非负整数，当前: {min_count}")

    return len(errors) == 0, errors


def validate_ion_groups(entries: List[Dict]) -> Tuple[bool, List[str]]:
    """
    验证离子组配平（A5）
    
    对于 salt_solution 类别中 charge != 0 的条目：
    - 必须有 ion_group（字符串 ID）
    - stoich 默认为 1
    - 每个 ion_group 的净电荷必须为 0: sum(charge_i * stoich_i) == 0
    """
    errors = []
    groups: Dict[str, List[Dict]] = {}
    ungrouped_charged = []
    
    for entry in entries:
        if entry.get('category') != 'salt_solution':
            continue
        
        charge = entry.get('charge', 0)
        if charge == 0:
            continue
            
        ion_group = entry.get('ion_group')
        if ion_group:
            groups.setdefault(ion_group, []).append(entry)
        else:
            ungrouped_charged.append(entry)
    
    # 检查未分组的带电条目
    for entry in ungrouped_charged:
        name = entry.get('name', 'N/A')
        charge = entry.get('charge', 0)
        errors.append(
            f"  带电 salt_solution 条目 '{name}' (charge={charge}) 必须指定 ion_group。\n"
            f"    解决方案:\n"
            f"      1) 将盐建模为中性实体（如 LiTFSI，charge=0），或\n"
            f"      2) 为拆分离子提供 ion_group 和 stoich 字段"
        )
    
    # 检查每个组的电荷中性
    for group_name, members in groups.items():
        net_charge = sum(
            (m.get('charge', 0) or 0) * (m.get('stoich', 1) or 1) 
            for m in members
        )
        if net_charge != 0:
            member_info = ", ".join([
                f"{m.get('name', 'N/A')}(q={m.get('charge', 0)}, stoich={m.get('stoich', 1)})"
                for m in members
            ])
            errors.append(
                f"  ion_group '{group_name}' 净电荷 = {net_charge} != 0\n"
                f"    成员: {member_info}\n"
                f"    请检查 stoich 值确保电荷平衡"
            )
    
    return len(errors) == 0, errors


def detect_unknown_categories(data: Dict, strict: bool) -> Tuple[bool, List[str]]:
    """
    检测未知的顶层类别（A1）
    
    返回: (是否通过, 消息列表)
    """
    known_keys = {cat[0] for cat in CATEGORY_ORDER} | KNOWN_METADATA_KEYS
    unknown = [k for k in data.keys() if k not in known_keys]
    
    if not unknown:
        return True, []
    
    msg_type = "ERROR" if strict else "WARN"
    messages = [f"[{msg_type}] 发现未知顶层类别: {', '.join(unknown)}"]
    
    if not strict:
        messages.append("  提示: 设置 STRICT_SCHEMA=1 或 --strict_schema 将其视为错误")
    
    return not strict, messages


def validate_simulation(sim: Dict) -> Tuple[bool, List[str], List[str]]:
    """
    验证 simulation 段 (A4: 模式规范化)

    返回: (是否有效, 错误列表, 警告列表)
    """
    errors = []
    warnings = []

    if sim is None:
        return True, [], ["[INFO] 未定义 simulation 段，跳过模拟条件验证"]

    # A4: 模式规范化
    mode = str(sim.get('mode', '')).lower().strip()
    
    # 自动检测模式
    if not mode:
        aimd_keys = {'temperature_C', 'dt_fs', 'nsteps', 'thermostat', 'gamma_1ps', 'ensemble'}
        if any(k in sim for k in aimd_keys):
            mode = 'aimd'
            warnings.append("[INFO] simulation.mode 未指定，检测到 AIMD 相关键，自动设为 'aimd'")
        else:
            mode = 'static'
            warnings.append("[INFO] simulation.mode 未指定，默认为 'static'")
    
    if mode not in ('aimd', 'static'):
        warnings.append(f"[WARN] simulation.mode='{mode}' 非标准值，建议使用 aimd 或 static")

    if mode == 'aimd':
        # 温度检查
        if 'temperature_C' not in sim:
            errors.append("[simulation] AIMD 模式必须指定 temperature_C")
        else:
            temp_c = sim['temperature_C']
            temp_float = safe_float(temp_c)
            if temp_float is None:
                errors.append(f"[simulation] temperature_C 必须是数值，当前: {type(temp_c).__name__}")
            elif temp_float < TEMP_MIN or temp_float > TEMP_MAX:
                warnings.append(f"[simulation] temperature_C={temp_c}°C 超出建议范围 [{TEMP_MIN}, {TEMP_MAX}]")

        # 时间步检查（可选，v3.0 后 dt_fs 可选）
        dt = sim.get('dt_fs', None)
        if dt is not None:
            dt_float = safe_float(dt)
            if dt_float is None or dt_float <= 0:
                errors.append(f"[simulation] dt_fs 必须是正数，当前: {dt}")

        # 步数检查
        nsteps = sim.get('nsteps', None)
        if nsteps is None:
            errors.append("[simulation] AIMD 模式必须指定 nsteps (总步数)")
        else:
            nsteps_int = safe_int(nsteps)
            if nsteps_int is None or nsteps_int <= 0:
                errors.append(f"[simulation] nsteps 必须是正整数，当前: {nsteps}")

        # 系综检查
        ensemble = str(sim.get('ensemble', 'nvt')).lower()
        if ensemble not in ['nvt', 'nve']:
            warnings.append(f"[simulation] ensemble='{ensemble}' 非标准值，建议使用 nvt 或 nve")

        # 恒温器检查
        thermostat = str(sim.get('thermostat', 'langevin')).lower()
        if thermostat not in ['langevin', 'nose_hoover']:
            warnings.append(f"[simulation] thermostat='{thermostat}' 非标准值，建议使用 langevin 或 nose_hoover")

        # gamma 检查
        gamma = sim.get('gamma_1ps', 10.0)
        gamma_float = safe_float(gamma)
        if gamma_float is None or gamma_float <= 0:
            warnings.append(f"[simulation] gamma_1ps 应为正数，当前: {gamma}")

        # nelm 检查
        nelm = sim.get('nelm', 100)
        nelm_int = safe_int(nelm)
        if nelm_int is None or nelm_int <= 0:
            warnings.append(f"[simulation] nelm 应为正整数，当前: {nelm}")

        # ediff 检查
        ediff = sim.get('ediff', 1e-5)
        ediff_float = safe_float(ediff)
        if ediff_float is None or ediff_float <= 0:
            warnings.append(f"[simulation] ediff 应为正数，当前: {ediff}")

    return len(errors) == 0, errors, warnings


def validate_recipe(data: Dict) -> Tuple[bool, float, List[str], List[Dict]]:
    """
    验证整个配方的组分部分

    返回: (是否有效, 总 wt_pct, 错误列表, 展平的条目列表)
    """
    all_errors = []
    total_wt = 0.0
    all_entries = []

    for cat_key, cat_name in CATEGORY_ORDER:
        entries = data.get(cat_key, [])

        # 处理 None 或非列表
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            all_errors.append(f"[{cat_key}] 应为列表类型，当前: {type(entries).__name__}")
            continue

        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                all_errors.append(f"[{cat_key}][{idx}] 条目应为字典类型")
                continue

            valid, errors = validate_entry(entry, cat_key, idx)
            all_errors.extend(errors)

            wt = safe_float(entry.get('wt_pct', 0), 0.0)
            if wt >= 0:
                total_wt += wt
            
            # 添加类别信息
            entry_copy = entry.copy()
            entry_copy['category'] = cat_key
            entry_copy['category_cn'] = cat_name
            all_entries.append(entry_copy)

    return len(all_errors) == 0, total_wt, all_errors, all_entries


def print_summary(data: Dict):
    """按固定顺序打印配方摘要"""
    print("\n" + "=" * 78)
    print("配方摘要 (Recipe Summary)")
    print("=" * 78)

    total_wt = 0.0
    total_entries = 0

    for cat_key, cat_name in CATEGORY_ORDER:
        entries = data.get(cat_key, []) or []

        print(f"\n>>> {cat_key} ({cat_name})")
        print("-" * 78)

        if not entries:
            print("    (空)")
            continue

        print(f"    {'名称':<40} {'wt%':>8} {'种类':<12} {'MW':>10}")
        print("    " + "-" * 74)

        cat_wt = 0.0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get('name', 'N/A'))
            wt = safe_float(entry.get('wt_pct', 0), 0.0)
            kind = str(entry.get('kind', 'N/A'))
            mw = entry.get('mw_g_mol', None)

            # 截断过长的名称
            if len(name) > 38:
                name_display = name[:35] + "..."
            else:
                name_display = name

            # 安全格式化 MW
            mw_float = safe_float(mw)
            mw_str = f"{mw_float:.2f}" if mw_float is not None else "N/A"

            print(f"    {name_display:<40} {wt:>8.2f} {kind:<12} {mw_str:>10}")
            cat_wt += wt
            total_entries += 1

        print(f"    {'小计:':<40} {cat_wt:>8.2f}")
        total_wt += cat_wt

    print("\n" + "=" * 78)
    print(f"总条目数: {total_entries}")
    print(f"总 wt%: {total_wt:.4f}")

    # 检查总和
    diff = abs(total_wt - 100.0)
    if diff <= TOLERANCE:
        print(f"[OK] 总和校验通过 (|{total_wt:.4f} - 100| = {diff:.6f} <= {TOLERANCE})")
    else:
        print(f"[ERROR] 总和校验失败: {total_wt:.4f} != 100 (差值: {diff:.4f})")

    print("=" * 78)


def print_simulation_summary(sim: Dict):
    """打印模拟条件摘要"""
    if sim is None:
        return

    print("\n" + "=" * 78)
    print("模拟条件摘要 (Simulation Settings)")
    print("=" * 78)

    mode = str(sim.get('mode', 'static')).lower().strip() or 'static'
    print(f"模式: {mode}")

    if mode == 'aimd':
        temp_c = sim.get('temperature_C', 'N/A')
        temp_float = safe_float(temp_c)
        if temp_float is not None:
            temp_k = temp_float + 273.15
            print(f"温度: {temp_c} °C = {temp_k:.2f} K")
        else:
            print(f"温度: {temp_c} °C")

        dt = sim.get('dt_fs', 'N/A')
        print(f"时间步长: {dt} fs (POTIM)")

        nsteps = sim.get('nsteps', 'N/A')
        print(f"总步数: {nsteps} (NSW)")

        dt_float = safe_float(dt)
        nsteps_int = safe_int(nsteps) if nsteps != 'N/A' else None
        if dt_float is not None and nsteps_int is not None:
            total_time_ps = dt_float * nsteps_int / 1000.0
            print(f"总模拟时间: {total_time_ps:.2f} ps")

        ensemble = str(sim.get('ensemble', 'nvt')).upper()
        print(f"系综: {ensemble}")

        thermostat = str(sim.get('thermostat', 'langevin'))
        print(f"恒温器: {thermostat}")

        if thermostat.lower() == 'langevin':
            gamma = sim.get('gamma_1ps', 10.0)
            print(f"摩擦系数: {gamma} 1/ps (LANGEVIN_GAMMA)")
        elif thermostat.lower() == 'nose_hoover':
            smass = sim.get('smass', -3)
            print(f"质量参数: SMASS = {smass}")

        nelm = sim.get('nelm', 100)
        ediff = sim.get('ediff', 1e-5)
        ediff_float = safe_float(ediff, 1e-5)
        print(f"电子步: NELM={nelm}, EDIFF={ediff_float:.0e}")

        encut = sim.get('encut', None)
        if encut:
            print(f"截断能: ENCUT = {encut} eV")
        else:
            print("截断能: 未指定 (需在 INCAR.base 或手动设置)")
        
        # 显示 target_atoms（如果存在）
        target_atoms = sim.get('target_atoms')
        if target_atoms:
            print(f"目标原子数: {target_atoms}")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="凝胶电解质配方验证工具 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python3 recipe_validate.py
    python3 recipe_validate.py --recipe my_recipe.yaml
    python3 recipe_validate.py --strict_schema
    STRICT_SCHEMA=1 python3 recipe_validate.py
        """
    )
    parser.add_argument("--recipe", default="recipe.yaml",
                        help="配方文件路径 (默认: recipe.yaml)")
    parser.add_argument("--strict_schema", action="store_true",
                        help="严格模式: 未知类别视为错误")

    args = parser.parse_args()
    
    # 检查环境变量
    strict_schema = args.strict_schema or os.environ.get('STRICT_SCHEMA', '').lower() in ('1', 'true', 'yes')

    print("=" * 78)
    print("recipe_validate.py - 配方验证工具 v2.0")
    print("=" * 78)
    print(f"配方文件: {args.recipe}")
    if strict_schema:
        print("[INFO] 严格模式已启用 (STRICT_SCHEMA)")

    # 加载配方
    data = load_yaml(args.recipe)

    has_error = False

    # A1: 检测未知类别
    print("\n>>> 检测未知类别...")
    unknown_ok, unknown_msgs = detect_unknown_categories(data, strict_schema)
    for msg in unknown_msgs:
        print(msg)
    if not unknown_ok:
        has_error = True

    # 验证组分
    print("\n>>> 验证组分...")
    valid_recipe, total_wt, recipe_errors, all_entries = validate_recipe(data)

    if recipe_errors:
        print("\n[ERROR] 组分验证发现问题:")
        for err in recipe_errors:
            print(err)

    # A5: 验证离子组配平
    print("\n>>> 验证离子组配平...")
    ion_valid, ion_errors = validate_ion_groups(all_entries)
    if ion_errors:
        print("\n[ERROR] 离子组配平验证失败:")
        for err in ion_errors:
            print(err)

    # 验证模拟条件
    print("\n>>> 验证模拟条件...")
    sim = data.get('simulation', None)
    valid_sim, sim_errors, sim_warnings = validate_simulation(sim)

    if sim_errors:
        print("\n[ERROR] 模拟条件验证发现问题:")
        for err in sim_errors:
            print(err)

    if sim_warnings:
        print("\n[WARN] 模拟条件警告:")
        for warn in sim_warnings:
            print(warn)

    # 打印摘要
    print_summary(data)
    print_simulation_summary(sim)

    # 返回状态
    if not valid_recipe:
        print("\n[FAIL] 组分验证失败，请修正上述错误。")
        has_error = True

    diff = abs(total_wt - 100.0)
    if diff > TOLERANCE:
        print(f"\n[FAIL] 总和不等于 100%: {total_wt:.4f}%")
        has_error = True

    if not valid_sim:
        print("\n[FAIL] 模拟条件验证失败，请修正上述错误。")
        has_error = True
    
    if not ion_valid:
        print("\n[FAIL] 离子组配平验证失败，请修正上述错误。")
        has_error = True

    if has_error:
        sys.exit(1)

    print("\n[PASS] 配方验证通过！")
    sys.exit(0)


if __name__ == "__main__":
    main()
