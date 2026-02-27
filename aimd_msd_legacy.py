#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aimd_msd.py - 从 XDATCAR 计算 MSD 并拟合扩散系数（含误差估计）

功能:
    - 计算均方位移 MSD(t)
    - Running diffusion coefficient D(t)
    - 窗口拟合与误差估计（block averaging）
    - 平台判定与警告

用法:
    python3 aimd_msd.py --specie Li --dt_fs 1.0
    python3 aimd_msd.py --specie Li --dt_fs 1.0 --t_skip_ps 2.0 --t_fit_start_ps 5.0 --t_fit_end_ps 10.0

输出:
    - msd_<specie>.dat: t_ps, MSD (Å²)
    - D_running_<specie>.dat: t_ps, D(t) (cm²/s)
    - msd_<specie>.png: MSD 曲线图
    - D_running_<specie>.png: Running-D 曲线图
    - msd_report.txt: 分析报告

依赖:
    pip install numpy matplotlib

作者: STAR0418-ABC
"""

import argparse
import sys
import os
from typing import List, Tuple, Optional

try:
    import numpy as np
except ImportError:
    print("[ERROR] 需要 numpy 库: pip install numpy")
    sys.exit(1)

# 可选 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，跳过绘图")


def parse_xdatcar(filepath: str) -> Tuple[np.ndarray, List[str], List[int], List[np.ndarray]]:
    """解析 XDATCAR 文件"""
    if not os.path.isfile(filepath):
        print(f"[ERROR] 文件不存在: {filepath}")
        sys.exit(1)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 8:
        print(f"[ERROR] XDATCAR 文件格式错误: {filepath}")
        sys.exit(1)

    scale = float(lines[1].strip())
    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)

    frames = []
    idx = 7

    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("Direct") or line.startswith("direct") or "configuration" in line.lower():
            idx += 1
            continue

        if idx + natoms > len(lines):
            break

        frame = np.zeros((natoms, 3))
        valid_frame = True
        for i in range(natoms):
            try:
                coords = lines[idx + i].split()[:3]
                frame[i] = [float(c) for c in coords]
            except (ValueError, IndexError):
                valid_frame = False
                break

        if valid_frame:
            frames.append(frame)
            idx += natoms
        else:
            idx += 1

    if len(frames) == 0:
        print(f"[ERROR] 未能从 XDATCAR 读取任何帧")
        sys.exit(1)

    return lattice, species, counts, frames


def get_species_indices(species: List[str], counts: List[int], target: str) -> np.ndarray:
    """获取目标物种的原子索引"""
    indices = []
    offset = 0
    found = False

    for sp, cnt in zip(species, counts):
        if sp == target:
            indices.extend(range(offset, offset + cnt))
            found = True
        offset += cnt

    if not found:
        print(f"[ERROR] 未找到物种: {target}")
        print(f"[INFO] 可用物种: {species}")
        sys.exit(1)

    return np.array(indices)


def unwrap_trajectory(frames: List[np.ndarray]) -> np.ndarray:
    """分数坐标 unwrapping"""
    nframes = len(frames)
    natoms = frames[0].shape[0]

    unwrapped = np.zeros((nframes, natoms, 3))
    unwrapped[0] = frames[0].copy()

    for t in range(1, nframes):
        d = frames[t] - frames[t - 1]
        d -= np.round(d)
        unwrapped[t] = unwrapped[t - 1] + d

    return unwrapped


def compute_msd(unwrapped: np.ndarray, lattice: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """计算选定原子的 MSD(t)"""
    traj = unwrapped[:, indices, :]
    cart = np.tensordot(traj, lattice, axes=([2], [0]))
    disp = cart - cart[0]
    msd = np.mean(np.sum(disp ** 2, axis=2), axis=1)
    return msd


def compute_running_D(t_ps: np.ndarray, msd: np.ndarray) -> np.ndarray:
    """
    计算 Running diffusion coefficient D(t) = MSD(t) / (6t)
    
    单位: cm²/s (1 Å²/ps = 1e-4 cm²/s)
    """
    D_running = np.zeros_like(msd)
    D_running[0] = 0
    D_running[1:] = msd[1:] / (6.0 * t_ps[1:]) * 1e-4
    return D_running


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    线性最小二乘拟合 y = a + b*x
    
    返回: (intercept, slope, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x * x)
    sxy = np.sum(x * y)
    syy = np.sum(y * y)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return 0.0, 0.0, 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    # R²
    ss_tot = syy - sy * sy / n
    ss_res = syy - 2 * slope * sxy - 2 * intercept * sy + slope * slope * sxx + 2 * slope * intercept * sx + n * intercept * intercept
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return intercept, slope, r_squared


def block_averaging_error(data: np.ndarray, n_blocks: int = 5) -> Tuple[float, float]:
    """
    Block averaging 误差估计
    
    将数据分成 n_blocks 块，计算每块的均值，然后求块间标准差
    
    返回: (mean, std_error)
    """
    n = len(data)
    block_size = n // n_blocks
    
    if block_size < 2:
        return np.mean(data), np.std(data)
    
    block_means = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_means.append(np.mean(data[start:end]))
    
    mean = np.mean(block_means)
    std_error = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
    
    return mean, std_error


def check_plateau(D_running: np.ndarray, t_ps: np.ndarray, 
                  fit_start_idx: int, fit_end_idx: int) -> Tuple[bool, float, str]:
    """
    检查 D(t) 是否形成平台
    
    计算拟合区间内 D(t) 的变化率（斜率相对于平均值）
    
    返回: (has_plateau, drift_ratio, message)
    """
    D_fit = D_running[fit_start_idx:fit_end_idx]
    t_fit = t_ps[fit_start_idx:fit_end_idx]
    
    if len(D_fit) < 3:
        return False, 1.0, "拟合区间太短"
    
    # 计算 D(t) 的线性趋势
    _, slope, _ = linear_fit(t_fit, D_fit)
    mean_D = np.mean(D_fit)
    
    if mean_D <= 0:
        return False, 1.0, "D(t) 均值 ≤ 0"
    
    # 斜率相对于均值的变化率
    time_span = t_fit[-1] - t_fit[0]
    total_drift = abs(slope * time_span)
    drift_ratio = total_drift / mean_D
    
    if drift_ratio < 0.2:
        return True, drift_ratio, "D(t) 形成稳定平台"
    elif drift_ratio < 0.5:
        return True, drift_ratio, f"D(t) 基本稳定，漂移 {drift_ratio*100:.1f}%"
    else:
        return False, drift_ratio, f"D(t) 漂移严重 ({drift_ratio*100:.1f}%)，AIMD 时间可能不足"


def estimate_D_with_error(msd: np.ndarray, t_ps: np.ndarray,
                          fit_start_idx: int, fit_end_idx: int,
                          n_blocks: int = 5) -> Tuple[float, float, float]:
    """
    估算扩散系数及误差
    
    使用窗口拟合 + block averaging
    
    返回: (D_mean, D_std, R²)
    """
    msd_fit = msd[fit_start_idx:fit_end_idx]
    t_fit = t_ps[fit_start_idx:fit_end_idx]
    
    # 全局拟合
    intercept, slope, r2 = linear_fit(t_fit, msd_fit)
    D_global = slope / 6.0 * 1e-4  # Å²/ps -> cm²/s
    
    # Block averaging
    n = len(msd_fit)
    block_size = n // n_blocks
    
    if block_size < 5:
        # 区间太短，无法做 block averaging
        return D_global, 0.0, r2
    
    D_blocks = []
    for i in range(n_blocks):
        start = fit_start_idx + i * block_size
        end = start + block_size
        if end > fit_end_idx:
            break
        _, slope_i, _ = linear_fit(t_ps[start:end], msd[start:end])
        D_blocks.append(slope_i / 6.0 * 1e-4)
    
    D_mean = np.mean(D_blocks)
    D_std = np.std(D_blocks, ddof=1)
    
    return D_mean, D_std, r2


def plot_msd(t_ps: np.ndarray, msd: np.ndarray, specie: str, 
             fit_start_idx: int, fit_end_idx: int, slope: float, intercept: float,
             outdir: str = "."):
    """绘制 MSD 曲线"""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t_ps, msd, 'b-', linewidth=1, label='MSD')
    
    # 拟合线
    t_fit = t_ps[fit_start_idx:fit_end_idx]
    msd_fit = intercept + slope * t_fit
    ax.plot(t_fit, msd_fit, 'r--', linewidth=2, label=f'Linear fit (slope={slope:.4f})')
    
    # 拟合区间标记
    ax.axvline(x=t_ps[fit_start_idx], color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=t_ps[fit_end_idx-1], color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('MSD (Å²)')
    ax.set_title(f'MSD of {specie}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'msd_{specie}.png'), dpi=150)
    plt.close()


def plot_running_D(t_ps: np.ndarray, D_running: np.ndarray, specie: str,
                   fit_start_idx: int, fit_end_idx: int, D_mean: float, D_std: float,
                   outdir: str = "."):
    """绘制 Running-D 曲线"""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 跳过 t=0
    ax.plot(t_ps[1:], D_running[1:], 'b-', linewidth=1, label='D(t)')
    
    # 平均值线
    ax.axhline(y=D_mean, color='r', linestyle='--', linewidth=2, 
               label=f'D = ({D_mean:.2e} ± {D_std:.2e}) cm²/s')
    
    # 误差带
    if D_std > 0:
        ax.axhspan(D_mean - D_std, D_mean + D_std, alpha=0.2, color='red')
    
    # 拟合区间
    ax.axvline(x=t_ps[fit_start_idx], color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=t_ps[fit_end_idx-1], color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('D(t) (cm²/s)')
    ax.set_title(f'Running Diffusion Coefficient of {specie}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'D_running_{specie}.png'), dpi=150)
    plt.close()


def write_report(specie: str, nframes: int, dt_fs: float, t_total_ps: float,
                 t_fit_start: float, t_fit_end: float,
                 D_mean: float, D_std: float, r2: float,
                 has_plateau: bool, drift_ratio: float, plateau_msg: str,
                 outdir: str = "."):
    """写入分析报告"""
    report_path = os.path.join(outdir, 'msd_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MSD/扩散系数分析报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"物种: {specie}\n")
        f.write(f"总帧数: {nframes}\n")
        f.write(f"时间步长: {dt_fs} fs\n")
        f.write(f"总模拟时间: {t_total_ps:.3f} ps\n\n")
        
        f.write("--- 拟合参数 ---\n")
        f.write(f"拟合区间: {t_fit_start:.3f} ~ {t_fit_end:.3f} ps\n")
        f.write(f"R²: {r2:.6f}\n\n")
        
        f.write("--- 扩散系数 ---\n")
        f.write(f"D = ({D_mean:.6e} ± {D_std:.6e}) cm²/s\n")
        f.write(f"D = {D_mean * 1e4:.6e} Å²/ps\n\n")
        
        f.write("--- 平台判定 ---\n")
        f.write(f"状态: {'✓ ' if has_plateau else '✗ '}{plateau_msg}\n")
        f.write(f"漂移率: {drift_ratio*100:.1f}%\n\n")
        
        if not has_plateau:
            f.write("!" * 70 + "\n")
            f.write("!!! 警告: D(t) 未形成稳定平台 !!!\n")
            f.write("!!! AIMD 模拟时间可能不足，扩散系数仅供参考 !!!\n")
            f.write("!!! 建议: 延长模拟时间或增加采样 !!!\n")
            f.write("!" * 70 + "\n")
        
        f.write("\n--- 输出文件 ---\n")
        f.write(f"msd_{specie}.dat: MSD 数据\n")
        f.write(f"D_running_{specie}.dat: Running-D 数据\n")
        f.write(f"msd_{specie}.png: MSD 图\n")
        f.write(f"D_running_{specie}.png: Running-D 图\n")
        f.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="从 XDATCAR 计算 MSD 并拟合扩散系数（含误差估计）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python3 aimd_msd.py --specie Li --dt_fs 1.0
    python3 aimd_msd.py --specie Li --dt_fs 1.0 --t_skip_ps 2.0 --t_fit_start_ps 5.0

注意:
    - AIMD 时间尺度有限（ps 级），扩散系数仅供趋势参考
    - 长程扩散（ns 级）请使用经典 MD
    - 若 D(t) 无平台，结果可靠性低
        """
    )
    parser.add_argument("--xdatcar", default="XDATCAR",
                        help="XDATCAR 文件路径 (默认: XDATCAR)")
    parser.add_argument("--specie", required=True,
                        help="目标物种符号，如 Li/Zn/K/Al")
    parser.add_argument("--dt_fs", type=float, required=True,
                        help="时间步长 POTIM (fs)")
    
    # 跳过参数
    parser.add_argument("--skip", type=int, default=0,
                        help="跳过前 N 帧 (默认: 0)")
    parser.add_argument("--t_skip_ps", type=float, default=1.0,
                        help="跳过初始弹道区时间 (ps, 默认: 1.0)")
    
    # 拟合区间
    parser.add_argument("--t_fit_start_ps", type=float, default=None,
                        help="拟合起始时间 (ps, 默认: 总时间的 30%%)")
    parser.add_argument("--t_fit_end_ps", type=float, default=None,
                        help="拟合终止时间 (ps, 默认: 总时间的 90%%)")
    parser.add_argument("--fit_start", type=float, default=0.3,
                        help="拟合起始位置 (0~1 比例, 默认: 0.3)")
    parser.add_argument("--fit_end", type=float, default=0.9,
                        help="拟合终止位置 (0~1 比例, 默认: 0.9)")
    
    # 误差估计
    parser.add_argument("--n_blocks", type=int, default=5,
                        help="Block averaging 分块数 (默认: 5)")
    
    # 输出
    parser.add_argument("--outdir", default=".",
                        help="输出目录 (默认: 当前目录)")

    args = parser.parse_args()

    print("=" * 70)
    print("aimd_msd.py - MSD 计算与扩散系数拟合 (v2.0)")
    print("=" * 70)
    print(f"XDATCAR: {args.xdatcar}")
    print(f"物种: {args.specie}")
    print(f"dt: {args.dt_fs} fs")
    print("=" * 70)

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    # 1. 读取 XDATCAR
    print("\n>>> 读取 XDATCAR...")
    lattice, species, counts, frames = parse_xdatcar(args.xdatcar)
    print(f"    物种: {species}")
    print(f"    原子数: {counts} (总计 {sum(counts)})")
    print(f"    总帧数: {len(frames)}")

    # 2. 跳过初始帧
    if args.skip >= len(frames):
        print(f"[ERROR] skip ({args.skip}) >= 总帧数 ({len(frames)})")
        sys.exit(1)

    frames = frames[args.skip:]
    nframes = len(frames)
    print(f"    有效帧数: {nframes} (跳过 {args.skip})")

    # 3. 获取目标物种索引
    print(f"\n>>> 查找物种 {args.specie}...")
    indices = get_species_indices(species, counts, args.specie)
    print(f"    找到 {len(indices)} 个 {args.specie} 原子")

    # 4. Unwrap 轨迹
    print("\n>>> Unwrapping 轨迹...")
    unwrapped = unwrap_trajectory(frames)

    # 5. 计算 MSD
    print(">>> 计算 MSD...")
    msd = compute_msd(unwrapped, lattice, indices)

    # 6. 生成时间轴
    dt_ps = args.dt_fs / 1000.0
    t_ps = np.arange(nframes) * dt_ps
    t_total_ps = t_ps[-1]

    # 7. 计算 Running-D
    print(">>> 计算 Running-D...")
    D_running = compute_running_D(t_ps, msd)

    # 8. 确定拟合区间
    print("\n>>> 确定拟合区间...")
    
    # 跳过弹道区
    skip_idx = int(args.t_skip_ps / dt_ps)
    skip_idx = max(1, min(skip_idx, nframes // 4))
    
    # 拟合起点
    if args.t_fit_start_ps is not None:
        fit_start_idx = int(args.t_fit_start_ps / dt_ps)
    else:
        fit_start_idx = int(nframes * args.fit_start)
    fit_start_idx = max(skip_idx, fit_start_idx)
    
    # 拟合终点
    if args.t_fit_end_ps is not None:
        fit_end_idx = int(args.t_fit_end_ps / dt_ps)
    else:
        fit_end_idx = int(nframes * args.fit_end)
    fit_end_idx = min(nframes, fit_end_idx)
    
    if fit_end_idx <= fit_start_idx + 5:
        print("[WARN] 拟合区间太短，调整参数")
        fit_start_idx = max(skip_idx, nframes // 4)
        fit_end_idx = min(nframes, nframes * 9 // 10)
    
    print(f"    弹道区跳过: {t_ps[skip_idx]:.3f} ps")
    print(f"    拟合区间: {t_ps[fit_start_idx]:.3f} ~ {t_ps[fit_end_idx-1]:.3f} ps")
    print(f"    拟合点数: {fit_end_idx - fit_start_idx}")

    # 9. 平台判定
    print("\n>>> 平台判定...")
    has_plateau, drift_ratio, plateau_msg = check_plateau(D_running, t_ps, fit_start_idx, fit_end_idx)
    print(f"    {plateau_msg}")
    
    if not has_plateau:
        print("")
        print("!" * 70)
        print("!!! 警告: D(t) 未形成稳定平台 !!!")
        print("!!! AIMD 模拟时间可能不足，扩散系数仅供参考 !!!")
        print("!" * 70)

    # 10. 扩散系数估算
    print("\n>>> 估算扩散系数...")
    D_mean, D_std, r2 = estimate_D_with_error(msd, t_ps, fit_start_idx, fit_end_idx, args.n_blocks)
    
    # 也计算全局线性拟合用于绘图
    t_fit = t_ps[fit_start_idx:fit_end_idx]
    msd_fit = msd[fit_start_idx:fit_end_idx]
    intercept, slope, _ = linear_fit(t_fit, msd_fit)

    # 11. 保存数据
    print("\n>>> 保存数据...")
    
    # MSD 数据
    msd_path = os.path.join(args.outdir, f"msd_{args.specie}.dat")
    with open(msd_path, 'w') as f:
        f.write("# t_ps  MSD_A2\n")
        for t, m in zip(t_ps, msd):
            f.write(f"{t:.6f}  {m:.6f}\n")
    print(f"    [OK] {msd_path}")
    
    # Running-D 数据
    D_path = os.path.join(args.outdir, f"D_running_{args.specie}.dat")
    with open(D_path, 'w') as f:
        f.write("# t_ps  D_cm2_s\n")
        for t, d in zip(t_ps[1:], D_running[1:]):
            f.write(f"{t:.6f}  {d:.6e}\n")
    print(f"    [OK] {D_path}")

    # 12. 绘图
    print("\n>>> 绘图...")
    plot_msd(t_ps, msd, args.specie, fit_start_idx, fit_end_idx, slope, intercept, args.outdir)
    plot_running_D(t_ps, D_running, args.specie, fit_start_idx, fit_end_idx, D_mean, D_std, args.outdir)
    if HAS_MPL:
        print(f"    [OK] msd_{args.specie}.png")
        print(f"    [OK] D_running_{args.specie}.png")

    # 13. 写入报告
    write_report(args.specie, nframes, args.dt_fs, t_total_ps,
                 t_ps[fit_start_idx], t_ps[fit_end_idx-1],
                 D_mean, D_std, r2, has_plateau, drift_ratio, plateau_msg,
                 args.outdir)
    print(f"    [OK] msd_report.txt")

    # 14. 输出摘要
    print("\n" + "=" * 70)
    print("计算结果摘要")
    print("=" * 70)
    print(f"总帧数: {nframes}")
    print(f"时间步长 dt: {args.dt_fs} fs = {dt_ps} ps")
    print(f"总模拟时间: {t_total_ps:.3f} ps")
    print(f"MSD 末值: {msd[-1]:.4f} Å²")
    print(f"拟合区间: {t_ps[fit_start_idx]:.3f} ~ {t_ps[fit_end_idx-1]:.3f} ps")
    print(f"拟合 R²: {r2:.6f}")
    print("-" * 70)
    print(f"扩散系数 D = ({D_mean:.4e} ± {D_std:.4e}) cm²/s")
    print("-" * 70)
    print(f"平台判定: {'✓ ' if has_plateau else '✗ '}{plateau_msg}")
    print("=" * 70)
    
    if not has_plateau:
        print("\n[WARN] D(t) 无稳定平台，结果仅供趋势参考！")
        print("[INFO] 建议: 延长 AIMD 模拟时间")


if __name__ == "__main__":
    main()
