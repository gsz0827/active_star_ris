# STAR-RIS 速度与论文输出补丁

此补丁面向仓库 `gsz0827/active_star_ris` 当前 `main` 分支，保留论文配置中的探测样本数、鲁棒蒙特卡洛样本数和 TD3 训练步数。

## 修改内容

1. 每个动作只执行一次动作解码和鲁棒功率投影，不再在每个蒙特卡洛样本内重复。
2. 用 Numba 编译块内 Gauss-Markov 时间递推；随机数顺序和统计模型保持不变。
3. 在最终汇总中保存协调后 KDR 的均值、CVaR 和最差样本值。
4. 新建 `full_scheme_v2_paper_corrected.yaml`，将训练阶段实际不可用的“协调后 KDR 权重”设为 0；原始 KDR、互易性、安全密钥率和功耗目标保持不变。
5. 新增安全的架构-种子进程级并行脚本、真实性能基准脚本、扩展论文绘图脚本和 VS Code 启动配置。

## 应用补丁

把本压缩包解压到任意目录，在仓库根目录执行：

```powershell
python 路径\apply_speed_quality_patch.py --repo .
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_all_checks_v2.py
python -m pytest tests/test_speed_quality_patch_v2.py -q
```

应用前，脚本会把被修改文件复制到：

```text
.star_ris_patch_backup/<时间戳>/
```

## VS Code 真实性能测试

```powershell
python scripts/benchmark_full_scheme_v2.py `
  --config configs/full_scheme_v2_paper_corrected.yaml `
  --architecture partially_active_fixed `
  --steps 10
```

首次调用包含 Numba JIT 编译，计时前脚本会自动预热。

## 单任务预实验

```powershell
python scripts/train_full_scheme_v2.py `
  --config configs/full_scheme_v2_paper_corrected.yaml `
  --architecture partially_active_fixed `
  --seed 0 `
  --steps 5000 `
  --output-dir results/full_scheme_v2/pilot_single/partially_active_fixed/seed_0
```

## 两进程并行预实验

```powershell
python scripts/run_parallel_full_scheme_v2.py `
  --config configs/full_scheme_v2_paper_corrected.yaml `
  --steps 5000 `
  --episodes 10 `
  --seeds 0 1 `
  --architectures passive partially_active_fixed partially_active_dynamic fully_active_fixed `
  --objective-samples 16 `
  --final-probing-samples 256 `
  --max-workers 2 `
  --skip-completed `
  --output-dir results/full_scheme_v2/pilot_parallel
```

不要一开始把 `--max-workers` 设得很大。先用 2；观察 CPU、内存和 GPU 显存后再增加。

## 正式论文实验

```powershell
python scripts/run_parallel_full_scheme_v2.py `
  --config configs/full_scheme_v2_paper_corrected.yaml `
  --max-workers 2 `
  --skip-completed `
  --output-dir results/full_scheme_v2/paper_parallel
```

## 绘图

```powershell
python scripts/plot_full_scheme_v2_extended.py `
  --results-dir results/full_scheme_v2/paper_parallel
```

每张图同时输出 PDF 矢量文件和 600 dpi PNG 文件。

## 论文表述注意

训练阶段没有实际执行 Cascade 协调，因此不要写成“TD3 直接优化实际协调后 KDR”。更严谨的表述是：训练联合优化安全密钥率代理、原始 KDR、互易性、表面功耗和约束违反；完整协调后 KDR 在最终协议评价阶段单独报告。
