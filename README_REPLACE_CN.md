# 直接覆盖版代码说明

本压缩包没有安装器，也没有 `.bat`。目录结构与仓库根目录一致。

## 使用方式

1. 先备份你的整个仓库。
2. 将本压缩包解压。
3. 把解压后的 `src`、`configs`、`scripts`、`tests` 四个文件夹复制到你的 `active_star_ris` 仓库根目录。
4. Windows 提示是否合并文件夹或替换同名文件时，选择“是”。

本包主要新增/替换：

- `src/active_star_ris/full_scheme_v2/`
- `configs/full_scheme_v2_paper.yaml`
- `scripts/train_full_scheme_v2.py`
- `scripts/run_full_scheme_experiments_v2.py`
- `scripts/run_all_checks_v2.py`
- `scripts/plot_full_scheme_v2_paper.py`
- `tests/test_full_scheme_v2.py`

不会删除你原来的旧版模块与结果。

## 已落实的修复

- 无源单元默认采用透射/反射正交相位耦合；同时保留 independent/hybrid 消融模式。
- 快速相位抖动按每个探测样本生成，并可配置正反向相关性。
- 均值、CVaR、最差样本指标分别统计，状态和主评价默认使用均值。
- 无源、固定部分有源、动态部分有源、固定全有源四种架构语义分开；全有源不会被启发式静默旁路。
- 加入几何位置、UPA阵列响应、距离路径损耗及Rician信道。
- 加入LS/LMMSE导频CSI估计和训练开销参数。
- 加入Eve观测、信息泄漏、有限长度安全比特代理、Cascade式协调、验证和Toeplitz隐私放大。
- 加入RF输出功率、单元饱和、DC功耗和样本外违反概率。
- 提供标准TD3、多架构多种子实验、95%置信区间汇总和绘图入口。

## 检查

在仓库根目录的 CMD 或 PowerShell 运行：

```bat
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts\run_all_checks_v2.py
```

## 快速训练

```bat
python scripts\train_full_scheme_v2.py --config configs\full_scheme_v2_paper.yaml --smoke --steps 20 --output-dir results\full_scheme_v2\smoke
```

## 正式实验

```bat
python scripts\run_full_scheme_experiments_v2.py --config configs\full_scheme_v2_paper.yaml --output-dir results\full_scheme_v2\paper
```

正式实验计算量很大。先运行 `--smoke`，确认环境和依赖无误后再运行完整配置。正式评价默认将探测块扩展到 1024 个样本，以降低有限长度惩罚；可用 `--final-probing-samples` 修改。

## 科研边界

代码中安全密钥训练目标是基于观测、量化、协调泄漏、Eve信息及有限长度惩罚构造的可操作代理。最终评价额外执行协调、验证和Toeplitz隐私放大，但这仍不自动等价于严格的可组合安全证明。论文中需要明确安全假设、Eve模型、公开信道模型和硬件参数来源。
