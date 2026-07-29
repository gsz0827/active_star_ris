# 部分有源 STAR-RIS 辅助物理层密钥生成

本仓库提供一个面向论文研究的 **部分有源 STAR-RIS（Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surface）辅助物理层密钥生成** 仿真与鲁棒深度强化学习框架。

当前推荐使用独立实现：

```text
src/active_star_ris/full_scheme_v2/
```

该版本在双向 TDD 信道中联合考虑：

- 部分有源 STAR-RIS 的放大增益、透射/反射相移与能量分配；
- 有源单元内部放大噪声及其级联传播；
- 静态、方向相关和逐探测样本变化的幅相硬件失配；
- 透射/反射相位耦合、增益和相位量化；
- LS、LMMSE 或 NMSE 型不完美 CSI；
- 几何位置、UPA 阵列响应、距离路径损耗和 Rician 衰落；
- Eve 观测、信息泄漏、有限长度安全比特代理；
- Guard-band 量化、Cascade 式协调、密钥验证和 Toeplitz 隐私放大；
- STAR-RIS RF 输出功率、单元饱和、DC 功耗及约束违反概率；
- 均值、CVaR 和最差样本分离统计；
- 鲁棒 TD3 联合优化。

> 旧版模块和脚本仍保留，用于历史结果复现和消融对比。新实验应优先使用名称中带有 `full_scheme_v2` 的配置、脚本和模块。

---

## 1. 研究问题

系统包含控制端 Alice、位于透射侧的 Bob-T、位于反射侧的 Bob-R、STAR-RIS，以及分别靠近两侧用户的被动窃听者 Eve-T 和 Eve-R。

```text
                         ┌────────────────────┐
                         │      STAR-RIS      │
                         │ N 个单元，部分有源 │
                         └──────┬──────┬──────┘
                                │      │
                    反射区域    │      │    透射区域
                                │      │
               Bob-R / Eve-R ──┘      └── Bob-T / Eve-T
                                
                         Alice（控制端）
```

Alice 与 Bob-T、Bob-R 分别执行双向 TDD 探测。强化学习智能体根据合法链路控制 CSI 和上一时隙性能状态，联合优化 STAR-RIS：

\[
\{a_n,\theta_{T,n},\theta_{R,n},\beta_{T,n}\}_{n=1}^{N},
\]

其中：

\[
\beta_{T,n}+\beta_{R,n}=1,
\]

\[
\phi_{T,n}=a_n\sqrt{\beta_{T,n}}e^{j\theta_{T,n}},\qquad
\phi_{R,n}=a_n\sqrt{\beta_{R,n}}e^{j\theta_{R,n}}.
\]

无源单元满足 \(a_n=1\)，有源单元满足 \(1\le a_n\le a_{\max}\)。论文主配置使用正交相位耦合：

\[
\theta_{R,n}=\theta_{T,n}+\frac{\pi}{2}\pmod{2\pi}.
\]

代码同时保留 `independent` 和 `hybrid` 模式，用于硬件假设或消融实验。

---

## 2. V2 版本实现内容

### 2.1 四种 STAR-RIS 架构

| 架构名称 | 含义 |
|---|---|
| `passive` | 所有单元均为无源单元 |
| `partially_active_fixed` | 固定位置的部分单元为有源单元 |
| `partially_active_dynamic` | 部分有源候选单元由连续门控动作动态激活 |
| `fully_active_fixed` | 所有单元固定为有源单元，不进行静默旁路 |

`fully_active_fixed` 与动态旁路方案严格区分，避免“全有源”基线在功率投影后实际只剩部分有源单元。

### 2.2 几何信道和时间相关性

- 根据节点三维位置计算距离和自由空间参考损耗；
- 支持 UPA 阵列响应；
- RIS 级联链路采用 Rician 信道；
- 支持块内 Gauss-Markov 演化和相邻环境步之间的相关性；
- 根据正反向探测时延和相干时间计算正反向相关系数；
- Eve 信道可通过 `eve_spatial_correlation` 与合法用户信道相关。

### 2.3 硬件失配

硬件模型包含：

| 失配项 | 时间尺度 |
|---|---|
| 公共增益误差 | 一个 episode 内固定 |
| 正反向方向增益误差 | 一个 episode 内固定 |
| 透射/反射静态相位误差 | 一个 episode 内固定 |
| 正反向方向相位误差 | 一个 episode 内固定 |
| 快速相位抖动 | 每个探测样本重新生成 |
| 幅相耦合 | 随实际增益偏差变化 |
| 能量分配误差 | 每次实现时施加 |
| 相位与增益量化 | 按配置位数离散化 |

快速相位抖动的正反向相关性由：

```yaml
hardware:
  fast_jitter_forward_reverse_correlation: 0.0
```

控制。

### 2.4 不完美 CSI

控制端 CSI 支持：

- `ls`：基于导频的 LS 估计；
- `lmmse`：基于先验方差和导频观测的 LMMSE 估计；
- `nmse`：直接指定 NMSE 的误差模型。

论文主配置默认使用：

```yaml
channel:
  control_csi_model: lmmse
  csi_pilot_symbols: 32
  csi_pilot_power: 0.1
```

### 2.5 有源噪声和功率模型

有源单元内部噪声从输入端注入，经有源系数和后续信道传播至 Alice、Bob 和 Eve。

在同一次正向探测中，Bob 与 Eve 共用同一个 STAR-RIS 内部噪声源，
但该噪声分别经过 Bob 和 Eve 的下游信道传播；在同一次反向探测中，
Alice 与 Eve 同样共用对应的内部噪声源。正向和反向位于不同时隙，
因此两次内部噪声实现相互独立。

功率约束同时考虑：

- 三个探测方向的 RF 输出功率；
- 单个有源单元饱和功率；
- 放大器效率；
- 控制器静态功耗；
- 无源和有源单元控制功耗；
- 有源单元偏置功耗；
- 开关网络静态功耗；
- 总 DC 功率限制。

请求增益不可行时，代码通过一维投影缩放有源增益。固定架构不会通过投影改变有源单元集合。

### 2.6 密钥协议和安全指标

每个透射/反射分支执行：

1. 双向信道探测；
2. 特征提取与标准化；
3. Guard-band 量化；
4. 原始 KDR 统计；
5. Cascade 式信息协调；
6. 密钥一致性验证；
7. Eve 信息估计；
8. 有限长度和公开泄漏扣除；
9. Toeplitz 隐私放大；
10. 安全密钥吞吐率计算。

安全密钥训练目标是面向仿真和优化的可操作代理，不自动等价于严格的可组合安全证明。论文中仍应明确 Eve 能力、公开信道、安全参数以及硬件模型来源。

### 2.7 鲁棒目标

每个动作在多次独立不确定性实现下评估。单次奖励联合考虑：

\[
R = w_R\frac{R_{\mathrm{key}}}{R_{\mathrm{ref}}}
-w_{\mathrm{raw}}\frac{\mathrm{KDR}_{\mathrm{raw}}}{K_{\mathrm{raw,ref}}}
-w_{\mathrm{post}}\frac{\mathrm{KDR}_{\mathrm{post}}}{K_{\mathrm{post,ref}}}
+w_\rho\rho
-w_P\frac{P_{\mathrm{surface}}}{P_{\mathrm{ref}}}
-w_VV^2.
\]

最终鲁棒奖励为：

\[
R_{\mathrm{robust}}
=\lambda_{\mathrm{mean}}\,\mathbb{E}[R]
+\lambda_{\mathrm{CVaR}}\,\mathrm{CVaR}_{\alpha}(R).
\]

代码分别保存均值、CVaR 和最差样本指标，避免将最差样本值错误地当作平均实验结果。

---

## 3. 目录结构

```text
active_star_ris/
├── configs/
│   ├── default.yaml
│   ├── full_scheme_v2.yaml
│   └── full_scheme_v2_paper.yaml       # 推荐论文配置
│
├── src/active_star_ris/
│   ├── full_scheme_v2/
│   │   ├── config.py                   # 配置数据类与 YAML 加载
│   │   ├── channels.py                 # 几何、路径损耗、Rician、CSI、时间演化
│   │   ├── hardware.py                 # 动作解码、架构掩码、相位耦合和硬件失配
│   │   ├── power.py                    # RF/DC 功率、饱和限制和增益投影
│   │   ├── probing.py                  # 双向探测、内部噪声和 Eve 观测
│   │   ├── key_protocol.py             # 量化、协调、验证、Toeplitz 隐私放大
│   │   ├── objective.py                # 联合奖励与鲁棒统计
│   │   ├── environment.py              # 强化学习环境
│   │   ├── td3.py                      # TD3 智能体与经验回放
│   │   ├── experiments.py              # 训练、评价和多架构实验
│   │   └── models.py                   # 公共结果数据结构
│   └── ...                             # 旧版模块，保留用于兼容和对比
│
├── scripts/
│   ├── run_all_checks_v2.py            # V2 测试和烟雾检查
│   ├── train_full_scheme_v2.py         # 单架构训练
│   ├── run_full_scheme_experiments_v2.py # 多架构、多种子实验
│   └── plot_full_scheme_v2_paper.py    # 论文汇总图
│
├── tests/
│   └── test_full_scheme_v2.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 4. 环境安装

### 4.1 环境要求

- Python 3.11 或更高版本；
- PyTorch；
- NumPy；
- PyYAML；
- Matplotlib；
- pytest。

CUDA 不是必须条件，但正式多种子 TD3 训练建议使用支持 CUDA 的 PyTorch 环境。

### 4.2 Windows CMD / PowerShell

在仓库根目录运行：

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

PowerShell 若禁止执行激活脚本，可临时运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

也可以直接双击仓库根目录的：

```text
start_here.bat

```

### 4.3 Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## 5. 覆盖代码后的完整检查

```bash
python scripts/run_all_checks_v2.py
```

该脚本会：

- 运行 `tests/test_full_scheme_v2.py`；
- 创建 V2 环境并执行一次 reset/step；
- 对 V2 源代码执行 Python 编译检查；
- 通过 pytest 验证相位耦合、逐样本快抖动、架构语义、CSI、安全率、鲁棒统计、全有源约束和完整密钥协议。

也可以单独运行：

```bash
python -m pytest tests/test_full_scheme_v2.py -q
```

检查通过只代表代码、接口和基础数值流程正常，不代表正式参数已经完成硬件标定，也不代表训练一定收敛。

---

## 6. 运行实验

### 6.1 快速烟雾实验

先确认完整多架构流水线能够运行：

```bash
python scripts/run_full_scheme_experiments_v2.py --smoke --output-dir results/full_scheme_v2/suite_smoke
```

烟雾模式会自动缩小网络、训练步数、探测块、鲁棒样本数和评价回合，并只运行：

- `passive`；
- `partially_active_fixed`。

烟雾结果仅用于检查流程，不能用于论文结论。

### 6.2 建议的预实验

正式实验前先运行 2 个种子、5000 步：

```bash
python scripts/run_full_scheme_experiments_v2.py --config configs/full_scheme_v2_paper.yaml --steps 5000 --episodes 10 --seeds 0 1 --objective-samples 16 --final-probing-samples 256 --output-dir results/full_scheme_v2/pilot
```

预实验应检查：

- reward、critic loss、actor loss 中没有 `NaN` 或 `inf`；
- `worst_reward <= cvar_reward <= mean_reward`；
- 安全密钥率没有长期全部为 0；
- `power_violation_probability` 在可接受范围内；
- `fully_active_fixed` 的平均有源单元数等于总单元数；
- 不同随机种子之间存在合理差异，但没有数量级失控。

### 6.3 正式多架构、多种子实验

最终评价不会只使用 episode 初始的全零历史状态。每个评价回合会先运行
`episode_length - 1` 个代理协议 burn-in 步骤，使时变信道和上一时隙
性能摘要进入智能体状态，再在最终时隙执行完整密钥协议评价。

```bash
python scripts/run_full_scheme_experiments_v2.py --config configs/full_scheme_v2_paper.yaml --output-dir results/full_scheme_v2/paper
```

论文配置默认：

```yaml
experiment:
  training_steps: 100000
  evaluation_episodes: 100
  seeds: [0, 1, 2, 3, 4, 5, 6, 7]
  architectures:
    - passive
    - partially_active_fixed
    - partially_active_dynamic
    - fully_active_fixed
```

正式实验脚本的关键命令行参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--steps` | YAML 中的 `training_steps` | 每个架构、每个种子的训练步数 |
| `--episodes` | YAML 中的 `evaluation_episodes` | 最终评价回合数 |
| `--seeds` | YAML 中的 `seeds` | 训练种子列表 |
| `--objective-samples` | 32 | 最终每个动作的鲁棒不确定性样本数 |
| `--final-probing-samples` | 1024 | 最终协议评价探测块长度 |
| `--output-dir` | `results/full_scheme_v2/paper` | 输出目录 |
| `--smoke` | 关闭 | 极小规模流程测试 |

> 训练阶段使用 YAML 中的 `robust.objective_samples` 和 `probing.samples_per_step`。`--objective-samples` 与 `--final-probing-samples` 主要控制最终评价，从而减少有限长度和蒙特卡洛统计误差。

### 6.4 单独训练一个架构

```bash
python scripts/train_full_scheme_v2.py --config configs/full_scheme_v2_paper.yaml --architecture partially_active_fixed --seed 0 --steps 100000 --output-dir results/full_scheme_v2/single_partial_seed0
```

可用架构：

```text
passive
partially_active_fixed
partially_active_dynamic
fully_active_fixed
```

快速单架构训练：

```bash
python scripts/train_full_scheme_v2.py --config configs/full_scheme_v2_paper.yaml --architecture partially_active_fixed --smoke --steps 20 --output-dir results/full_scheme_v2/train_smoke
```

```markdown
### 6.5 训练和评估进度显示

多架构实验运行时，终端会显示：

- 当前总任务编号，例如 `总任务 3/32`；
- 当前架构和随机种子；
- 训练完成步数和百分比；
- reward、安全密钥率、原始 KDR 和表面功耗；
- Critic 和 Actor 损失；
- 当前训练速度与预计剩余时间；
- 最终协议评估回合进度。

训练进度间隔由：

```python
progress_interval = max(1, steps // 100)

---

## 7. 绘制论文汇总图

完整实验结束后，根输出目录会生成：

```text
results/full_scheme_v2/paper/all_seed_summaries.csv
```

运行：

```bash
python scripts/plot_full_scheme_v2_paper.py --input results/full_scheme_v2/paper/all_seed_summaries.csv --output-dir results/full_scheme_v2/paper/figures
```

当前脚本生成：

```text
secure_key_rate.png
raw_kdr.png
surface_power.png
violation_probability.png
```

对应：

- 平均安全密钥生成速率；
- 原始密钥不一致率；
- STAR-RIS 总表面 DC 功耗；
- 功率约束违反概率。

误差棒根据跨种子结果计算近似 95% 置信区间。

预实验结果也可以画图：

```bash
python scripts/plot_full_scheme_v2_paper.py --input results/full_scheme_v2/pilot/all_seed_summaries.csv --output-dir results/full_scheme_v2/pilot/figures
```

---

## 8. 输出目录和文件说明

正式运行后的实际结构为：

```text
results/full_scheme_v2/paper/
├── passive/
│   ├── seed_0/
│   │   ├── td3_checkpoint.pt
│   │   ├── training_history.csv
│   │   └── evaluation.csv
│   ├── seed_1/
│   └── ...
├── partially_active_fixed/
│   └── seed_*/
├── partially_active_dynamic/
│   └── seed_*/
├── fully_active_fixed/
│   └── seed_*/
├── all_seed_summaries.csv
├── config_snapshot.json
└── figures/                         # 运行绘图脚本后生成
    ├── secure_key_rate.png
    ├── raw_kdr.png
    ├── surface_power.png
    └── violation_probability.png
```

### 8.1 `training_history.csv`

训练过程中每 100 个环境步记录一条历史数据到内存，并在当前
“架构 × 种子”的全部训练完成后统一写入 `training_history.csv`。
训练结束时同时保存 `td3_checkpoint.pt`。

当前版本尚未实现完整的周期检查点和断点续训。如果在一个模型训练
完成前中断程序，该模型尚未写出的训练历史和最终检查点可能丢失；
已经完成并写入目录的其他架构或种子结果不会受影响。

| 字段 | 含义 |
|---|---|
| `step` | 环境步 |
| `reward` | 当前鲁棒奖励 |
| `mean_secure_key_rate_bps` | 当前动作多不确定性样本的平均安全密钥率 |
| `mean_raw_kdr` | 平均原始 KDR |
| `mean_surface_power_watt` | 平均总 DC 功耗 |
| `power_violation_probability` | 功率约束违反概率 |
| `critic_loss` | Critic 损失 |
| `actor_loss` | Actor 损失；非更新步可能为空值或 NaN |

### 8.2 `evaluation.csv`

每个最终评价回合保存：

- `robust_reward`、`mean_reward`、`cvar_reward`、`worst_reward`；
- `mean/cvar/worst_secure_key_rate_bps`；
- `mean/cvar/worst_raw_kdr`；
- `mean_reciprocity`；
- `mean_surface_power_watt`；
- `power_violation_probability`；
- `mean_active_elements`；
- `mean_projection_scale`。

### 8.3 `all_seed_summaries.csv`

每个“架构 × 种子”占一行，汇总该种子的多个评价回合，并提供：

- 均值；
- 标准差；
- 95% 置信区间半宽。

绘图脚本以该文件作为输入。

### 8.4 `config_snapshot.json`

保存本次实验加载后的完整配置，便于复现实验。命令行临时覆盖的训练步数、评价回合、最终探测块等参数应同时记录在实验日志或论文实验表中。

---

## 9. 关键指标解释

| 指标 | 趋势 | 含义 |
|---|---|---|
| `mean_secure_key_rate_bps` | 越高越好 | 合法互信息、Eve 泄漏、协调开销、有限长度和协议耗时共同决定的安全密钥率代理 |
| `mean_raw_kdr` | 越低越好 | 信息协调前 Alice 与 Bob 量化比特的不一致比例 |
| `mean_reciprocity` | 越高越好 | 双向观测的复相关幅值 |
| `mean_surface_power_watt` | 越低越好 | STAR-RIS 平均总 DC 功耗 |
| `power_violation_probability` | 越低越好 | 鲁棒样本中出现总 RF 输出、总 DC 功耗或单个有源单元饱和功率违反的比例 |
| `mean_active_elements` | 视架构而定 | 实际有源单元数量 |
| `mean_projection_scale` | 越接近 1 越好 | 请求增益经功率投影后的保留比例 |
| `cvar_*` | 关注尾部 | 低奖励尾部样本对应的鲁棒性能 |
| `worst_*` | 极端诊断 | 所有不确定性样本中的最差实现 |

鲁棒奖励的正常统计关系应满足：

```text
worst_reward <= cvar_reward <= mean_reward
```

对于安全密钥率等“越大越好”的指标，在尾部样本通常也应观察到：

```text
worst_secure_key_rate_bps
<= cvar_secure_key_rate_bps
<= mean_secure_key_rate_bps
```

但 KDR 是“越小越好”指标，其 CVaR 值是按奖励尾部选出的样本计算，不应机械要求与均值满足固定大小关系。

---

## 10. 配置说明

推荐配置：

```text
configs/full_scheme_v2_paper.yaml
```

### 10.1 几何参数 `geometry`

包括：

- 载波频率；
- RIS 行列数和阵元间距；
- Alice、STAR-RIS、Bob-T、Bob-R、Eve-T、Eve-R 的三维位置；
- RIS 和直达链路路径损耗指数；
- 附加链路损耗。

### 10.2 信道与 CSI `channel`

```yaml
channel:
  rician_k_factor: 3.0
  within_block_correlation: 0.995
  between_step_correlation: 0.98
  forward_reverse_delay_seconds: 0.001
  channel_coherence_time_seconds: 0.01
  eve_enabled: true
  eve_spatial_correlation: 0.20
  control_csi_model: lmmse
```

### 10.3 探测参数 `probing`

包括探测块长度、三方导频符号数与功率、内部放大噪声、接收机噪声、符号时长和保护间隔。

### 10.4 硬件参数 `hardware`

论文主实验应使用：

```yaml
hardware:
  phase_coupling_mode: quadrature
```

可选模式：

| 模式 | 含义 |
|---|---|
| `quadrature` | 所有单元满足透射/反射正交相位耦合 |
| `hybrid` | 无源单元耦合，有源单元允许独立相位 |
| `independent` | 透射和反射相位完全独立，仅用于特殊硬件假设或消融 |

### 10.5 功率参数 `power`

应重点标定：

- `maximum_rf_output_power`；
- `maximum_total_dc_power`；
- `amplifier_efficiency`；
- 有源单元控制和偏置功耗；
- `csi_power_margin_std`；
- `hardware_gain_margin_db`。

### 10.6 密钥协议 `key_generation`

包括量化 Guard-band、选择策略、协调开销、协调轮数、验证标签长度、隐私余量、安全参数、最终密钥长度限制和公开信道速率。

### 10.7 鲁棒评价 `robust`

```yaml
robust:
  objective_samples: 16
  cvar_alpha: 0.25
  mean_weight: 0.5
  cvar_weight: 0.5
```

当 `objective_samples=16`、`cvar_alpha=0.25` 时，CVaR 使用最低奖励的 4 个样本，而不是仅使用一个最差样本。

---

## 11. 论文实验建议

代码流水线可以运行并不意味着默认参数已经足以支撑投稿。正式论文至少应补充：

- 多训练种子和独立测试种子；
- 95% 置信区间或配对 bootstrap；
- 随机相位、启发式相位、非鲁棒 TD3 等算法基线；
- 无源、固定部分有源、动态部分有源和固定全有源架构比较；
- 有源比例、功率预算、最大增益、内部噪声和 Eve 位置扫描；
- CSI 误差、硬件误差和快相位抖动敏感性分析；
- KGR–功耗 Pareto 曲线；
- 样本外功率违反概率；
- 训练收敛曲线和在线推理时间；
- 最终协调成功率、验证成功率和有效密钥长度。

论文主结论不应只依赖单个 reward。部分有源方案的优势可以表现为更好的安全密钥率、功耗、CVaR 或 Pareto 折中。

---

## 12. 重要科研边界

1. `full_scheme_v2_paper.yaml` 中的功耗、噪声、带宽、路径损耗和硬件失配参数是可运行的研究配置，不代表特定商用硬件平台的实测数据。
2. 投稿前应根据论文采用的器件、原型平台或可靠参考文献重新标定参数。
3. 当前安全密钥率包含 Eve 信息、协调泄漏和有限长度惩罚，但仍属于仿真安全代理。
4. 严格的可组合安全证明需要额外明确攻击模型、参数估计方法、认证公开信道和安全参数组合方式。
5. 当前 `evaluation.csv` 主要导出联合安全密钥率、原始 KDR、互易性和功耗统计；协议内部计算的分支级协调后 KDR、公开泄漏和最终密钥一致性尚未全部展开为 CSV 列。如论文需要逐项报告，应扩展 `evaluate_policy()` 的导出字段。

---

## 13. 旧版代码

仓库根目录原有的下列模块和脚本仍可用于历史实验：

```text
src/active_star_ris/*.py
scripts/run_full_experiments.py
scripts/run_main_experiment.py
scripts/run_sweep.py
configs/default.yaml
```

这些入口与 `full_scheme_v2` 的物理假设、指标定义和输出结构不同。不要在同一张论文表中混用两个版本的结果，除非明确说明模型差异。

---

## 14. 参考文献

- S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” ICML, 2018.
- X. Mu et al., “Simultaneously Transmitting and Reflecting (STAR) RIS Aided Wireless Communications,” IEEE Transactions on Wireless Communications, 2022.
- R. Long et al., “Active Reconfigurable Intelligent Surface-Aided Wireless Communications,” IEEE Transactions on Wireless Communications, 2022.

使用本仓库形成论文时，还应补充与 STAR-RIS 物理层密钥生成、有源 RIS 密钥生成、有限长度物理层安全和信息协调相关的最新文献。

---

## 15. 最小运行流程

```bash
# 1. 安装
python -m pip install -r requirements.txt
python -m pip install -e .

# 2. 检查
python scripts/run_all_checks_v2.py

# 3. 预实验
python scripts/run_full_scheme_experiments_v2.py --config configs/full_scheme_v2_paper.yaml --steps 5000 --episodes 10 --seeds 0 1 --objective-samples 16 --final-probing-samples 256 --output-dir results/full_scheme_v2/pilot

# 4. 正式实验
python scripts/run_full_scheme_experiments_v2.py --config configs/full_scheme_v2_paper.yaml --output-dir results/full_scheme_v2/paper

# 5. 绘图
python scripts/plot_full_scheme_v2_paper.py --input results/full_scheme_v2/paper/all_seed_summaries.csv --output-dir results/full_scheme_v2/paper/figures
```
