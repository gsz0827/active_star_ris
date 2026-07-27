# Active STAR-RIS 完整方案 v2：安装与替换说明

## 一、怎么放到你的仓库

把本代码包解压后，将其中的 `src`、`scripts`、`configs`、`tests` 四个文件夹复制到你的仓库根目录：

```text
active_star_ris/                 ← 你的仓库根目录
├─ configs/
├─ scripts/
├─ src/
│  └─ active_star_ris/
├─ tests/
├─ pyproject.toml
└─ requirements.txt
```

选择“合并文件夹”。本代码包全部使用新文件名，不要求覆盖你的旧版核心文件。

建议先建立分支：

```bash
git checkout feature/full-scheme
git checkout -b fix/full-scheme-v2
```

## 二、原文件是否需要替换

不需要替换以下旧文件：

```text
src/active_star_ris/joint_objective.py
src/active_star_ris/probing.py
src/active_star_ris/star_key_system.py
src/active_star_ris/hardware_impairments.py
src/active_star_ris/surface_power.py
src/active_star_ris/action_mapping.py
src/active_star_ris/rl_environment.py
src/active_star_ris/td3_agent.py
src/active_star_ris/td3_training.py
```

原因是新版完整实现放在独立子包：

```text
src/active_star_ris/full_scheme_v2/
```

新训练脚本只调用 `full_scheme_v2`，旧基线仍然可以继续运行。确认新版稳定后，再决定是否删除或重定向旧入口。

## 三、需要新增的文件及准确位置

### 1. 核心包

全部放到：

```text
src/active_star_ris/full_scheme_v2/
```

| 文件 | 对应阶段 | 作用 |
|---|---:|---|
| `__init__.py` | 公共入口 | 导出配置、环境和 TD3 API |
| `config.py` | 全阶段 | 所有模型、训练和鲁棒配置；读取 YAML |
| `models.py` | 全阶段 | 信道、表面、功率、密钥和目标结果数据结构 |
| `channels.py` | 阶段 2 | Rician/Gauss-Markov 信道、正反向探测时延和不完美控制 CSI |
| `probing.py` | 阶段 2 | 双向导频观测、有源内部噪声转发和终端 RF 链失配 |
| `key_protocol.py` | 阶段 1 | 保护间隔量化、Cascade 协调、验证、Toeplitz 隐私放大和协议 KGR |
| `hardware.py` | 阶段 3 | 联合动作映射、幅相失配、能量分配误差、量化、耦合相移和端点硬件 |
| `power.py` | 阶段 3 | RF 输出、总 DC 功耗、逐单元饱和和二分功率投影 |
| `objective.py` | 阶段 1/3 | KGR、KDR、互易性、功耗和约束联合奖励 |
| `environment.py` | 阶段 4 | 域随机化、多样本下尾 CVaR、状态、动作和时变环境 |
| `td3.py` | 阶段 4 | 双 Critic TD3、回放池、目标平滑、延迟更新及正确截断 bootstrap |
| `experiments.py` | 阶段 5 | 基线评价、参数扫描和 CSV 输出 |

### 2. 配置

放到：

```text
configs/full_scheme_v2.yaml
```

主要参数都可从该文件调整，包括：

- 有源单元比例和最大增益；
- 正反向探测时间间隔和相干时间；
- 控制 CSI 的 NMSE 范围；
- 有源放大器输入参考噪声；
- 幅相误差、能量分配误差和有限比特控制；
- RF、DC 和逐单元饱和预算；
- KGR、KDR、互易性和功耗权重；
- 域随机化与 CVaR 参数。

### 3. 训练和实验脚本

放到：

```text
scripts/train_full_scheme_v2.py
scripts/run_full_scheme_experiments_v2.py
scripts/run_all_checks_v2.py
```

### 4. 测试

放到：

```text
tests/test_full_scheme_v2.py
```

## 四、五个阶段在代码中的连接方式

```text
TD3 动作
  ↓
hardware.decode_action
  ↓
power.project_command_to_power_constraints
  ↓
hardware.apply_hardware
  ↓
channels.build_bidirectional_block
  ↓
probing.simulate_dual_side_probing
  ↓
key_protocol.evaluate_key_rate
  ↓
objective.evaluate_objective
  ↓
environment 的均值 + 下尾 CVaR 奖励
```

### 阶段 1：协议级密钥指标

训练模式使用：

```text
有限长度条件熵代理
- 估计协调泄漏
- 验证泄漏
- 隐私裕量
```

评价模式使用：

```text
保护间隔量化
→ Cascade 协调
→ 验证标签
→ Toeplitz 隐私放大
→ 最终密钥位数 / 完整帧时间
```

同时报告：

- 理论高斯互信息，仅作为诊断；
- 训练密钥率；
- 最终协议密钥率；
- 协调前 KDR；
- 协调后 KDR；
- 保留率和密钥成功率。

### 阶段 2：双向信道

正向和反向不再强制使用完全相同的传播信道。相关性由：

```text
exp(-forward_reverse_delay_seconds / channel_coherence_time_seconds)
```

决定。

控制器的不完美 CSI 和密钥探测的接收噪声是两套独立模型：

- 控制 CSI：用于状态和功率投影；
- 导频观测误差：由导频长度、接收噪声和放大器噪声产生。

### 阶段 3：硬件和功率

联合动作维度为：

```text
有源增益数 + N 个透射相位 + N 个反射相位 + N 个能量分配系数
```

功率约束包含：

- 三个探测方向中的最大 RF 输出；
- 按时隙比例计算的平均附加 RF 功率；
- 放大器 DC 功耗；
- 控制、偏置和开关网络功耗；
- 逐有源单元输出饱和。

投影顺序为：

```text
CSI 保守上界
→ 硬件增益裕量
→ 单元饱和裁剪
→ RF/DC 联合二分投影
→ 增益向下量化
→ 最终可行性复核
```

### 阶段 4：鲁棒 TD3

环境在每个 episode 随机化：

- NMSE；
- 放大器噪声；
- 接收机噪声；
- RF 预算；
- DC 预算；
- 静态硬件 realization。

每个动作执行多次随机评价，奖励为：

```text
mean_weight × 平均奖励 + cvar_weight × 最差尾部平均奖励
```

时间上限 `truncated=True` 不会被写成 MDP 吸收终止，因此 TD3 仍会对下一状态 bootstrap。

### 阶段 5：实验

实验脚本输出：

```text
baseline_comparison.csv
active_ratio_sweep.csv
nmse_sweep.csv
delay_sweep.csv
rf_power_budget_sweep.csv
dc_power_budget_sweep.csv
ablation_comparison.csv
```

有源比例会改变 TD3 动作维度，因此代码不会错误地把一个比例训练的模型直接用于另一个比例。若论文要比较不同有源比例的 TD3，每个比例必须单独训练。

## 五、安装和运行

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 1. 一键检查

```bash
python scripts/run_all_checks_v2.py
```

该命令依次执行：

- Python 编译检查；
- 新版自动测试；
- 10 步训练冒烟测试；
- 快速基线和扫描实验。

### 2. 单独运行测试

```bash
python -m pytest -q tests/test_full_scheme_v2.py
```

### 3. 快速训练检查

```bash
python scripts/train_full_scheme_v2.py \
  --smoke \
  --steps 20 \
  --output-dir results/full_scheme_v2/smoke
```

Windows PowerShell 可写成一行：

```powershell
python scripts/train_full_scheme_v2.py --smoke --steps 20 --output-dir results/full_scheme_v2/smoke
```

### 4. 正式训练

```bash
python scripts/train_full_scheme_v2.py \
  --config configs/full_scheme_v2.yaml \
  --steps 100000 \
  --seed 0 \
  --output-dir results/full_scheme_v2/seed_0
```

建议至少运行五个随机种子：

```powershell
0..4 | ForEach-Object {
  python scripts/train_full_scheme_v2.py `
    --config configs/full_scheme_v2.yaml `
    --steps 100000 `
    --seed $_ `
    --output-dir "results/full_scheme_v2/seed_$_"
}
```

### 5. 快速实验

```bash
python scripts/run_full_scheme_experiments_v2.py \
  --quick \
  --output-dir results/full_scheme_v2/quick_experiments
```

### 6. 使用已训练检查点运行正式实验

```bash
python scripts/run_full_scheme_experiments_v2.py \
  --config configs/full_scheme_v2.yaml \
  --checkpoint results/full_scheme_v2/seed_0/td3_final.pt \
  --episodes 10 \
  --output-dir results/full_scheme_v2/final_experiments
```

## 六、重要科研说明

`minimum_entropy_bits_per_retained_bit` 是未显式接入 Eve 观测时的保守工程代理，不等于可组合安全证明。

如果论文需要声称“安全密钥率”，下一步必须把 Eve 条件最小熵或有限长度安全下界传入 `key_protocol.evaluate_key_rate()`，替代固定熵比例。当前实现适合称为：

```text
协议级有效密钥生成速率
```

而不是在没有 Eve 安全证明时直接称为：

```text
可组合安全密钥生成速率
```

## 七、已完成的本地验证

本代码包已执行：

```text
python -m compileall
pytest
训练 smoke test
实验 --quick
```

自动测试包括：

- 配置读取；
- 探测时延降低平均互易性；
- 高相关观测能生成最终密钥；
- RF/DC/单元饱和联合投影；
- 时间截断不被存为真正终止；
- 环境单步完整链路。
