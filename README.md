# Active STAR-RIS Complete Baseline

这是一个可直接运行、可继续修改的完整基线项目，统一包含：

1. Rayleigh / Rician / Gauss-Markov 时变信道；
2. 无源 STAR-RIS 能量分割模型；
3. 部分有源单元选择与放大增益；
4. 有源单元内部噪声及其向用户端的转发；
5. STAR-RIS 总输出功率约束；
6. 透射侧和反射侧用户的 SNR、速率计算；
7. 随机相位、无源优化、部分有源优化等基线；
8. 能量分割系数一维搜索；
9. Monte Carlo 主实验和参数扫描；
10. 自动测试。

## 1. 模型约定

第 n 个单元满足：

- beta_T,n + beta_R,n = 1
- phi_T,n = a_n sqrt(beta_T,n) exp(j theta_T,n)
- phi_R,n = a_n sqrt(beta_R,n) exp(j theta_R,n)

其中：

- 被动单元：a_n = 1
- 有源单元：1 <= a_n <= a_max

用户 q 的等效信道：

h_eff,q = h_d,q + sum_n h_q,n phi_q,n g_n

有源器件噪声转发至用户 q 的方差：

sigma_forward,q^2
= sigma_RIS^2 sum_(n in A) |h_q,n phi_q,n|^2

用户 SNR：

gamma_q
= P_tx |h_eff,q|^2
  / (sigma_user,q^2 + sigma_forward,q^2)

本基线中的 STAR-RIS 有源输出功率约束为：

sum_(n in A) a_n^2
(P_tx |g_n|^2 + sigma_RIS^2)
<= P_RIS,max

这是一个清晰、可检验的工程基线。后续可根据论文中的具体硬件功耗定义替换。

## 2. Windows 安装与运行

在 VS Code 的项目根目录打开 CMD 或 PowerShell。

创建虚拟环境：

```powershell
py -3.11 -m venv .venv
```

PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD 激活：

```cmd
.venv\Scripts\activate
```

安装依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行全部检查：

```powershell
python scripts\run_all_checks.py
```

运行主实验：

```powershell
python scripts\run_main_experiment.py
```

运行参数扫描：

```powershell
python scripts\run_sweep.py
```

运行自动测试：

```powershell
python -m pytest -q
```

## 3. 输出文件

主实验会在 outputs 中生成：

- main_metrics.json
- main_average_rates.png
- main_average_snrs.png
- main_rate_cdf.png

参数扫描会生成：

- sweep_results.csv
- sweep_sum_rate.png

## 4. 建议修改顺序

后续建议按以下顺序修改：

1. 明确论文的系统拓扑与变量符号；
2. 替换路径损耗和几何位置模型；
3. 明确有源单元功率模型；
4. 明确目标函数；
5. 增加优化算法；
6. 增加论文所需对比算法与消融实验。
