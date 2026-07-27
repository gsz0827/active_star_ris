# Active STAR-RIS Assisted Physical-Layer Key Generation

A complete simulation and robust deep reinforcement learning framework for **partially-active STAR-RIS assisted physical-layer key generation** over bidirectional TDD channels, with full consideration of:

- **Internal amplifier noise** forwarded from active elements to legitimate users
- **Amplitude/phase hardware mismatches** (static, directional, fast jitter, and amplitude-phase coupling)
- **Imperfect CSI** (NMSE-based channel estimation errors)
- **Per-element power constraints** with robust safety margins
- **Joint performance metrics**: key generation rate (KGR), key disagreement rate (KDR), observation reciprocity, and surface power consumption
- **Robust TD3 deep reinforcement learning** for jointly optimizing per-element amplifier gains, transmission/reflection phase shifts, and energy-splitting ratios

---

## 1. System Model

### 1.1 Topology

```
            ┌──────────────────┐
            │   STAR-RIS       │
            │  (N elements,    │
            │  N_a active)     │
            └──┬────────────┬──┘
       g_n    │            │    h_n
              │            │
    ┌─────────┴──┐      ┌──┴──────────┐
    │ Controller │      │ User T / R   │
    │   (Alice)  │      │   (Bob)      │
    └────────────┘      └──────────────┘
```

- **Controller (Alice)** ↔ **STAR-RIS** ↔ **Transmission User (Bob-T)** (transmission side)
- **Controller (Alice)** ↔ **STAR-RIS** ↔ **Reflection User (Bob-R)** (reflection side)
- Both links perform TDD bidirectional channel probing for secret key generation

### 1.2 STAR-RIS Element Model

For the n-th element:

```
β_T,n + β_R,n = 1                                    (energy conservation)

phi_T,n = a_n · sqrt(β_T,n) · exp(j θ_T,n)           (transmission coefficient)
phi_R,n = a_n · sqrt(β_R,n) · exp(j θ_R,n)           (reflection coefficient)
```

where:
- **Passive element**: `a_n = 1`
- **Active element**: `1 ≤ a_n ≤ a_max` (configurable, default `a_max = 3.0`)

### 1.3 Effective Channel

For user q on side q ∈ {T, R}:

```
h_eff,q = h_direct,q + Σ_n h_q,n · phi_q,n · g_n
```

### 1.4 Forwarded Active Noise

Active elements inject internal amplification noise that propagates to the user:

```
σ²_forward,q = σ²_RIS · Σ_{n∈A} |h_q,n · phi_q,n|²
```

where A is the set of active element indices.

### 1.5 Bidirectional Probing

- **Forward slot**: Controller → STAR-RIS → User (pilot power P_A)
- **Reverse slot**: User → STAR-RIS → Controller (pilot power P_B)
- Forward and reverse active noise realizations are **independent**
- Hardware mismatch causes different forward/reverse coefficients

### 1.6 Hardware Impairments

Each active element's actual coefficient differs from the ideal one due to:

| Error Type | Description | Duration |
|---|---|---|
| Static gain error | Common to forward & reverse, per-element | Episode-fixed |
| Directional gain error | Independent forward vs reverse | Episode-fixed |
| Static phase error | Transmission & reflection branches | Episode-fixed |
| Directional phase error | Forward vs reverse non-reciprocity | Episode-fixed |
| Fast phase jitter | Resampled every probing slot | Per-step |
| Amplitude-phase coupling | Phase shift ∝ gain deviation (rad/dB) | Episode-fixed |

Passive elements are limited to insertion loss only (no net gain from errors).

### 1.7 Robust Power Constraint

The STAR-RIS total output power across all three probing directions (controller →, transmission user →, reflection user →) must satisfy:

```
max{ P_out,ctrl, P_out,T, P_out,R } ≤ P_RIS,max
```

A **robust margin** is applied using estimated CSI uncertainty bounds:

```
|h_true,n| ≤ |h_hat,n| + γ · σ_e
```

If the requested gains violate the constraint at unit gain, elements are **bypassed** (switched to passive mode) in order of ascending utility-to-burden ratio, or the gain vector is **uniformly scaled down** via a convex projection.

### 1.8 Surface Power Consumption

Total surface power includes:

```
P_total = P_controller_static
        + P_switching_network
        + N_passive × P_passive_ctrl
        + N_active × (P_active_ctrl + P_bias)
        + P_RF_additional / η_amplifier
```

where `P_RF_additional` is the extra RF output power beyond unit gain.

---

## 2. Key Generation Metrics

### 2.1 Per-Branch Metrics

For each (transmission / reflection) branch:

| Metric | Definition |
|---|---|
| **Mutual Information** | `I = -log₂(1 - |ρ|²)` bits/sample (Gaussian approximation) |
| **Raw KDR** | Fraction of mismatched bits after sign quantization |
| **Observation Reciprocity** | `|ρ|` = magnitude of complex correlation between Alice's and Bob's observations |

### 2.2 Joint Objective (RL Reward)

```
R = w₁ · (KGR / KGR_ref)
  - w₂ · (KDR / KDR_ref)
  + w₃ · reciprocity
  - w₄ · (P_total / P_ref)
  - w₅ · (violation / P_budget)²
```

Weights and references are configurable in `configs/default.yaml` → `joint_objective`.

### 2.3 Robust Reward (CVaR)

Each action is evaluated under **multiple independent noise realizations** (fast phase jitter, internal noise, receiver noise). The final reward is a weighted combination of:

```
R_robust = α_mean · R_mean + α_cvar · CVaR_α(R)
```

where `CVaR_α` is the conditional value-at-risk at the lower α-tail (default α = 0.25).

---

## 3. RL Approach: Robust TD3

### 3.1 Agent

- **Algorithm**: Twin Delayed Deep Deterministic Policy Gradient (TD3)
- **Actor network**: 2 hidden layers [256, 256] with ReLU, tanh output in [-1, 1]
- **Twin Critic**: Two independent Q-networks for reduced overestimation bias

### 3.2 Action Space

For N elements and N_a active element candidates:

```
dim(action) = N_a + 3N

action = [active_gains (N_a),
          transmission_phases (N),
          reflection_phases (N),
          transmission_energy_splits (N)]
```

All raw actions ∈ [-1, 1]; mapped to physical ranges via the action mapping module.

### 3.3 Observation Space

- Normalized estimated CSI (3N complex → 6N real)
- Normalized direct channel estimates (4 real)
- Impairment context (11 dim): NMSE, noise variances, power budget, HW error stds
- Previous step metrics (5 dim): reward, KGR, KDR, reciprocity, power

### 3.4 Domain Randomization

Each episode samples uniformly from ranges for:
- NMSE, RIS internal noise, receiver noise, power budget
- All hardware impairment standard deviations

This ensures the learned policy is **robust to uncertainty**.

---

## 4. Project Structure

```
active_star_ris/
├── configs/
│   └── default.yaml                       # Master configuration file
│
├── src/active_star_ris/
│   ├── __init__.py                        # Public API exports
│   ├── channels.py                        # Rayleigh, Rician, Gauss-Markov channels
│   ├── surface.py                         # STAR-RIS coefficient construction
│   ├── system.py                          # Two-user link evaluation (SNR, rate, power)
│   ├── optimization.py                    # Surface design: passive/active, element selection,
│   │                                      #   common/vector gain, scalar/element-wise beta search
│   ├── probing.py                         # TDD bidirectional channel probing
│   ├── star_key_system.py                 # Dual-side (T+R) key generation orchestration
│   ├── key_generation.py                  # Key metrics: complex correlation, Gaussian MI,
│   │                                      #   median-based sign quantization, raw KDR
│   ├── secure_key_generation.py           # Eve observation model & branch secrecy metrics
│   ├── finite_length_security.py          # Finite block-length security bounds
│   ├── operational_security.py            # Pre-reconciliation conditional entropy bounds
│   ├── quantized_security.py              # Quantized adversary classification (LDA, GBDT)
│   ├── practical_key_generation.py        # Guard-band quantization, Cascade reconciliation,
│   │                                      #   BLAKE2s privacy amplification
│   ├── hardware_impairments.py            # Static/directional gain & phase errors,
│   │                                      #   fast phase jitter, amplitude-phase coupling
│   ├── csi_estimation.py                  # Imperfect CSI (NMSE-based additive error model)
│   ├── surface_power.py                   # Bidirectional surface power model &
│   │                                      #   robust amplitude projection (common + vector)
│   ├── joint_objective.py                 # Unified reward: KGR + KDR + reciprocity + power
│   ├── action_mapping.py                  # RL action → physical STAR-RIS coefficients
│   │                                      #   with robust power projection
│   ├── rl_environment.py                  # Gymnasium-style robust RL environment
│   │                                      #   with domain randomization & CVaR
│   ├── td3_agent.py                       # PyTorch TD3: Actor, TwinCritic, ReplayBuffer
│   ├── td3_training.py                    # TD3 training loop & deterministic evaluation
│   ├── experiment_pipeline.py             # Full experiment suite: 3 baselines + 7 ablations
│   ├── simulation.py                      # Monte Carlo experiment framework
│   │                                      #   (imperfect CSI scenario evaluation)
│   ├── imperfect_csi_sweep.py             # CSI NMSE sweep aggregation & comparison
│   ├── parameter_sweep.py                 # Multi-dimensional parameter sweep
│   │                                      #   (NMSE, hardware impairments, block length)
│   ├── sweep_statistics.py                # Confidence intervals & aggregate sweep statistics
│   └── paper_plots.py                     # Publication-quality figure generation (PNG + PDF)
│
├── scripts/
│   ├── run_full_experiments.py            # ★ Main entry: train & evaluate all scenarios
│   ├── run_main_experiment.py             # Legacy: basic Monte Carlo rate simulation
│   ├── run_sweep.py                       # Legacy: parameter sweep (N, active ratios)
│   ├── run_all_checks.py                  # Run pytest + quick main experiment
│   │
│   │   # Incremental check scripts (steps 1–11):
│   ├── check_step1_key_metrics.py         #   Key generation metric validation
│   ├── check_step2_bidirectional_probing.py  # Bidirectional probing simulation
│   ├── check_step3_dual_side_key_generation.py  # Dual-side key generation
│   ├── check_step4_hardware_mismatch.py   #   Hardware impairment application
│   ├── check_step4_imperfect_csi_impact.py   # Imperfect CSI impact evaluation
│   ├── check_step5_secure_key_generation.py  # Eve observation & secrecy
│   ├── check_step6_finite_length_security.py # Finite block-length bounds
│   ├── check_step7_practical_key_generation.py  # Guard-band + Cascade + BLAKE2s
│   ├── check_step8_parameter_sweep.py     #   Multi-dimensional parameter sweep
│   ├── check_step9_statistical_sweep.py   #   Sweep statistics & confidence intervals
│   ├── check_step10_quantized_security.py #   Quantized adversary evaluation
│   ├── check_step11_paper_outputs.py      #   Paper figure & CSV generation
│   │
│   │   # RL pipeline check scripts:
│   ├── check_rl_action_mapping.py         #   Action decoding & power projection
│   ├── check_rl_environment.py            #   Environment reset/step/observation
│   └── check_td3_agent.py                 #   TD3 agent smoke test
│
├── tests/
│   ├── conftest.py                        # Shared pytest fixtures
│   ├── test_channels.py                   #   Channel generation
│   ├── test_surface.py                    #   Surface coefficient construction
│   ├── test_system.py                     #   Two-user link evaluation
│   ├── test_optimization.py               #   Surface design & beta search
│   ├── test_probing.py                    #   Bidirectional probing
│   ├── test_star_key_system.py            #   Dual-side key generation
│   ├── test_key_generation.py             #   Key metrics
│   ├── test_secure_key_generation.py      #   Eve observation & secrecy
│   ├── test_finite_length_security.py     #   Finite block-length bounds
│   ├── test_operational_sweep_statistics.py  # Operational sweep stats
│   ├── test_quantized_security.py         #   Quantized adversary evaluation
│   ├── test_quantized_sweep_outputs.py    #   Quantized sweep I/O
│   ├── test_practical_key_generation.py   #   Guard-band + Cascade + BLAKE2s
│   ├── test_hardware_impairments.py       #   Hardware impairment model
│   ├── test_csi_estimation.py             #   Imperfect CSI generation
│   ├── test_imperfect_channel_sampling.py #   Channel sampling with NMSE
│   ├── test_imperfect_csi_evaluation.py   #   Imperfect CSI scenario eval
│   ├── test_imperfect_csi_sweep.py        #   CSI NMSE sweep
│   ├── test_parameter_sweep.py            #   Parameter sweep pipeline
│   ├── test_surface_power.py              #   Surface power & robust projection
│   ├── test_joint_objective.py            #   Joint objective reward
│   ├── test_action_mapping.py             #   Action mapping & projection
│   ├── test_rl_environment.py             #   RL environment
│   ├── test_td3_agent.py                  #   TD3 agent (network, buffer, updates)
│   ├── test_td3_training.py               #   TD3 training loop & evaluation
│   └── test_experiment_pipeline.py        #   Full experiment pipeline
│
├── start_here.bat                         # Windows one-click: venv + install + checks
├── requirements.txt                       # numpy, matplotlib, PyYAML, torch, pytest
├── pyproject.toml                         # Build config & pytest settings
└── README.md
```

---

## 5. Installation

### 5.1 Prerequisites

- Python ≥ 3.11
- PyTorch ≥ 2.2 (CUDA optional but recommended for training)

### 5.2 Setup (Windows)

In VS Code or terminal, from the project root:

```powershell
# Create virtual environment
py -3.11 -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (CMD)
.venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5.3 Setup (Linux / macOS)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.4 Verify Installation

```bash
# Run all check scripts (validates each module independently)
python scripts/run_all_checks.py

# Run the full test suite
python -m pytest -q
```

---

## 6. Running Experiments

### 6.1 Quick Smoke Test

Verify the full pipeline works before running long experiments:

```bash
python scripts/run_full_experiments.py --quick
```

This runs a single short training for each scenario (≈1-2 minutes on CPU, <1 minute on GPU).

### 6.2 Full Experiment Suite

The **main experiment entry point** is `scripts/run_full_experiments.py`. It trains and evaluates:

**3 Architectural Baselines** (with `full_model` configuration):
| Scenario | Active Elements | Description |
|---|---|---|
| `passive` | 0 | All-passive STAR-RIS (baseline) |
| `partially_active` | 8 / 32 | Mixed active/passive elements |
| `fully_active` | 32 / 32 | All elements active |

**7 Ablations** (all based on `partially_active`):
| Ablation | What's Disabled |
|---|---|
| `full_model` | Nothing (complete model, same as partially_active baseline) |
| `no_internal_noise` | RIS internal noise set to zero |
| `perfect_csi` | CSI NMSE ≈ -100 dB |
| `no_hardware_mismatch` | All HW impairment stds set to zero |
| `no_amplitude_phase_coupling` | Amplitude-phase coupling coefficients zeroed |
| `no_cvar` | CVaR reward disabled (mean-only) |
| `no_surface_power_penalty` | Surface power penalty weight = 0 |

#### Basic run (default config):

```bash
python scripts/run_full_experiments.py
```

This trains for 100,000 environment steps per scenario per seed (× 3 seeds × 10 scenarios = 30 runs). Expect several hours on GPU.

#### Customize:

```bash
# Fewer seeds, fewer steps (faster iteration)
python scripts/run_full_experiments.py --seeds 0 --steps 20000

# Custom config and output
python scripts/run_full_experiments.py --config configs/default.yaml --output results/my_experiment

# Skip baselines or ablations
python scripts/run_full_experiments.py --skip-baselines
python scripts/run_full_experiments.py --skip-ablations

# Force CPU
python scripts/run_full_experiments.py --device cpu
```

### 6.3 Output Structure

```
results/full_experiment_suite/
├── baselines/
│   ├── passive/
│   │   └── seed_0/
│   │       ├── checkpoints/
│   │       │   ├── best.pt              # Best validation-return checkpoint
│   │       │   └── final.pt             # Final model after training
│   │       ├── csv/
│   │       │   ├── episodes.csv         # Per-episode return & length
│   │       │   ├── critic_losses.csv    # Critic loss over gradient updates
│   │       │   ├── actor_losses.csv     # Actor loss over gradient updates
│   │       │   ├── periodic_evaluations.csv  # Evaluation metrics at intervals
│   │       │   ├── final_evaluation_steps.csv    # Per-step detailed eval
│   │       │   ├── final_evaluation_episodes.csv # Per-episode eval summary
│   │       │   └── final_evaluation_summary.csv  # Single-row aggregate
│   │       ├── figures/
│   │       │   ├── episode_return.png          # Training episode returns
│   │       │   ├── critic_loss.png             # Critic convergence
│   │       │   ├── actor_loss.png              # Actor convergence
│   │       │   ├── evaluation_return.png       # Periodic eval returns
│   │       │   ├── evaluation_kgr_bps.png      # Key generation rate (bit/s)
│   │       │   ├── evaluation_kdr.png          # Key disagreement rate
│   │       │   ├── evaluation_reciprocity.png  # Observation reciprocity
│   │       │   ├── evaluation_surface_power.png  # Surface power (W)
│   │       │   └── evaluation_cvar_reward.png  # CVaR tail reward
│   │       ├── final_evaluation_summary.json   # Key metrics ± std
│   │       └── run_metadata.json               # Full config, paths, seeds
│   ├── partially_active/
│   │   └── seed_0/  (same structure)
│   └── fully_active/
│       └── seed_0/  (same structure)
├── ablations/
│   ├── no_internal_noise/
│   │   └── seed_0/  (same structure)
│   ├── perfect_csi/
│   ├── no_hardware_mismatch/
│   ├── no_amplitude_phase_coupling/
│   ├── no_cvar/
│   └── no_surface_power_penalty/
├── aggregate/
│   ├── baseline_summary.csv             # Across-seed mean ± std per baseline
│   ├── ablation_summary.csv             # Across-seed mean ± std per ablation
│   ├── baseline_kgr_bps.png             # Baseline comparison bar charts
│   ├── baseline_kdr.png
│   ├── baseline_reciprocity.png
│   ├── baseline_surface_power.png
│   ├── baseline_episode_return.png
│   ├── ablation_kgr_bps.png             # Ablation comparison bar charts
│   ├── ablation_kdr.png
│   ├── ablation_reciprocity.png
│   ├── ablation_surface_power.png
│   └── ablation_episode_return.png
└── experiment_manifest.json             # Run metadata & summary pointers
```

### 6.4 Interpreting Key Metrics

| Metric | File | Meaning |
|---|---|---|
| **KGR (bit/s)** | `evaluation_kgr_bps.png`, `periodic_evaluations.csv` | Theoretical key generation rate. Higher is better. |
| **Raw KDR** | `evaluation_kdr.png`, `periodic_evaluations.csv` | Pre-reconciliation bit disagreement rate. Lower is better. |
| **Observation Reciprocity** | `evaluation_reciprocity.png` | Complex correlation magnitude between Alice and Bob. Higher is better. |
| **Surface Power (W)** | `evaluation_surface_power.png` | Total STAR-RIS power consumption. Lower is better. |
| **CVaR Reward** | `evaluation_cvar_reward.png` | Worst-tail reward (robustness metric). Higher is better. |
| **Episode Return** | `episode_return.png` | Cumulative reward per episode. Higher is better. |

### 6.5 Legacy Scripts

For non-RL experiments (Monte Carlo, parameter sweeps):

```bash
python scripts/run_main_experiment.py    # Basic Monte Carlo rate simulation
python scripts/run_sweep.py             # Parameter sweep over N and active ratios
```

---

## 7. Configuration Guide

All parameters are in [configs/default.yaml](configs/default.yaml). Key sections:

### 7.1 System Parameters (`system`)

| Parameter | Default | Description |
|---|---|---|
| `num_elements` | 32 | Total STAR-RIS elements |
| `num_active_elements` | 8 | Active elements (for partial mode) |
| `transmit_power` | 1.0 | Pilot transmit power |
| `user_noise_variance` | 0.01 | Receiver noise variance |
| `ris_internal_noise_variance` | 0.002 | Active element internal noise |
| `ris_output_power_budget` | 35.0 | Maximum RIS output power |
| `maximum_active_amplitude` | 3.0 | Maximum active gain a_max |

### 7.2 Channel Models (`channel`)

Each link (`alice_to_ris`, `ris_to_transmission_user`, etc.) supports:
- `model`: `rayleigh` or `rician`
- `k_factor_db`: Rician K-factor (dB)
- `large_scale_power`: Path loss scale

### 7.3 CSI Impairment (`csi`)

| Parameter | Description |
|---|---|
| `nmse_db_min/max` | NMSE range for training domain randomization |
| `evaluation_nmse_db` | Fixed NMSE values for evaluation sweeps |

### 7.4 Hardware Impairments (`hardware_impairments`)

All impairment standard deviations (static gain, directional gain, static phase, directional phase, fast jitter, amplitude-phase coupling).

### 7.5 RL Training (`td3`, `td3_training`)

| Parameter | Default | Description |
|---|---|---|
| `total_environment_steps` | 100000 | Training steps per run |
| `replay_capacity` | 200000 | Replay buffer size |
| `batch_size` | 256 | TD3 batch size |
| `policy_delay` | 2 | Actor update frequency |
| `exploration_noise_std` | 0.10 | Gaussian exploration noise |

### 7.6 Experiment Suite (`experiment_suite`)

| Parameter | Default | Description |
|---|---|---|
| `seeds` | [0, 1, 2] | Training random seeds |
| `evaluation_seed` | 500000 | Fixed evaluation seed |
| `final_evaluation_episodes` | 30 | Episodes for final evaluation |
| `partial_active_elements` | 8 | Active elements for partial mode |
| `output_directory` | `results/full_experiment_suite` | Output root |

---

## 8. Key Papers & References

The codebase implements concepts from:

- **TD3**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", ICML 2018
- **STAR-RIS**: Mu et al., "Simultaneously Transmitting and Reflecting (STAR) RIS Aided Wireless Communications", IEEE TWC 2022
- **Active RIS**: Long et al., "Active Reconfigurable Intelligent Surface-Aided Wireless Communications", IEEE TWC 2022
- **PLKG**: Various works on physical-layer key generation using RIS

---

## 9. Development

### Running Tests

```bash
# Full test suite
python -m pytest -q

# With coverage
python -m pytest --cov=active_star_ris -q

# Specific test file
python -m pytest tests/test_td3_agent.py -q
```

### Adding New Ablations

Edit `ABLATIONS` tuple in [src/active_star_ris/experiment_pipeline.py](src/active_star_ris/experiment_pipeline.py) and add the corresponding logic in `build_environment_config()`.

---

## 10. License

This project is for academic research purposes. Please cite appropriately if used in publications.
