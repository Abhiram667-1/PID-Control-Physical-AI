# PID Control Simulation — Physical AI Assignment

A reproducible Python implementation of a discrete PID controller and the spring–mass–damper experiments.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

## Setup and run

Open a terminal in this repository folder:

```bash
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell):**
```powershell
.venv\\Scripts\\Activate.ps1
```

**Windows (Command Prompt):**
```bat
.venv\\Scripts\\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python src/run_all.py
```

The script saves all generated PNG graphs to `figures/` inside the repository, regardless of the terminal's current working directory.

## What's implemented

- A discrete PID controller coded directly in Python (no ready-made control library).
- Proportional, integral and derivative terms.
- Derivative-on-measurement option to avoid setpoint derivative kick.
- First-order filtered derivative.
- Output saturation and conditional-integration anti-windup.
- Spring–mass–damper simulation with `m = 1 kg`, `c = 10 N·s/m`, `k = 20 N/m`, and a 1 m setpoint.
- P-only gain sweep; PI, PD and PID response plots; response metrics.
- Experiments for actuator saturation, bounded sensor noise, derivative kick and slower sampling.
- Ziegler–Nichols ultimate-gain tuning demonstration for `G(s) = 1/(s+1)^3`.

## Reproducibility

A fixed random seed is used for the bounded sensor-noise experiment. Run the script to regenerate the plots. The console reports rise time, overshoot, settling time and final error for P, PI, PD and PID.

## AI assistance disclosure

An AI tool was used to help organize and review parts of the written explanation and code. The implementation and results are included in full for inspection. I will review and explain the model, equations, controller logic and plots during the technical discussion.
