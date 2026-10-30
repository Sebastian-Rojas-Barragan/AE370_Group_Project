# AE 370 — Hydrogen–Oxygen Reacting Flow

A numerical-methods project studying a stiff initial-value problem for premixed hydrogen–oxygen flow in a constant-area tube. The project uses backward Euler integration, nonlinear root solving, and Jacobian eigenvalue analysis to investigate reaction progress and numerical behavior.

The predicted reaction distance is **not a rocket combustion-chamber sizing result**: injector mixing, ignition-region heat transport, and other chamber processes are outside the model.

## Setup

The repository targets **Python 3.14**. From the repository directory:

```sh
uv venv --python 3.14
uv pip install -r requirements.txt
uv pip install ipykernel
```

Open `Analysis.ipynb` in VS Code, select the `.venv` interpreter as the notebook kernel, and run cells in order. Restart the kernel after editing `main.py` to reload the solver.

For JupyterLab instead:

```sh
uv pip install jupyterlab
uv run --no-sync jupyter lab
```

Dependencies are currently pinned in `requirements.txt`. The dependency list in `pyproject.toml` is empty, so `uv sync` alone does not install the scientific packages.

## Usage

`Analysis.ipynb` is the main analysis entry point. `main.py` defines the solver; running it by itself does not launch a simulation.

Example from a notebook or separate Python script:

```python
import numpy as np
from main import chamber_solver

Ar = 1.8e13 / np.sqrt(1000.0)
Ea = 17614.0 * 8.314e3

solver = chamber_solver(
    mdot=5.0,  # mass flow rate [kg/s]
    Pt=3e6,    # pressure used to initialize Cantera [Pa]
    D=0.3,     # tube diameter [m]
    phi=1.0,   # equivalence ratio; 1 = stoichiometric
    Ar=Ar,
    Ea=Ea,
)
J0, eigenvalues = solver.stiffness()
x, states = solver.solve(dx=1e-9, Y_tol=1e-6)
solver.plot_tube(states, x, "Stoichiometric H2/O2")
```

`x` contains axial positions in metres. The state columns are:

| Column | Variable | Units |
| --- | --- | --- |
| 0 | Density, `rho` | kg/m³ |
| 1 | Temperature, `T` | K |
| 2 | Velocity, `V` | m/s |
| 3 | Oxygen mass fraction, `Y1` | dimensionless |
| 4 | Hydrogen mass fraction, `Y2` | dimensionless |
| 5 | Water mass fraction, `Y3` | dimensionless |

Backward Euler uses `scipy.optimize.fsolve` with an analytical residual Jacobian. Failed steps are retried at smaller increments. Integration stops at reactant depletion (`Y_tol`), negligible species gradients (`settle_tol`, in m⁻¹), or an error/step limit (`max_steps`). Excess reactant can remain for rich or lean mixtures.

## Model assumptions

- One-dimensional, premixed, constant-area flow with global irreversible chemistry: `2 H2 + O2 -> 2 H2O`.
- `phi` sets the initial molar ratio `H2:O2 = 2*phi:1`.
- The initial composition assumes 10% consumption of the limiting reactant. The current code then sets the inlet to **1000 K**, overriding the enthalpy-conserving partial-burn temperature. These choices are in `initial_conditions()`.
- Cantera's `h2o2.yaml` supplies thermodynamic properties; its detailed kinetics are not used for the reaction rate.
- Species enthalpies are updated at each step's starting temperature and held fixed within that implicit solve. `R` and `gamma` remain at inlet values.
- The symbolic Jacobian holds thermodynamic properties fixed during differentiation. Its eigenvalues describe local **spatial** growth or decay.

## Kinetics and limitations

The rate follows Marinov, Westbrook, and Pitz, *Detailed and Global Chemical Kinetics Model for Hydrogen* (1995), equation (2):

```text
H2 + 0.5 O2 -> H2O
q = A exp(-17614/T) C_H2 C_O2^0.5
```

For concentrations in kmol/m³, `A = 1.8e13/sqrt(1000)` in `(m³/kmol)^0.5/s` and `Ea = 17614*Ru` in J/kmol. The function `r()` returns `q/2` to match the doubled stoichiometry of the source terms.

The [original report](https://digital.library.unt.edu/ark:/67531/metadc794810/) reports good global-model flame-speed predictions for atmospheric hydrogen–air mixtures around `phi = 0.55–1.1`. High-pressure pure H₂/O₂ calculations extrapolate beyond those conditions.

Mixing, atomization and vaporization where applicable, conduction, diffusion, and dissociation are omitted. Complete conversion to water and simplified thermodynamic properties can overpredict temperature. A converged nonlinear solve does not establish physical validity or step-size independence; compare multiple spatial step sizes before interpreting reaction distances.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Solver, inlet initialization, Jacobian analysis, and plotting |
| `Analysis.ipynb` | Main analysis notebook and example cases |
| `requirements.txt` | Pinned Python dependencies |
| `fig*.png` | Saved figures |
| `src/ae370_group_project/` | Package scaffold; its console command is not the simulation entry point |

## License

This project is distributed under the [MIT License](LICENSE).
