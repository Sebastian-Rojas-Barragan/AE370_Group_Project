#Imports
from functools import lru_cache
import numpy as np
import numpy.linalg as la
import scipy as sc
import sympy as sp
import matplotlib.pyplot as plt
import cantera as ct
import matplotlib.pyplot as plt

class chamber_solver:
    def __init__(self, mdot, Pt, D, phi, Ar, Ea):
        self.mdot = mdot
        self.Pt = Pt
        self.D = D
        self.phi = phi
        self.Ar = Ar
        self.Ea = Ea
        self.Ru = 8.314e3
        self.MW_H2 = 2.016
        self.MW_O2 = 31.998
        self.MW_H2O = 18.015

    def Bstar(self, p, gamma, h1, h2, h3, W1, W2, W3, V):
        """B* = (1/p) * ((gamma - 1) / gamma * V) * sum_i( h_i * W_i )

        p     : pressure [Pa]
        gamma : specific heat ratio
        h     : array of species enthalpies h_i
        W     : array of the W_i values from your equation
        """
        return (1.0 / p) * ((gamma - 1.0) / (gamma * V)) * (h1*W1 + h2*W2 + h3*W3)

    def Astar(self, R, T, p, gamma, h1, h2, h3, W1, W2, W3, V):
        """A* = (Ru T / p V) * sum_i( W_i / MW_i ) - B*"""
        return (self.Ru * T / (p * V)) * (W1/self.MW_O2 + W2/self.MW_H2 + W3/self.MW_H2O) - self.Bstar(p, gamma, h1, h2, h3, W1, W2, W3, V)

    def N(self, V, gamma, T, R):
        """Mach number squared"""
        return V**2 / (gamma*T*R)

    def density_rhs(self, V, rho, gamma, T, R, A_star):
        """d(rho)/dx"""
        N = self.N(V, gamma, T, R)
        return -rho * A_star * (1 - N / (N - 1.0))

    def temp_rhs(self, T, V, gamma, R, A_star, B_star):
        """dT/dx"""
        N = self.N(V, gamma, T, R)
        return -T * (-((gamma - 1.0) * N / (N - 1.0)) * A_star + B_star)

    def velocity_rhs(self, V, gamma, T, R, A_star):
        """dV/dx from constant-area continuity, with N = M squared"""
        N = self.N(V, gamma, T, R)
        return -V * A_star / (N - 1.0)

    def Mass_frac_rhs(self, W, rho, V):
        """dW/dx"""
        return (W) / (rho*V)

    def r(self, T, rho, Y1, Y2):
        """Marinov global rate; Ar is in (m^3/kmol)^0.5/s, Ea in J/kmol.

        Source: Marinov, Westbrook and Pitz (1995), equation (2).
        https://digital.library.unt.edu/ark:/67531/metadc794810/
        Validated for atmospheric H2-air flame speeds, phi = 0.55-1.1.
        """
        C_O2 = rho * Y1 / self.MW_O2
        C_H2 = rho * Y2 / self.MW_H2
        q = self.Ar * sp.exp(-self.Ea/(self.Ru*T)) * C_H2 * C_O2**0.5
        # q is for H2 + 0.5 O2 -> H2O; our W terms use 2 H2 + O2 -> 2 H2O.
        return q / 2

    def _system(self, R, gamma, h1, h2, h3, variable_enthalpy=False):
        """Numerical Jacobian in state order [rho, T, V, Y1, Y2, Y3].

        R, gamma, h1, h2, and h3 are held constant during differentiation.
        """
        rho, T, V, Y1, Y2, Y3 = sp.symbols("rho T V Y1 Y2 Y3", positive=True)

        if variable_enthalpy:
            h1, h2, h3 = sp.symbols("h1 h2 h3")

        P = rho*R*T

        rate = self.r(T, rho, Y1, Y2)
        # Mass production rates [kg/(m^3 s)] for O2 + 2 H2 -> 2 H2O.
        W1 = -self.MW_O2 * rate
        W2 = -2 * self.MW_H2 * rate
        W3 = 2 * self.MW_H2O * rate

        A_star = self.Astar(R, T, P, gamma, h1, h2, h3, W1, W2, W3, V)
        B_star = self.Bstar(P, gamma, h1, h2, h3, W1, W2, W3, V)

        drho_dx = self.density_rhs(V, rho, gamma, T, R, A_star)
        dT_dx = self.temp_rhs(T, V, gamma, R, A_star, B_star)
        dV_dx = self.velocity_rhs(V, gamma, T, R, A_star)
        dY1_dx = self.Mass_frac_rhs(W1, rho, V)
        dY2_dx = self.Mass_frac_rhs(W2, rho, V)
        dY3_dx = self.Mass_frac_rhs(W3, rho, V)

        variables = sp.Matrix([rho, T, V, Y1, Y2, Y3])
        rhs = sp.Matrix([drho_dx, dT_dx, dV_dx, dY1_dx, dY2_dx, dY3_dx])
        J = rhs.jacobian(variables)
        arguments = tuple(variables)
        if variable_enthalpy:
            arguments += (h1, h2, h3)
        return (sp.lambdify(arguments, rhs, "numpy"),
                sp.lambdify(arguments, J, "numpy"))

    def rhs(self, state, R, gamma, h1, h2, h3):
        """Spatial derivatives in state order [rho, T, V, Y1, Y2, Y3]."""
        f, _ = self._system(R, gamma, h1, h2, h3)
        return np.asarray(f(*state), dtype=float).reshape(6)

    def jacobian(self, state, R, gamma, h1, h2, h3):
        """Numerical state Jacobian with fixed thermodynamic properties."""
        _, J = self._system(R, gamma, h1, h2, h3)
        return np.asarray(J(*state), dtype=float)

    def eigenanalysis(self, J):
        """Return spatial eigenvalues [1/m] and right eigenvectors.

        Column i of eigenvectors corresponds to eigenvalues[i].
        Positive/negative real parts indicate local growth/decay along x.
        """
        eigenvalues, eigenvectors = la.eig(J)
        order = np.argsort(eigenvalues.real)[::-1]
        return eigenvalues[order], eigenvectors[:, order]

    def initial_conditions(self):
        ### Initial conditions
        #Assume some percentage of the propellant has been reacted at the first step

        Area = np.pi*(self.D/2)**2
        T_feed = 300.0      # K
        alpha = 0.1        # 10% of the limiting reactant consumed

        gas = ct.Solution("h2o2.yaml")

        # phi = (H2/O2) / (H2/O2)_stoich; stoichiometric molar ratio is 2.
        n_H2 = 2 * self.phi
        n_O2 = 1.0
        gas.TPX = T_feed, self.Pt, {"H2": n_H2, "O2": n_O2}
        h_feed = gas.enthalpy_mass

        # Partially reacted mixture
        reacted = alpha * min(n_H2 / 2, n_O2)
        X_reacted = {
            "H2": n_H2 - 2 * reacted,
            "O2": n_O2 - reacted,
            "H2O": 2 * reacted
        }
        gas.HPX = h_feed, self.Pt, X_reacted

        # Apply the prescribed inlet temperature before reading gas properties.
        gas.TP = 1000.0, self.Pt

        Y10 = gas.Y[gas.species_index("O2")]
        Y20 = gas.Y[gas.species_index("H2")]
        Y30 = gas.Y[gas.species_index("H2O")]
        T0 = gas.T
        rho0 = gas.density
        V0 = self.mdot/(rho0*Area)

        h = gas.partial_molar_enthalpies / gas.molecular_weights
        h1 = h[gas.species_index("O2")]
        h2 = h[gas.species_index("H2")]
        h3 = h[gas.species_index("H2O")]

        R = ct.gas_constant / gas.mean_molecular_weight  # J/(kg·K)
        gamma = gas.cp_mass / gas.cv_mass                # dimensionless

        state0 = np.array([rho0, T0, V0, Y10, Y20, Y30])
        return state0, R, gamma, h1, h2, h3

    def stiffness(self):
        state0, R, gamma, h1, h2, h3 = self.initial_conditions()
        J0 = self.jacobian(state0, R, gamma, h1, h2, h3)
        eigenvalues, eigenvectors = self.eigenanalysis(J0)

        # Ignore numerical zero modes when computing the eigenvalue ratio.
        magnitudes = np.abs(eigenvalues)
        tol = 1e-10 * np.max(magnitudes)
        nonzero = magnitudes[magnitudes > tol]
        eigenvalue_ratio = np.max(nonzero) / np.min(nonzero) if nonzero.size else np.nan

        print("Inlet Jacobian (state order: rho, T, V, Y1, Y2, Y3):")
        print(J0)
        print("Spatial eigenvalues [1/m]:")
        print(eigenvalues)
        print("Eigenvalue ratio (max |lambda| / min nonzero |lambda|):")
        print(eigenvalue_ratio)
        return J0, eigenvalues

    def stability_along_solution(self, x, states):
        """Scan every saved state using the solver's frozen-property Jacobian.

        Enthalpies are evaluated locally but held fixed in differentiation,
        as in fsolve. R and gamma retain their inlet values.
        """
        _, R, gamma, h1, h2, h3 = self.initial_conditions()
        _, J_h = self._system(R, gamma, h1, h2, h3, variable_enthalpy=True)
        gas = ct.Solution("h2o2.yaml")
        indices = [gas.species_index(name) for name in ("O2", "H2", "H2O")]
        E = np.empty((len(x), 6), dtype=complex)
        ratios = np.full(len(x), np.nan)
        dx_lim = np.full(len(x), np.inf)

        for k, state in enumerate(states):
            gas.TP = state[1], self.Pt
            h1, h2, h3 = (gas.partial_molar_enthalpies / gas.molecular_weights)[indices]
            J = np.asarray(J_h(*state, h1, h2, h3), dtype=float)
            E[k] = la.eigvals(J)
            magnitudes = np.abs(E[k])
            tol = 1e-10 * magnitudes.max()
            nonzero = magnitudes[magnitudes > tol]
            if nonzero.size:
                ratios[k] = nonzero.max() / nonzero.min()
            decaying = E[k][E[k].real < -tol]
            if decaying.size:
                dx_lim[k] = np.min(-2*decaying.real / np.abs(decaying)**2)

        growth = E.real.max(axis=1)
        k_growth = np.argmax(growth)
        locations = [("Largest real eigenvalue", k_growth)]
        if np.isfinite(dx_lim).any():
            locations.append(("Smallest decaying-mode forward Euler limit", np.argmin(dx_lim)))
        if np.isfinite(ratios).any():
            locations.append(("Largest eigenvalue magnitude ratio", np.nanargmax(ratios)))

        for label, k in locations:
            print(f"\n{label}: x = {x[k]*1e6:.6g} µm, T = {states[k, 1]:.6g} K")
            print("Eigenvalues [1/m]:", E[k])
            print("Eigenvalue ratio:", ratios[k])
            print("Decaying-mode FE limit [m]:", dx_lim[k])

        fig, (axE, axDx) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axE.plot(x*1e6, growth, label="max Re(lambda)")
        axE.plot(x*1e6, E.real.min(axis=1), label="min Re(lambda)")
        axE.axhline(0, color="black", linewidth=0.5)
        axE.set_ylabel("Re(lambda) [1/m]")
        axE.legend()
        axDx.semilogy(x*1e6, np.where(np.isfinite(dx_lim), dx_lim, np.nan),
                      label="Decaying-mode FE limit")
        axDx.semilogy(x[1:]*1e6, np.diff(x), label="Actual backward Euler step")
        axDx.set_ylabel("Step size [m]")
        axDx.set_xlabel("x (µm)")
        axDx.legend()
        for ax in (axE, axDx):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        return E, ratios, dx_lim

    def backward_euler(self, dx, state0, R, gamma, h1, h2, h3,
                       Y_tol=1e-8, max_steps=1000000, settle_tol=1e-3):
        """Stop at depletion or max |dYi/dx| <= settle_tol [1/m]."""
        x = [0.0]
        states = [np.array(state0, dtype=float)]
        k = 0
        # Compile once with enthalpies as inputs; update them each spatial step.
        f_h, J_h = self._system(R, gamma, h1, h2, h3, variable_enthalpy=True)
        gas = ct.Solution("h2o2.yaml")
        i1 = gas.species_index("O2")
        i2 = gas.species_index("H2")
        i3 = gas.species_index("H2O")

        def f(*y):
            return f_h(*y, h1, h2, h3)

        def J(*y):
            return J_h(*y, h1, h2, h3)

        print("step, x [m], Y_O2, Y_H2, Y_H2O", flush=True)
        print(f"{k}, {x[-1]:.8e}, {states[-1][3]:.8e}, {states[-1][4]:.8e}, {states[-1][5]:.8e}", flush=True)

        while states[-1][3] > Y_tol and states[-1][4] > Y_tol:
            # Hold h fixed during this implicit solve, evaluated at the old state.
            gas.TP = states[-1][1], self.Pt
            h = gas.partial_molar_enthalpies / gas.molecular_weights
            h1, h2, h3 = h[i1], h[i2], h[i3]
            dY_dx = np.asarray(f(*states[-1]), dtype=float).reshape(6)[3:]
            if np.max(np.abs(dY_dx)) <= settle_tol:
                print("Mass fractions settled.")
                break
            if k >= max_steps:
                raise RuntimeError("Reached max_steps before depletion or settling.")
            old = states[-1]

            step = dx

            def residual(y):
                return y - old - step*np.asarray(f(*y), dtype=float).reshape(6)

            def residual_jacobian(y):
                return np.eye(6) - step*np.asarray(J(*y), dtype=float)

            while True:
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    new, info, ier, msg = sc.optimize.fsolve(
                        residual, old, fprime=residual_jacobian, full_output=True
                    )
                error = np.max(np.abs(info["fvec"])/(1e-10 + 1e-8*np.abs(old)))
                if np.isfinite(error) and error <= 1 and np.isfinite(new).all() and np.all(new[:3] > 0) and np.all(new[3:] >= 0):
                    break
                step *= 0.5
                if step < 1e-16:
                    raise RuntimeError(f"Step {k+1} failed even after reducing dx. {msg}")

            k += 1
            x.append(x[-1] + step)
            states.append(new)
            #print(f"{k}, {x[-1]:.8e}, {new[1]:.8e}, {new[3]:.8e}, {new[4]:.8e}, {new[5]:.8e}", flush=True)

        return np.array(x), np.array(states)

    def solve(self, dx=1e-9, Y_tol=1e-8, max_steps=100000, settle_tol=1e-3):
        """Return positions and states up to reactant depletion or settling."""
        state0, R, gamma, h1, h2, h3 = self.initial_conditions()
        return self.backward_euler(dx, state0, R, gamma, h1, h2, h3,
                                   Y_tol, max_steps, settle_tol)

if __name__ == "__main__":
    Ar = 1.8e13 / np.sqrt(1000.0)
    Ea = 17614.0 * 8.314e3
    test = chamber_solver(5, 3e6, 0.3, 1, Ar, Ea)
    J0, eigenvalues = test.stiffness()
    x, states = test.solve(dx=1e-9, Y_tol=1e-6)
    E, ratios, dx_lim = test.stability_along_solution(x, states)

    print("Outlet state [rho, T, V, Y1, Y2, Y3]:")
    print(states[-1])

    fig, (axY, axT, axR, axV) = plt.subplots(
        4, 1, figsize=(8, 13), sharex=True
    )
    xm = x * 1e6
    axY.plot(xm, states[:, 3], label="O₂")
    axY.plot(xm, states[:, 4], label="H₂")
    axY.plot(xm, states[:, 5], label="H₂O")
    axY.set_ylabel("Mass fraction")
    axY.legend()

    axT.plot(xm, states[:, 1])
    axT.set_ylabel("T (K)")
    axR.plot(xm, states[:, 0])
    axR.set_ylabel("ρ (kg/m³)")
    axV.plot(xm, states[:, 2])
    axV.set_ylabel("V (m/s)")
    axV.set_xlabel("x (µm)")

    for ax in (axY, axT, axR, axV):
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
