# Model definitions and evidence boundaries

This document defines the equations used by UFF v4.0.0. All radii are positive.
Galaxy calculations use kpc, km/s, and solar masses unless an SI unit is shown.

## 1. SPARC baryonic contribution

The reference component curves are combined in velocity-squared space:

\[
V_{\rm bar}^2(R)=
f_{\rm gas}V_{\rm gas}|V_{\rm gas}|+
\Upsilon_{\rm disk}V_{\rm disk}^2+
\Upsilon_{\rm bulge}V_{\rm bulge}^2.
\]

The gas sign is preserved because SPARC can report a negative gas contribution
at some inner radii. Stellar mass-to-light ratios scale `V²`, not `V`.

Defaults are \(\Upsilon_{\rm disk}=0.5\) and
\(\Upsilon_{\rm bulge}=0.7\). They are fitted unless
`--fixed-stellar-ml` is used.

References:

- [Lelli, McGaugh & Schombert (2016), SPARC](https://arxiv.org/abs/1606.09251)
- [Official SPARC data site](https://astroweb.case.edu/SPARC/)
- [Flynn & Cannaliato (2026), unified H I rotation-curve corpus](https://arxiv.org/abs/2604.13489) — preprint describing the signed-gas convention and a larger computational corpus

## 2. Distance and inclination nuisance scaling

For a distance ratio \(d=D/D_{\rm ref}\), catalog radii become \(R'=dR\),
and SPARC component curves obey \(V_{\rm component}^2\rightarrow
dV_{\rm component}^2\). Halo and SMBH laws are evaluated at \(R'\).

If the catalog curve was deprojected at inclination \(i_{\rm ref}\), a trial
inclination \(i\) is compared in catalog coordinates with

\[
V_{\rm model,cat}=V_{\rm model}\frac{\sin i}{\sin i_{\rm ref}}.
\]

`--fit-distance` uses the exploratory bound \(0.5\le d\le1.5\).
`--fit-inclination` uses ±15° around the supplied/catalog reference, clipped to
10°–89.5°. Publication work should replace these broad boxes with observational
priors and propagate correlated systematics.

## 3. Central SMBH

At galaxy-curve radii, a central mass enters through the weak-field law

\[
V_{\bullet}^2(R)=\frac{G M_{\bullet}}{R}.
\]

This is appropriate only when \(R\gg r_g=GM_{\bullet}/c^2\). The
`compact-object` command separately calculates the Kerr outer horizon,
equatorial photon orbit, and ISCO. A fitted SMBH mass is not credible unless
the data resolve approximately \(r_{\rm infl}=GM_{\bullet}/\sigma^2\).

## 4. NFW halo

The NFW density profile is

\[
\rho(r)=\frac{\rho_s}{x(1+x)^2},\qquad x=r/r_s.
\]

UFF parameterizes it with \(M_{200}\) and \(c_{200}=r_{200}/r_s\), where

\[
r_{200}=\left(\frac{3M_{200}}{4\pi\,200\rho_{\rm crit}}\right)^{1/3},
\quad
\rho_{\rm crit}=\frac{3H_0^2}{8\pi G}.
\]

With \(f(x)=\ln(1+x)-x/(1+x)\),

\[
M(<r)=M_{200}\frac{f(c_{200}r/r_{200})}{f(c_{200})},\qquad
V_{\rm NFW}=\sqrt{\frac{GM(<r)}{r}}.
\]

Reference: [Navarro, Frenk & White (1996/1997)](https://arxiv.org/abs/astro-ph/9611107).

## 5. Burkert halo

The cored Burkert profile is

\[
\rho(r)=\frac{\rho_0}{(1+x)(1+x^2)},\qquad x=r/r_c,
\]

with enclosed mass

\[
M(<r)=\pi\rho_0r_c^3
\left[\ln\big((1+x)^2(1+x^2)\big)-2\arctan x\right].
\]

Reference: [Burkert (1995)](https://arxiv.org/abs/astro-ph/9504041).

## 6. MOND and the RAR

For \(y=g_N/a_0\), UFF supports three algebraic boost functions
\(g=\nu(y)g_N\):

\[
\nu_{\rm simple}(y)=\frac12+\sqrt{\frac14+\frac1y},
\]

\[
\nu_{\rm standard}(y)=
\sqrt{\frac12+\frac12\sqrt{1+\frac4{y^2}}},
\]

\[
\nu_{\rm RAR}(y)=\frac{1}{1-e^{-\sqrt{y}}}.
\]

The default is \(a_0=1.2\times10^{-10}\,\mathrm{m\,s^{-2}}\). It can be
fitted with `--fit-a0`.

The `mond-efe` candidate applies a vector-aligned algebraic external-field
proxy:

\[
\mathbf g_{\rm int}\approx
\nu\!\left(\frac{|\mathbf g_N+\mathbf g_e|}{a_0}\right)
(\mathbf g_N+\mathbf g_e)-
\nu\!\left(\frac{g_e}{a_0}\right)\mathbf g_e,
\]

then uses its radial component. This is a sensitivity test, **not** an exact
AQUAL or QUMOND solution for a disk.

References:

- [McGaugh, Lelli & Schombert (2016), original RAR analysis](https://arxiv.org/abs/1609.05917)
- [Desmond (2023), underlying RAR and nuisance parameters](https://arxiv.org/abs/2303.11314)
- [Desmond, Hees & Famaey (2024), RAR/Solar-System tension](https://arxiv.org/abs/2401.04796)

## 7. UFF empirical v4 law

The repository-specific extra contribution is

\[
V_{\rm UFF}(r)=V_\infty
\sqrt{1-\frac{\arctan x}{x}}
\exp\!\left(\beta\frac{x}{1+x}\right),
\qquad x=r/r_c.
\]

The first factor is a cored pseudo-isothermal-like circular-speed law; the
exponential term is bounded because \(x/(1+x)\in(0,1)\). The fitted range is
\(-1\le\beta\le1\).

Status: **empirical research model**. It currently has no covariant action,
relativistic lensing prescription, cosmological solution, or independent
derivation of its parameter bounds. It must not be described as a completed
unified field theory.

## 8. LQG scale diagnostics

The compact-object module reports the convention

\[
\Delta=4\sqrt3\pi\gamma\ell_P^2
\]

and the dimensionless scale ratio \(\Delta/r^2\). An opt-in API helper exposes
the bookkeeping ansatz

\[
V^2=\frac{GM}{r}\left[1+\alpha(\Delta/r^2)^p\right].
\]

That ansatz is not part of any default fit and is not claimed to be a published
LQG effective metric. Different LQG black-hole programs make different
effective choices; a named metric must be implemented separately with its own
citation and limiting-case tests.

Context:

- [Perez (2017), Black Holes in Loop Quantum Gravity](https://arxiv.org/abs/1703.09149)
- [Motaharfar & Singh (2025), LQG signatures via Love numbers](https://arxiv.org/abs/2501.09151)
- [Ali & Ghosh (2026), rotating holonomy-corrected shadows](https://arxiv.org/abs/2605.28871) — recent preprint, not a consensus LQG metric

## 9. Total speeds and fitting

For Newtonian halo candidates,

\[
V_{\rm total}^2=V_{\rm bar}^2+V_\bullet^2+V_{\rm halo}^2.
\]

UFF uses independent Gaussian errors, deterministic bounded multi-start least
squares, and the normalized Gaussian log likelihood. It reports χ², reduced
χ², RMSE, AIC, AICc, and BIC. Comparisons are valid only when candidates use
the same observations and error model.

Relative criterion weights are

\[
w_i=\frac{e^{-\Delta_i/2}}{\sum_j e^{-\Delta_j/2}}.
\]

They summarize the declared candidate set; they are not probabilities that a
fundamental theory is true.

### Optional posterior sampler

`--mcmc-steps` starts multiple bounded Metropolis chains with uniform density
inside the same declared parameter boxes used by the optimizer. A full proposal
covariance is adapted during burn-in only; retained draws use a fixed kernel.
The report includes per-chain acceptance, Gelman–Rubin R-hat, an
autocorrelation-based effective sample size estimate, and marginal quantiles.

These diagnostics do not guarantee convergence. Publication work should run
longer chains, inspect trace behavior, test prior sensitivity, and use a sampler
appropriate to the posterior geometry.
