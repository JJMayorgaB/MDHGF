import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# -------------------------------------------------------
# Estilo de figura
# -------------------------------------------------------
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 17,
    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
})

# -------------------------------------------------------
# Parámetros del sistema
# -------------------------------------------------------
a  = 1.0
u  = 0.5
v  = 1.0
m1 = v - u          # masa 1
m2 = u * a**2 / 2   # masa 2
A  = u * a          # coeficiente off-diagonal

print(f"Parámetros: u={u}, v={v}, a={a}, m1={m1}, m2={m2}, A={A}")

# -------------------------------------------------------
# Grillas
# -------------------------------------------------------
x_max   = 10.0
N_x     = 500
N_omega = 500

x_vals     = np.linspace(0, x_max, N_x)          # x > 0 (semi-infinito)
omega_vals = np.linspace(-5 * m1, 5 * m1, N_omega)

eta = 0.1 * (omega_vals[1] - omega_vals[0])            # broadening

# -------------------------------------------------------
# kappa_±(omega): momentos al cuadrado
# -------------------------------------------------------
def kappa_plus(omega):
    disc = omega**2 + 2*u*v - u**2
    return (2.0 / (u * a**2)) * (-v + np.sqrt(disc + 0j))

def kappa_minus(omega):
    disc = omega**2 + 2*u*v - u**2
    return (2.0 / (u * a**2)) * (-v - np.sqrt(disc + 0j))


#se define globalmente el signo de omega para usarlo en beta_+ sin recalcularlo cada vez
def signo_omega(omega):
    return np.sign(omega) if omega != 0 else -1.0  # evita sgn(0)=0

# -------------------------------------------------------
# Momentos físicos k_±
#
#   k_+ = sgn(omega)*sqrt(kappa_+)  si |omega| >= m1   (propagante)
#       = i*sqrt(|kappa_+|)          si |omega| < m1    (evanescente)
#
#   k_- = i*sqrt(|kappa_-|)          siempre evanescente
# -------------------------------------------------------
def k_plus(omega):
    kp = kappa_plus(omega)
    if np.abs(omega) >= np.abs(m1):
        # propagante: raíz real, signo determinado por velocidad de grupo
        s = np.real(np.sqrt(kp + 0j))
        return signo_omega(omega) * s
    else:
        # evanescente: parte imaginaria positiva → decae hacia +x
        return 1j * np.sqrt(np.abs(kp))

def k_minus(omega):
    # k_- siempre evanescente
    km = kappa_minus(omega)
    return 1j * np.sqrt(np.abs(km))

# -------------------------------------------------------
# M(k) = -m1 + m2*k^2  y  alpha(k) = (omega - M(k)) / (A*k)
# -------------------------------------------------------
def M_k(k):
    return -m1 + m2 * k**2

def alpha(k, omega):
    w = omega + 0j
    Mk = M_k(k)
    Ak = A * k
    return (w - Mk) / Ak

# -------------------------------------------------------
# N(k) = [1 + |alpha(k)|^2]^{-1/2}   (normalizacion)
# -------------------------------------------------------
def N_norm(k, omega):
    alp = alpha(k, omega)
    return 1.0 / np.sqrt(1.0 + np.abs(alp)**2)


# -------------------------------------------------------
# Coeficientes de reflexión
# -------------------------------------------------------
def reflection_coeffs(omega):
    kp = k_plus(omega)
    km = k_minus(omega)

    ap = alpha(kp, omega)
    am = alpha(km, omega)
    Np = N_norm(kp, omega)
    Nm = N_norm(km, omega)

    denom_pm = ap - am   # alpha_+ - alpha_-

    r_pp = (ap + am) / denom_pm
    r_pm = 2.0 * Np * ap / (Nm * (am - ap))

    r_mm = - r_pp
    r_mp = 2.0 * Nm * am / (Np * (ap - am))

    return r_pp, r_pm, r_mm, r_mp


# -------------------------------------------------------
# G^r(x,x,omega) para x > 0
# LDOS:  rho(x, omega) = -1/pi * Im{ Tr[G^r(x,omega)] }
# -------------------------------------------------------
def ldos(x, omega):
    kp = k_plus(omega)
    km = k_minus(omega)

    r_pp, r_pm, r_mm, r_mp = reflection_coeffs(omega)

    ap = alpha(kp, omega)
    am = alpha(km, omega)

    Mk_p = M_k(kp)
    Mk_m = M_k(km)

    prefactor = -1j / (2.0 * m2**2 * (kp**2 - km**2))

    term_pp = 2 * r_pp * (
        (Mk_p / kp) * np.exp(1j * 2.0 * kp * x)
        + (Mk_m / km) * np.exp(1j * 2.0 * km * x)
    )

    term_mix = (
        np.sqrt(omega + Mk_p)
        * np.sqrt(omega + Mk_m)
        * (1.0 + ap * am)
        * ((r_pm / kp) - (r_mp / km))
        * np.exp(1j * (kp + km) * x)
    )

    trace_g_r = prefactor * (term_pp + term_mix)

    ldos_val = -np.imag(trace_g_r) / np.pi

    return np.real(ldos_val)

# -------------------------------------------------------
# Loop principal
# -------------------------------------------------------
rho = np.zeros((N_omega, N_x), dtype=float)

print("Calculando LDOS ...")
for i_o, omega_raw in enumerate(omega_vals):
    omega = omega_raw + 1j * eta                     # prescripción retardada
    for i_x, x in enumerate(x_vals):
        try:
            rho[i_o, i_x] = ldos(x, omega)
        except Exception:
            rho[i_o, i_x] = 0.0

    if i_o % 100 == 0:
        print(f"  omega paso {i_o}/{N_omega}  (omega = {omega_raw:.3f})")

print("Cálculo completado.")

# -------------------------------------------------------
# Gráfica
# -------------------------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(10, 7))
ax = axes

vmax = np.percentile(rho, 99)
vmin = max(np.percentile(rho, 1), 0)

im = ax.imshow(
    rho.T,
    origin='lower',
    aspect='auto',
    extent=[omega_vals[0], omega_vals[-1], x_vals[0], x_vals[-1]],
    cmap='inferno',
    vmin=vmin,
    vmax=vmax
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\rho(x,\omega)$', rotation=270, labelpad=15)
ax.set_xlabel(r'$\omega / m_1$')
ax.set_ylabel(r'$x / a$')
ax.set_title(r'LDOS sistema semi-infinito SSH: $\rho(x,\omega)$')

# Ticks en unidades de m1
xticks_mult   = np.arange(-4, 5)
xticks_pos    = xticks_mult * m1
xticks_labels = [f'{int(m)}' if m != 0 else '0' for m in xticks_mult]
ax.set_xticks(xticks_pos)
ax.set_xticklabels(xticks_labels)

plt.suptitle(
    rf'$u={u:.1g}$, $v={v:.1g}$, $a={a:.1g}$ → $m_1={m1:.1g}$, fase topológica ($v>u$)',
    fontsize=15
)
plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
outdir = os.path.join(project_root, 'figures', 'green_semi_infinito')
os.makedirs(outdir, exist_ok=True)
fase  = 'top' if v > u else 'triv'
fecha = datetime.now().strftime('%d%m%Y')
filename = os.path.join(outdir, f'ldos_semiinfinite_u{u}_v{v}_{fase}_{fecha}.png')
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figura guardada en {filename}")