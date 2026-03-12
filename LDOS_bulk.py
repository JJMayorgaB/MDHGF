import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{amsfonts}
    ''',
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

# Parámetros del sistema
u = 0.4
v = 1.0
a = 1.0

m1 = v - u
m2 = u * a**2 / 2

N_omega = 1000
omega = np.linspace(-5 * m1, 5 * m1, N_omega)
delta_omega = omega[1] - omega[0]
eta = 2 * delta_omega  # 0+

def kappa_plus(w):
    disc = w**2 + 2*u*v - u**2
    return (2.0 / (u * a**2)) * (-v + np.sqrt(disc + 0j))

def kappa_minus(w):
    disc = w**2 + 2*u*v - u**2
    return (2.0 / (u * a**2)) * (-v - np.sqrt(disc + 0j))

def rho0(w):
    kp = kappa_plus(w)
    km = kappa_minus(w)
    abs_km = np.abs(km)
    sq_kp = np.sqrt(kp + 0j)
    sq_km = np.sqrt(abs_km + 0j)
    denom = np.pi * m2**2 * np.real(kp + abs_km)
    return (np.abs(w) / sq_kp - eta / sq_km) / denom

# Gráfica
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(omega, rho0(omega), color='red', linewidth=1)
ax.set_xlabel(r'$\omega / m_1$')
ax.set_ylabel(r'$\rho_0(\omega)$')
ax.set_title(rf'DOS del bulk: $u={u}$, $v={v}$, $a={a}$')

# Ticks en unidades de m1
xticks_mult = np.arange(-5, 6)
ax.set_xticks(xticks_mult * m1)
ax.set_xticklabels([f'${int(m)}$' if m != 0 else '0' for m in xticks_mult])

ax.set_xlim(omega[0], omega[-1])
ax.set_ylim(bottom=-0.05)
plt.tight_layout()

from datetime import datetime
fecha = datetime.now().strftime('%d%m%Y')
outdir = os.path.join('figures', f'LDOS_bulk_{fecha}')
os.makedirs(outdir, exist_ok=True)
filename = os.path.join(outdir, 'rho0_bulk.png')
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figura guardada en {filename}")