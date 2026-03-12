import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Estilo de figura
# -------------------------------------------------------
plt.rcParams.update({
    'text.usetex': False,
    'text.latex.preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{amsfonts}
        \usepackage{bm}
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

# -------------------------------------------------------
# Parámetros del sistema
# -------------------------------------------------------
a  = 1.0
u  = 1.0
v  = 0.5
m1 = v - u          # masa 1
m2 = u * a**2 / 2   # masa 2
A  = u * a          # coeficiente off-diagonal

# Parámetro de la perturbación
U0 = 1000         # amplitud del potencial delta

# Posición del corte / impureza
x0 = 0.0

# -------------------------------------------------------
# Matriz T̂  (identidad por defecto; fácil de cambiar)
# -------------------------------------------------------
def T_matrix(tipo='I'):

    if tipo == 'I':
        T_label = r'$\hat{T}=\mathbb{I}$'
        return np.array([[1, 0],
                         [0, 1]], dtype=complex), T_label
    elif tipo == 'A':
        T_label = r'$\hat{T}=\hat{\tau}_a$'
        return np.array([[1, 0],
                         [0, 0]], dtype=complex), T_label
    elif tipo == 'B':
        T_label = r'$\hat{T}=\hat{\tau}_b$'
        return np.array([[0, 0],
                         [0, 1]], dtype=complex), T_label

tipo = 'I'
T_hat, T_label = T_matrix(tipo)

# -------------------------------------------------------
# Dispersión: kappa_+(omega) y kappa_-(omega)
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
# Construcción de matrices beta_± (2x2)
# -------------------------------------------------------
def sqrt_retarded(k, omega):
    """
    Calcula el momento efectivo combinando la raíz cuadrada 
    y la condición de contorno físicamente correcta.
    """
    s = np.sqrt(k + 0j)
    
    # Corte físico: si la energía supera el gap, estamos en el bulk (onda propagante)
    if np.abs(omega) >= np.abs(m1):
        return signo_omega(omega) * s
    else:
        # Estamos dentro del gap (onda evanescente)
        # Forzamos que la parte imaginaria sea estrictamente positiva para el decaimiento
        if np.imag(s) < 0:
            return -s
        return s

def beta_plus(omega, x, xp):
    w    = omega + 1j * eta
    kp   = kappa_plus(omega)
    
    # CORRECCIÓN 2: agregamos 'omega' como segundo argumento
    sqkp = sqrt_retarded(kp, omega)      

    d11 = (w - m1) / sqkp + m2 * sqkp
    d22 = (w + m1) / sqkp - m2 * sqkp
    off = A * np.sign(x - xp)

    return np.array([[d11, off],
                     [off, d22]], dtype=complex)

def beta_minus(omega, x, xp):
    w     = omega + 1j * eta
    km    = kappa_minus(omega)
    sqkm  = np.sqrt(np.abs(km) + 0j)
    i_sq  = 1j * sqkm

    d11 = (w - m1) / i_sq + m2 * i_sq
    d22 = (w + m1) / i_sq - m2 * i_sq
    off = A * np.sign(x - xp)

    return np.array([[d11, off],
                     [off, d22]], dtype=complex)

def green_r(omega, x, xp):
    kp    = kappa_plus(omega)
    km    = kappa_minus(omega)
    abskm = np.abs(km)
    
    # CORRECCIÓN 3: agregamos 'omega' como segundo argumento
    sqkp  = sqrt_retarded(kp, omega)         
    sqkm  = np.sqrt(abskm + 0j)

    denom = 2.0 * m2**2 * (kp + abskm)

    Bp = beta_plus (omega, x, xp)
    Bm = beta_minus(omega, x, xp)

    dist  = np.abs(x - xp)
    exp_p = np.exp(1j * sqkp * dist)  
    exp_m = np.exp(-sqkm * dist)

    return (-1j / denom) * (Bp * exp_p - Bm * exp_m)

# -------------------------------------------------------
# delta g^r con la fórmula de Dyson generalizada:
#
#   delta g^r(x,x') = -g^r(x,x0) T̂ [g^r(x0,x0) T̂ - I/U0]^{-1} g^r(x0,x')
#
# -------------------------------------------------------
def delta_green_r(omega, x, xp, M_inv):
    """
    Contribución perturbativa a g^r(x,x',omega).
    M_inv = [g^r(x0,x0) T̂ - I/U0]^{-1} se precalcula fuera.
    """
    g_x_x0   = green_r(omega, x,  x0)
    g_x0_xp  = green_r(omega, x0, xp)

    return -(g_x_x0 @ T_hat @ M_inv @ g_x0_xp)

# -------------------------------------------------------
# Grillas
# -------------------------------------------------------
x_max = 10
N_x = 100*x_max
N_omega = 500

x_vals     = np.linspace(-x_max+x0, x_max+x0, N_x)
omega_vals = np.linspace(-5 * m1, 5 * m1, N_omega)

# Pequeño broadening imaginario
eta = 2 * (omega_vals[1] - omega_vals[0])

# -------------------------------------------------------
# Loop principal: delta_rho[i_omega, i_x]
# -------------------------------------------------------
delta_rho = np.zeros((N_omega, N_x), dtype=float)
rho0_arr  = np.zeros(N_omega,        dtype=float)

print("Calculando LDOS ...")

for i_o, omega in enumerate(omega_vals):

    g_x0_x0 = green_r(omega, x0, x0)

    rho0_arr[i_o] = -np.imag(np.trace(g_x0_x0)) / np.pi

    # M = g^r(x0,x0) T̂ - I/U0  (no depende de x)
    M = g_x0_x0 @ T_hat - np.eye(2, dtype=complex) / U0

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print(f"No se pudo invertir M en omega = {omega:.6f} (paso {i_o})")
        continue

    for i_x, x in enumerate(x_vals):
        dg = delta_green_r(omega, x, x, M_inv)
        delta_rho[i_o, i_x] = -np.imag(np.trace(dg)) / np.pi

    if i_o % 50 == 0:
        print(f"  omega paso {i_o}/{N_omega}  (omega = {omega:.3f})")

print("Cálculo completado.")

# -------------------------------------------------------
# LDOS total rho = rho_0 + delta_rho
# rho_0 no depende de x → broadcast (N_omega,) → (N_omega, N_x)
# -------------------------------------------------------
rho_total = rho0_arr[:, np.newaxis] + delta_rho

# -------------------------------------------------------
# Gráficas: cortes de rho
# -------------------------------------------------------
import os
from datetime import datetime

fase = 'top' if v > u else 'triv'
fecha = datetime.now().strftime('%d%m%Y')
outdir = os.path.join('..', 'figures', 'cut_potential', f'LDOS_edgestate_{fecha}')
os.makedirs(outdir, exist_ok=True)

# Ticks del eje omega en unidades de m1
xticks_mult = np.arange(-5, 6)
xticks_pos = xticks_mult * m1
xticks_labels = [f'${int(m)}$' if m != 0 else '0' for m in xticks_mult]

# --- Figura 1: cortes rho(omega) en x = 1 y x = 8 ---
x_cuts  = [1, 8]
colors  = ['red', 'blue']
fig1, ax1 = plt.subplots(figsize=(10, 6))
for xc, col in zip(x_cuts, colors):
    idx = np.argmin(np.abs(x_vals - xc))
    ax1.plot(omega_vals, rho_total[:, idx], color=col, linewidth=1, label=rf'$x = {xc}$')
ax1.set_xlabel(r'$\omega/m_1$')
ax1.set_ylabel(r'$\rho(\omega, x)$')
ax1.set_title(
    rf'Cortes de $\rho(\omega,x)$  —  {T_label}, $U_0={U0}$, $u={u}$, $v={v}$'
)
ax1.set_xticks(xticks_pos)
ax1.set_xticklabels(xticks_labels)
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()

fname1 = os.path.join(outdir, f'rho_cuts_x_T{tipo}_U0{U0}_u{u}_v{v}_a{a}_x0{x0}_{fase}.png')
fig1.savefig(fname1, dpi=150, bbox_inches='tight')
print(f"Figura 1 guardada en {fname1}")

# --- Figura 2: corte rho(x) en omega = 0 ---
idx_w0 = np.argmin(np.abs(omega_vals - 0.0))
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(x_vals, rho_total[idx_w0, :], color='red', linewidth=1)
ax2.set_xlabel(r'$x/a$')
ax2.set_ylabel(r'$\rho(\omega=0, x)$')
ax2.set_title(
    rf'$\rho(x)$ en $\omega=0$  —  {T_label}, $U_0={U0}$, $u={u}$, $v={v}$'
)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

fname2 = os.path.join(outdir, f'rho_cut_w0_T{tipo}_U0{U0}_u{u}_v{v}_a{a}_x0{x0}_{fase}.png')
fig2.savefig(fname2, dpi=150, bbox_inches='tight')
print(f"Figura 2 guardada en {fname2}")

plt.show()