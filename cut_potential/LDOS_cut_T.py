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
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 25,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
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

# Parámetro de la perturbación
U0 = 1000.0        # amplitud del potencial delta

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
print(f"Usando matriz T̂ tipo {tipo}: {T_label}")
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
def delta_green_r(U0, omega, x, xp, M_inv):
    """
    Contribución perturbativa a g^r(x,x',omega).
    M_inv = [U0 g^r(x0,x0) T̂ - I]^{-1} se precalcula fuera.
    """
    g_x_x0   = green_r(omega, x,  x0)
    g_x0_xp  = green_r(omega, x0, xp)

    return -U0 * (g_x_x0 @ T_hat @ M_inv @ g_x0_xp)

# -------------------------------------------------------
# Grillas
# -------------------------------------------------------
x_max = 10.0
N_x = int(20*x_max)
N_omega = 200

x_vals     = np.linspace(-x_max+x0, x_max+x0, N_x)
omega_vals = np.linspace(-5 * m1, 5 * m1, N_omega)

# Pequeño broadening imaginario
eta = 1.0 * (omega_vals[1] - omega_vals[0])

# -------------------------------------------------------
# Loop principal: delta_rho[i_omega, i_x]
# -------------------------------------------------------
delta_rho = np.zeros((N_omega, N_x), dtype=float)
rho0_arr  = np.zeros(N_omega,        dtype=float)

print("Calculando LDOS ...")

for i_o, omega in enumerate(omega_vals):

    g_x0_x0 = green_r(omega, x0, x0)

    rho0_arr[i_o] = -np.imag(np.trace(g_x0_x0)) / np.pi

    # M = U0 * g^r(x0,x0) T̂ - I  (no depende de x)
    M = U0 * (g_x0_x0 @ T_hat) - np.eye(2, dtype=complex)

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print(f"No se pudo invertir M en omega = {omega:.6f} (paso {i_o})")
        continue

    for i_x, x in enumerate(x_vals):
        dg = delta_green_r(U0, omega, x, x, M_inv)
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
# Gráfica: mapa de calor rho(x, omega)
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# --- Panel izquierdo: LDOS total ---
ax = axes[0]
vmax = np.percentile(rho_total, 99)
vmin = max(np.percentile(rho_total, 1), 0)

im = ax.imshow(
    rho_total.T,
    origin='lower',
    aspect='auto',
    extent=[omega_vals[0], omega_vals[-1], x_vals[0], x_vals[-1]],
    cmap='inferno',
    vmin=vmin,
    vmax=vmax
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\rho(x,\omega)$', rotation=270, labelpad=15)
ax.set_xlabel(r'$\omega/m_1$')
ax.set_ylabel(r'$x/a$')
ax.set_title(r'$\rho(x,\omega)$ = $\rho_0 + \delta\rho$')

# Ticks del eje x en unidades de m1
xticks_mult = np.arange(-5, 6)
xticks_pos = xticks_mult * m1
xticks_labels = [f'${int(m)}$' if m != 0 else '0' for m in xticks_mult]
ax.set_xticks(xticks_pos)
ax.set_xticklabels(xticks_labels)

# --- Panel derecho: solo delta_rho ---
ax = axes[1]
vmax2 = np.percentile(np.abs(delta_rho), 99)
im2 = ax.imshow(
    delta_rho.T,
    origin='lower',
    aspect='auto',
    extent=[omega_vals[0], omega_vals[-1], x_vals[0], x_vals[-1]],
    cmap='seismic',
    vmin=-vmax2,
    vmax= vmax2
)
cbar2 = fig.colorbar(im2, ax=ax)
cbar2.set_label(r'$\delta\rho(x,\omega)$', rotation=270, labelpad=15)
ax.set_xlabel(r'$\omega/m_1$')
ax.set_ylabel(r'$x/a$')
ax.set_title(r'$\delta\rho$')

# Ticks del eje x en unidades de m1
ax.set_xticks(xticks_pos)
ax.set_xticklabels(xticks_labels)

plt.suptitle(
    rf'LDOS con {T_label}, $U_0={U0}$, $u={u}$, $v={v}$, $a={a}$',
    fontsize=25
)
plt.tight_layout()
import os
from datetime import datetime
fase = 'top' if v > u else 'triv'
fecha = datetime.now().strftime('%d%m%Y')
outdir = os.path.join('..', 'figures', 'cut_potential', f'LDOS_cut_T_{fecha}')
os.makedirs(outdir, exist_ok=True)
filename = os.path.join(outdir, f'ldos_T{tipo}_U0{U0}_u{u}_v{v}_a{a}_x0{x0}_{fase}.png')
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figura guardada en {filename}")