import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime

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
u  = 0.5
v  = 1.0
m1 = v - u          # masa 1
m2 = u * a**2 / 2   # masa 2
A  = u * a          # coeficiente off-diagonal

# Parámetro de la perturbación (mallado no uniforme)
# Pasos de 1 en los extremos (rápido)
U0_izq = np.arange(-31, -10, 1)
# Pasos de 0.2 en el centro (suave y detallado: 5 veces más fotogramas)
U0_centro = np.arange(-10, 10, 0.05)
# Pasos de 1 en el otro extremo (rápido)
U0_der = np.arange(10, 31, 1)

# Unimos todo en un solo arreglo
U0_vals = np.concatenate((U0_izq, U0_centro, U0_der))

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
x_max = 10
N_x = 20*x_max
N_omega = 200

x_vals     = np.linspace(-x_max+x0, x_max+x0, N_x)
omega_vals = np.linspace(-5 * m1, 5 * m1, N_omega)

# Pequeño broadening imaginario
eta = 1.0 * (omega_vals[1] - omega_vals[0])

# -------------------------------------------------------
# Precomputar funciones de Green (no dependen de U0)
# -------------------------------------------------------
print("Precomputando funciones de Green ...")

# rho_0(omega) y g^r(x0,x0,omega)
rho0_arr   = np.zeros(N_omega, dtype=float)
g_x0_x0_arr = np.empty(N_omega, dtype=object)

# g^r(x, x0) y g^r(x0, x) para cada (omega, x)
g_x_x0_arr  = np.empty((N_omega, N_x), dtype=object)
g_x0_x_arr  = np.empty((N_omega, N_x), dtype=object)

for i_o, omega in enumerate(omega_vals):
    g00 = green_r(omega, x0, x0)
    g_x0_x0_arr[i_o] = g00
    rho0_arr[i_o] = -np.imag(np.trace(g00)) / np.pi

    for i_x, x in enumerate(x_vals):
        g_x_x0_arr[i_o, i_x]  = green_r(omega, x,  x0)
        g_x0_x_arr[i_o, i_x]  = green_r(omega, x0, x)

    if i_o % 50 == 0:
        print(f"  precomputo omega {i_o}/{N_omega}  (omega = {omega:.3f})")

print("Precomputación completada.")

# -------------------------------------------------------
# Función para calcular rho_total dado un U0
# -------------------------------------------------------
def compute_rho_total(U0):
    delta_rho = np.zeros((N_omega, N_x), dtype=float)
    for i_o in range(N_omega):
        g00 = g_x0_x0_arr[i_o]
        M = U0 * (g00 @ T_hat) - np.eye(2, dtype=complex)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        for i_x in range(N_x):
            dg = -U0 * (g_x_x0_arr[i_o, i_x] @ T_hat @ M_inv @ g_x0_x_arr[i_o, i_x])
            delta_rho[i_o, i_x] = -np.imag(np.trace(dg)) / np.pi
    return rho0_arr[:, np.newaxis] + delta_rho

# -------------------------------------------------------
# Configuración de la figura para la animación
# -------------------------------------------------------
fase = 'top' if v > u else 'triv'
fecha = datetime.now().strftime('%d%m%Y')
outdir = os.path.join('..', 'figures', 'cut_potential', f'LDOS_U0evolution_{fecha}')
os.makedirs(outdir, exist_ok=True)

xticks_mult = np.arange(-5, 6)
xticks_pos = xticks_mult * m1
xticks_labels = [f'${int(m)}$' if m != 0 else '0' for m in xticks_mult]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calcular primer frame para establecer escalas
rho_first = compute_rho_total(U0_vals[0])

# --- Panel izquierdo: LDOS total ---
ax_l = axes[0]
im_l = ax_l.imshow(
    rho_first.T, origin='lower', aspect='auto',
    extent=[omega_vals[0], omega_vals[-1], x_vals[0], x_vals[-1]],
    cmap='inferno'
)
cbar_l = fig.colorbar(im_l, ax=ax_l)
cbar_l.set_label(r'$\rho(x,\omega)$', rotation=270, labelpad=15)
ax_l.set_xlabel(r'$\omega/m_1$')
ax_l.set_ylabel(r'$x/a$')
ax_l.set_xticks(xticks_pos)
ax_l.set_xticklabels(xticks_labels)

# --- Panel derecho: delta_rho ---
ax_r = axes[1]
im_r = ax_r.imshow(
    (rho_first - rho0_arr[:, np.newaxis]).T,
    origin='lower', aspect='auto',
    extent=[omega_vals[0], omega_vals[-1], x_vals[0], x_vals[-1]],
    cmap='seismic'
)
cbar_r = fig.colorbar(im_r, ax=ax_r)
cbar_r.set_label(r'$\delta\rho(x,\omega)$', rotation=270, labelpad=15)
ax_r.set_xlabel(r'$\omega/m_1$')
ax_r.set_ylabel(r'$x/a$')
ax_r.set_xticks(xticks_pos)
ax_r.set_xticklabels(xticks_labels)

title = fig.suptitle('', fontsize=18)
plt.tight_layout()

# -------------------------------------------------------
# Función de animación
# -------------------------------------------------------
def update(frame_idx):
    U0 = U0_vals[frame_idx]
    print(f"  Frame {frame_idx+1}/{len(U0_vals)}  (U0 = {U0:.1f})")

    rho_total = compute_rho_total(U0)
    delta_rho = rho_total - rho0_arr[:, np.newaxis]

    # Actualizar panel izquierdo
    vmax_l = np.percentile(rho_total, 99)
    vmin_l = max(np.percentile(rho_total, 1), 0)
    im_l.set_data(rho_total.T)
    im_l.set_clim(vmin=vmin_l, vmax=vmax_l)
    ax_l.set_title(r'$\rho(x,\omega)$ = $\rho_0 + \delta\rho$')

    # Actualizar panel derecho
    vmax_r = np.percentile(np.abs(delta_rho), 99)
    if vmax_r == 0:
        vmax_r = 1e-10
    im_r.set_data(delta_rho.T)
    im_r.set_clim(vmin=-vmax_r, vmax=vmax_r)
    ax_r.set_title(r'$\delta\rho$')

    title.set_text(
        rf'LDOS con {T_label}, $U_0={U0:.0f}$, $u={u}$, $v={v}$, $a={a}$'
    )
    return [im_l, im_r, title]

# -------------------------------------------------------
# Crear y guardar GIF
# -------------------------------------------------------
print("Generando GIF ...")
anim = animation.FuncAnimation(
    fig, update, frames=len(U0_vals), interval=100, blit=False
)

gif_name = os.path.join(outdir, f'ldos_T{tipo}_U0sweep_u{u}_v{v}_a{a}_x0{x0}_{fase}.gif')
anim.save(gif_name, writer='pillow', fps=15)
print(f"GIF guardado en {gif_name}")
plt.close()