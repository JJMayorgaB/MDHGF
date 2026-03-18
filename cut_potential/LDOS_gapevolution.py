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
v  = 1.0
U0 = 1000.0  # Constante

# Parámetro variable: u desde 0.1 hasta 2.0 en pasos de 0.1, excluyendo 1.0
u_vals = np.arange(0.1, 2.1, 0.05)
u_vals = u_vals[np.abs(u_vals - 1.0) > 1e-9]  # Excluir u=1.0

# Posición del corte / impureza
x0 = 0.0  # En el centro donde está el corte SSH

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
def kappa_plus(omega, u_param):
    disc = omega**2 + 2*u_param*v - u_param**2
    return (2.0 / (u_param * a**2)) * (-v + np.sqrt(disc + 0j))

def kappa_minus(omega, u_param):
    disc = omega**2 + 2*u_param*v - u_param**2
    return (2.0 / (u_param * a**2)) * (-v - np.sqrt(disc + 0j))


#se define globalmente el signo de omega para usarlo en beta_+ sin recalcularlo cada vez
def signo_omega(omega):
    return np.sign(omega) if omega != 0 else -1.0  # evita sgn(0)=0

# -------------------------------------------------------
# Construcción de matrices beta_± (2x2)
# -------------------------------------------------------
def sqrt_retarded(k, omega, m1_param):
    """
    Calcula el momento efectivo combinando la raíz cuadrada 
    y la condición de contorno físicamente correcta.
    """
    s = np.sqrt(k + 0j)
    
    # Corte físico: si la energía supera el gap, estamos en el bulk (onda propagante)
    if np.abs(omega) >= np.abs(m1_param):
        return signo_omega(omega) * s
    else:
        # Estamos dentro del gap (onda evanescente)
        # Forzamos que la parte imaginaria sea estrictamente positiva para el decaimiento
        if np.imag(s) < 0:
            return -s
        return s

def beta_plus(omega, x, xp, u_param):
    m1_param = v - u_param
    m2_param = u_param * a**2 / 2
    A_param = u_param * a
    
    w    = omega + 1j * eta
    kp   = kappa_plus(omega, u_param)
    sqkp = sqrt_retarded(kp, omega, m1_param)

    d11 = (w - m1_param) / sqkp + m2_param * sqkp
    d22 = (w + m1_param) / sqkp - m2_param * sqkp
    off = A_param * np.sign(x - xp)

    return np.array([[d11, off],
                     [off, d22]], dtype=complex)

def beta_minus(omega, x, xp, u_param):
    m1_param = v - u_param
    m2_param = u_param * a**2 / 2
    A_param = u_param * a
    
    w     = omega + 1j * eta
    km    = kappa_minus(omega, u_param)
    sqkm  = np.sqrt(np.abs(km) + 0j)
    i_sq  = 1j * sqkm

    d11 = (w - m1_param) / i_sq + m2_param * i_sq
    d22 = (w + m1_param) / i_sq - m2_param * i_sq
    off = A_param * np.sign(x - xp)

    return np.array([[d11, off],
                     [off, d22]], dtype=complex)

def green_r(omega, x, xp, u_param):
    m1_param = v - u_param
    m2_param = u_param * a**2 / 2
    
    kp    = kappa_plus(omega, u_param)
    km    = kappa_minus(omega, u_param)
    abskm = np.abs(km)
    
    sqkp  = sqrt_retarded(kp, omega, m1_param)
    sqkm  = np.sqrt(abskm + 0j)

    denom = 2.0 * m2_param**2 * (kp + abskm)

    Bp = beta_plus (omega, x, xp, u_param)
    Bm = beta_minus(omega, x, xp, u_param)

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
def delta_green_r(omega, x, xp, M_inv, u_param):
    """
    Contribución perturbativa a g^r(x,x',omega).
    M_inv = [U0 g^r(x0,x0) T̂ - I]^{-1} se precalcula fuera.
    """
    g_x_x0   = green_r(omega, x,  x0, u_param)
    g_x0_xp  = green_r(omega, x0, xp, u_param)

    return -U0 * (g_x_x0 @ T_hat @ M_inv @ g_x0_xp)

# -------------------------------------------------------
# Grillas y configuración
# -------------------------------------------------------
x_max = 10
N_x = 50*x_max
N_omega = 500

x_vals = np.linspace(-x_max+x0, x_max+x0, N_x)

# Broadening pequeño (será sobrescrito iterativamente)
eta = 0.01

print("Configurando parámetros ...")
print(f"Parámetros: v={v}, a={a}, U0={U0}")
print(f"Rango de u: {u_vals[0]:.1f} a {u_vals[-1]:.1f} (excluyendo 1.0)")
print(f"N_omega={N_omega}, N_x={N_x}")
print("m1 y eta se recalcularán dinámicamente para cada u")
print("Configuración completada.")

# -------------------------------------------------------
# Función para calcular rho_total dado un u
# -------------------------------------------------------
def compute_rho_total(u_param):
    global eta  # Permitir que se actualice eta globalmente
    
    m1_param = v - u_param
    omega_vals_local = np.linspace(-5 * m1_param, 5 * m1_param, N_omega)
    eta = 0.1 * (omega_vals_local[1] - omega_vals_local[0])  # Recalcular eta iterativamente
    
    rho0_arr = np.zeros(N_omega, dtype=float)
    delta_rho = np.zeros((N_omega, N_x), dtype=float)
    
    for i_o, omega_raw in enumerate(omega_vals_local):
        omega = omega_raw  # ✓ Pasar omega sin broadening; se suma en beta_+/beta_-
        
        g00 = green_r(omega, x0, x0, u_param)
        rho0_arr[i_o] = -np.imag(np.trace(g00)) / np.pi
        
        # M = U0 * g^r(x0,x0) T̂ - I  (fórmula de LDOS_cut_T)
        M = U0 * (g00 @ T_hat) - np.eye(2, dtype=complex)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
            
        for i_x, x in enumerate(x_vals):
            dg = delta_green_r(omega, x, x, M_inv, u_param)
            delta_rho[i_o, i_x] = -np.imag(np.trace(dg)) / np.pi
    
    return rho0_arr[:, np.newaxis] + delta_rho, omega_vals_local, m1_param, eta

# -------------------------------------------------------
# Configuración de la figura para la animación
# -------------------------------------------------------
fecha = datetime.now().strftime('%d%m%Y')
outdir = os.path.join('..', 'figures', 'cut_potential', f'LDOS_gapevolution_{fecha}')
os.makedirs(outdir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Calcular primer frame para establecer escalas
rho_first, omega_vals_first, m1_first, eta_first = compute_rho_total(u_vals[0])

xticks_mult = np.arange(-5, 6)
xticks_pos = xticks_mult * m1_first
xticks_labels = [f'${int(m)}$' if m != 0 else '0' for m in xticks_mult]

# --- Panel izquierdo: LDOS total ---
ax_l = axes[0]
im_l = ax_l.imshow(
    rho_first.T, origin='lower', aspect='auto',
    extent=[omega_vals_first[0], omega_vals_first[-1], x_vals[0], x_vals[-1]],
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
    np.zeros((N_omega, N_x)).T,
    origin='lower', aspect='auto',
    extent=[omega_vals_first[0], omega_vals_first[-1], x_vals[0], x_vals[-1]],
    cmap='seismic'
)
cbar_r = fig.colorbar(im_r, ax=ax_r)
cbar_r.set_label(r'$\delta\rho(x,\omega)$', rotation=270, labelpad=15)
ax_r.set_xlabel(r'$\omega/m_1$')
ax_r.set_ylabel(r'$x/a$')
ax_r.set_xticks(xticks_pos)
ax_r.set_xticklabels(xticks_labels)

title = fig.suptitle('', fontsize=25, y=0.98)
plt.subplots_adjust(top=0.80, hspace=0.40)  # Espacio generoso para título global y títulos de subplots

# -------------------------------------------------------
# Función de animación
# -------------------------------------------------------
def update(frame_idx):
    u_param = u_vals[frame_idx]
    print(f"  Frame {frame_idx+1}/{len(u_vals)}  (u = {u_param:.2f})")

    rho_total, omega_vals_local, m1_param, eta_param = compute_rho_total(u_param)
    
    # Calcular rho0 para poder restar (con los mismos omega_vals_local)
    rho0 = np.zeros(N_omega, dtype=float)
    for i_o, omega_raw in enumerate(omega_vals_local):
        omega = omega_raw
        g00 = green_r(omega, x0, x0, u_param)
        rho0[i_o] = -np.imag(np.trace(g00)) / np.pi
    
    delta_rho = rho_total - rho0[:, np.newaxis]
    
    # Actualizar ticks dinámicamente según m1
    xticks_mult = np.arange(-5, 6)
    xticks_pos = xticks_mult * m1_param
    xticks_labels = [f'${int(m)}$' if m != 0 else '0' for m in xticks_mult]
    
    # Actualizar panel izquierdo
    vmax_l = np.percentile(rho_total, 99)
    vmin_l = max(np.percentile(rho_total, 1), 0)
    im_l.set_data(rho_total.T)
    im_l.set_clim(vmin=vmin_l, vmax=vmax_l)
    im_l.set_extent([omega_vals_local[0], omega_vals_local[-1], x_vals[0], x_vals[-1]])
    ax_l.set_xticks(xticks_pos)
    ax_l.set_xticklabels(xticks_labels)
    ax_l.set_title(r'$\rho(x,\omega)$ = $\rho_0 + \delta\rho$')

    # Actualizar panel derecho
    vmax_r = np.percentile(np.abs(delta_rho), 99)
    if vmax_r == 0:
        vmax_r = 1e-10
    im_r.set_data(delta_rho.T)
    im_r.set_clim(vmin=-vmax_r, vmax=vmax_r)
    im_r.set_extent([omega_vals_local[0], omega_vals_local[-1], x_vals[0], x_vals[-1]])
    ax_r.set_xticks(xticks_pos)
    ax_r.set_xticklabels(xticks_labels)
    ax_r.set_title(r'$\delta\rho$')

    phase = 'topologica' if v > u_param else 'trivial'
    title.set_text(
        f'LDOS con T={tipo}, U0={U0:.0f}, u={u_param:.2f}, m1={m1_param:.2f}, v={v:.1f}, a={a:.1f}, fase={phase}'
    )
    return [im_l, im_r, title]

# -------------------------------------------------------
# Crear y guardar animación (cambiar formato aquí)
# -------------------------------------------------------
formato = 'mp4'  # 'gif' o 'mp4'

anim = animation.FuncAnimation(
    fig, update, frames=len(u_vals), interval=100, blit=False
)

base_name = f'ldos_T{tipo}_gapevolution.1-2.0_U0{U0:.0f}_v{v}_a{a}_x0{x0}'

if formato == 'gif':
    out_file = os.path.join(outdir, base_name + '.gif')
    print("Generando GIF ...")
    anim.save(out_file, writer='pillow', fps=3, dpi=150)
elif formato == 'mp4':
    out_file = os.path.join(outdir, base_name + '.mp4')
    print("Generando MP4 ...")
    anim.save(out_file, writer='ffmpeg', fps=3, dpi=150)

print(f"Animación guardada en {out_file}")
plt.close()