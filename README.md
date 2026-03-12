# MDHGF — Modified Dirac Hamiltonian Green Function

## Contenido del semillero

Este repositorio contiene los códigos desarrollados en el proyecto de investigación **MDHGF** (Modified Dirac Hamiltonian Green Function). 

## Estructura del repositorio

```
MDHGF/
├── README.md              ← Este archivo
├── LDOS_cut_T.py          ← LDOS con impureza finita via matriz T
├── LDOS_bulk.py           ← DOS del bulk (sin impurezas)
└── figures/               ← Directorio de salida de figuras
    ├── LDOS_cut_T_DDMMAAAA/   ← Figuras generadas por LDOS_cut_T.py
    └── LDOS_bulk_DDMMAAAA/    ← Figuras generadas por LDOS_bulk.py
```

---

## Descripción de cada código

### `LDOS_cut_T.py` — LDOS con impureza tipo delta y matriz T

**Propósito:** Calcula y grafica la densidad local de estados $\rho(x, \omega)$ en presencia de una impureza puntual en $x_0$, resuelta mediante la ecuación de Dyson con una matriz T arbitraria.

#### Parámetros configurables (al inicio del archivo)

| Parámetro | Descripción |
|-----------|-------------|
| `a` | Constante de red |
| `u` | Hopping intracell |
| `v` | Hopping intercell |
| `U0` | Amplitud del potencial delta |
| `x0` | Posición de la impureza |
| `tipo` | Tipo de matriz T: `'I'`, `'A'` o `'B'` |
| `N_x` | Número de puntos en espacio real |
| `N_omega` | Número de puntos en frecuencia |
| `x_max` | Extensión espacial (de $-x_\text{max}+x_0$ a $x_\text{max}+x_0$) |

#### Opciones de la matriz T

- **`'I'` (Identidad):** $\hat{T} = \mathbb{I}$ — la impureza acopla ambas subredes por igual.
- **`'A'` (Proyector A):** $\hat{T} = \hat{\tau}_a = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ — la impureza solo actúa sobre la subred A.
- **`'B'` (Proyector B):** $\hat{T} = \hat{\tau}_b = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ — la impureza solo actúa sobre la subred B.

#### Funciones principales

| Función | Descripción |
|---------|-------------|
| `T_matrix(tipo)` | Devuelve la matriz T 2×2 y su etiqueta LaTeX según el tipo (`'I'`, `'A'`, `'B'`). |
| `kappa_plus(omega)` | Rama positiva de la dispersión $\kappa_+(\omega)$. |
| `kappa_minus(omega)` | Rama negativa de la dispersión $\kappa_-(\omega)$. |
| `signo_omega(omega)` | Devuelve $\text{sgn}(\omega)$; para $\omega=0$ retorna $-1$ para evitar singularidades. |
| `beta_plus(omega, x, xp)` | Matriz $\beta_+(x, x')$ con elementos diagonales multiplicados por $\text{sgn}(\omega)$. |
| `beta_minus(omega, x, xp)` | Matriz $\beta_-(x, x')$. |
| `green_r(omega, x, xp)` | Función de Green retardada completa $g^r(x, x', \omega)$ (matriz 2×2). |
| `delta_green_r(omega, x, xp, M_inv)` | Corrección perturbativa $\delta g^r$ usando $M^{-1}$ precalculado. |
#### Salida

Genera una figura con **dos paneles**:

1. **Panel izquierdo:** Mapa de calor de $\rho(x, \omega) = \rho_0 + \delta\rho$ (colormap `inferno`).
2. **Panel derecho:** Mapa de calor de $\delta\rho(x, \omega)$ (colormap `seismic`, centrado en cero).

- Ejes: $\omega/m_1$ (horizontal) y $x/a$ (vertical), con ticks en unidades de $m_1$.
- Título dinámico con el tipo de $\hat{T}$, $U_0$, $u$, $v$ y $a$.
- Nombre de archivo dinámico: `ldos_T{tipo}_U0{U0}_u{u}_v{v}_a{a}_x0{x0}_{fase}.png`, donde `fase` es `top` (topológica, $v > u$) o `triv` (trivial, $v < u$).
- Se guarda en `figures/LDOS_cut_T_{fecha}/` con la fecha del día en formato `DDMMAAAA`.

---

### `LDOS_bulk.py` — DOS del bulk

**Propósito:** Calcula y grafica la densidad de estados del bulk $\rho_0(\omega)$ del modelo SSH continuo **sin impurezas** como función de la frecuencia.

#### Parámetros configurables

| Parámetro | Descripción |
|-----------|-------------|
| `u` | Hopping intracell |
| `v` | Hopping intercell |
| `a` | Constante de red |
| `N_omega` | Número de puntos en frecuencia |

#### Fórmula implementada

La DOS del bulk se calcula como:

$$\rho_0(\omega) = \frac{1}{\pi\, m_2^2\, \text{Re}(\kappa_+ + |\kappa_-|)} \left(\frac{|\omega|}{\sqrt{\kappa_+}} - \frac{\eta}{\sqrt{|\kappa_-|}}\right)$$

donde el término $\eta/\sqrt{|\kappa_-|}$ captura la contribución evanescente.

#### Salida

- Gráfica 1D de $\rho_0(\omega)$ vs $\omega/m_1$ (línea roja sobre fondo blanco).
- Ticks del eje horizontal en unidades de $m_1$.
- Se guarda en `figures/LDOS_bulk_{fecha}/rho0_bulk.png`.

---

## Dependencias

- **Python 3.8+**
- **NumPy** — álgebra lineal y cálculo numérico
- **Matplotlib** — visualización
- **LaTeX** (opcional pero recomendado) — renderizado de etiquetas en las figuras. `LDOS_bulk.py` usa `text.usetex: True` por defecto; `LDOS_cut_T.py` lo tiene en `False` actualmente.

## Ejecución

```bash
# Desde el directorio MDHGF/
python LDOS_cut_T.py
python LDOS_bulk.py
```

Las figuras se guardan automáticamente en subdirectorios dentro de `figures/` con la fecha del día.