# MDHGF — Modified Dirac Hamiltonian Green Function

Notas y documentación de los códigos del proyecto MDHGF. 

## Estructura del repositorio

```
MDHGF/
├── README.md
├── cut_potential/                  ← Códigos del potencial de corte
│   ├── LDOS_bulk.py               ← DOS del bulk sin impurezas
│   ├── LDOS_cut_T.py              ← Mapa de calor rho(x,omega) con impureza
│   ├── LDOS_edgestate.py          ← Cortes 1D de rho (edge states)
│   └── LDOS_U0evolution.py        ← GIF animado variando U0
└── figures/
    └── cut_potential/
        ├── LDOS_bulk_DDMMAAAA/
        ├── LDOS_cut_T_DDMMAAAA/
        ├── LDOS_edgestate_DDMMAAAA/
        └── LDOS_U0evolution_DDMMAAAA/
```

---

## Descripción de cada código

### `cut_potential/LDOS_bulk.py` — DOS del bulk (sin impurezas)

Calcula $\rho_0(\omega)$ del modelo SSH continuo. Solo depende de $u$, $v$, $a$ y del broadening $\eta$.

**Salida:** Gráfica 1D de $\rho_0(\omega)$ vs $\omega/m_1$. Se guarda en `figures/cut_potential/LDOS_bulk_{fecha}/`.

---

### `cut_potential/LDOS_cut_T.py` — Mapa de calor $\rho(x,\omega)$

Calcula la LDOS completa $\rho(x,\omega) = \rho_0 + \delta\rho$ para un valor fijo de $U_0$ y genera mapas de calor.

**Parámetros clave a modificar:** `u`, `v`, `U0`, `tipo`, `x_max`, `N_x`, `N_omega`.

**Salida:** Figura con dos paneles:
1. Mapa de calor de $\rho(x,\omega)$
2. Mapa de calor de $\delta\rho(x,\omega)$ 

Se guarda en `figures/cut_potential/LDOS_cut_T_{fecha}/`. Nombre incluye todos los parámetros y la fase (`top` si $v>u$, `triv` si $v<u$).

---

### `cut_potential/LDOS_edgestate.py` — Cortes 1D para edge states

Mismo cálculo que `LDOS_cut_T.py` pero las gráficas son cortes 1D en lugar de mapas de calor. Pensado para la fase topológica ($v > u$) donde se buscan estados de borde.

**Salida:** Dos figuras separadas:
1. **Cortes $\rho(\omega)$** en posiciones fijas $x=1$ y $x=8$ — rojo y azul respectivamente.
2. **Corte $\rho(x)$** en $\omega=0$ — rojo. Muestra la localización espacial del estado de borde.

Se guarda en `figures/cut_potential/LDOS_edgestate_{fecha}/`.

---

### `cut_potential/LDOS_U0evolution.py` — GIF animado variando $U_0$

Genera una animación (GIF) que muestra cómo evoluciona $\rho(x,\omega)$ al barrer $U_0$.

**Optimización clave:** Precomputa todas las funciones de Green ($g^r(x_0,x_0)$, $g^r(x,x_0)$, $g^r(x_0,x)$) una sola vez, ya que no dependen de $U_0$. Solo la inversión de $M$ y el producto matricial se recalculan en cada frame.

**Mallado de $U_0$:** configurable. Actualmente usa un mallado no uniforme (pasos finos cerca de $U_0=0$ para capturar transiciones, pasos gruesos lejos).

**Salida:** GIF con dos paneles ($\rho$ total y $\delta\rho$) que se actualiza frame a frame. Se guarda en `figures/cut_potential/LDOS_U0evolution_{fecha}/`.

---

## Dependencias

- **Python 3.8+**
- **NumPy** — álgebra lineal y cálculo numérico
- **Matplotlib** — visualización y animación (`matplotlib.animation` para el GIF)
- **Pillow** — writer para guardar el GIF

## Ejecución

Desde el directorio `cut_potential/`:

```bash
cd cut_potential
python LDOS_bulk.py
python LDOS_cut_T.py
python LDOS_edgestate.py
python LDOS_U0evolution.py
```

Las figuras se guardan automáticamente en `figures/cut_potential/` con subdirectorios por fecha (formato `DDMMAAAA`).

---

## Notas importantes

- El broadening imaginario es $\eta = 2\Delta\omega$ (dos veces el paso en frecuencia). Si se aumenta `N_omega` los picos se vuelven más finos.
- `LDOS_cut_T.py` y `LDOS_edgestate.py` usan formulaciones algebraicamente equivalentes de Dyson. La diferencia es que una multiplica/divide por $U_0$; ambas dan el mismo resultado.
- La fase topológica ($v > u$) vs trivial ($v < u$) se marca automáticamente en los nombres de archivo.
- Para $U_0 \to \infty$, $M \to U_0\, g\hat{T}$ y la corrección se satura, recuperando la condición de contorno tipo hard-wall.