# =============================================================================
# home.py  —  Métodos Numéricos  (Nivel Producción / Defensive Programming)
# =============================================================================
# BUGS CORREGIDOS EN ESTA VERSIÓN:
#   - Gráficas con ejes negros: forzamos colores en cada figura individualmente
#     en vez de depender solo de rcParams (que Flask puede resetear entre requests)
#   - Secante/divergencia falsa: _DIVERGE_THRESH subido a 1e15; mejor manejo
#     de funciones que crecen antes de converger
#   - Bairstow deflación incorrecta: cociente era b[:n-1] pero cuando grado=3
#     produce solo 2 coefs (grado 1), saltando el caso cuadrático. Corregido
#     a b[:n-1] con verificación de longitud antes de continuar el while
#   - Bairstow raíces complejas: threshold Im < 1e-6 demasiado estricto para
#     raíces que matemáticamente son reales pero tienen error de punto flotante
#     de magnitud mayor. Subido a 1e-4
#   - Bairstow tabla completa: mostrar TODAS las filas, no solo las últimas 5
#   - Muller ea tipo: abs() de complejo ya da float, pero ea.real podía fallar
#     si ea era ya un float puro. Ahora se usa float(ea) en todos los casos
#   - _trazar_funcion: el clip 1e6 cortaba curvas de polinomios de alto grado.
#     Ahora el clip usa el rango real de y_vals visible
#   - Ejes de color negro en gráficas oscuras: se aplican colores directamente
#     en cada ax y fig, no solo en rcParams
# =============================================================================

from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, cmath, math
from sympy.parsing.latex import parse_latex

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES DE SEGURIDAD
# ─────────────────────────────────────────────────────────────────────────────
_DIV_ZERO_THRESH = 1e-12    # Denominador mínimo tolerable
_DIVERGE_THRESH  = 1e15     # FIX: era 1e10, demasiado estricto para algunas funciones
_ROUND_DIGITS    = 8
_COMPLEX_THRESH  = 1e-4     # FIX: era 1e-6, ahora más tolerante con errores de float
_MAX_ITER        = 1000     # Límite operativo para evitar respuestas enormes o cuelgues

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────────────────────────────────────
_BG     = "#0f1117"
_BG2    = "#161923"
_GRID   = "#1e2233"
_AXIS   = "#3a3f55"     # Color de los ejes (visible en fondo oscuro)
_TICK   = "#7a7f95"
_TEXT   = "#c8ccd8"
_LEGEND = "#1b1f2d"

_COLOR = {
    "biseccion":     "#4f9cf9",
    "falsa":         "#43d9a2",
    "newton":        "#f97b4f",
    "secante":       "#c792ea",
    "taylor_f":      "#4f9cf9",
    "taylor_p":      "#ffd580",
    "punto_fijo_g":  "#c792ea",
    "punto_fijo_id": "#3a3f55",
    "horner":        "#ffd580",
    "horner_newton": "#43d9a2",
    "muller":        "#f97b4f",
    "bairstow":      "#c792ea",
    "raiz":          "#43d9a2",
    "raiz_compleja": "#f97b4f",
    "limite":        "#ffd580",
}

_COLORES_RAICES = [
    "#43d9a2", "#ffd580", "#f97b4f", "#4f9cf9",
    "#c792ea", "#80cbc4", "#ffcb6b", "#cf6679",
]


# =============================================================================
# HELPERS COMPARTIDOS
# =============================================================================

def _safe_float(val):
    """
    Convierte val a float real. Si Im es significativa retorna None (señal de complejo).
    """
    try:
        c = complex(val)
        if abs(c.imag) > 1e-8:
            return None
        return float(c.real)
    except Exception:
        return None


def _fmt_num(num, digits=_ROUND_DIGITS):
    """
    Formatea número para tabla: maneja reales, complejos y strings ("---").
    NUNCA llama round() sobre un complejo directamente.
    """
    if isinstance(num, str):
        return num
    try:
        c = complex(num)
        if abs(c.imag) < 1e-8:
            return round(c.real, digits)
        signo = "+" if c.imag >= 0 else "−"
        return f"{round(c.real, 4)} {signo} {round(abs(c.imag), 4)}i"
    except Exception:
        return str(num)


def parsear_funcion(latex_str):
    """
    LaTeX de MathLive → (expresión SymPy, variable, None)
    Error               → (None, None, dict_error)
    """
    if not latex_str or not latex_str.strip():
        return None, None, {
            "error": True,
            "titulo": "🛑 Campo vacío",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Usa la pizarra virtual para escribir f(x).",
        }
    try:
        limpio = (latex_str
                  .replace(r"\mathrm{e}", "e")
                  .replace(r"\exponentialE", "e")
                  .replace(r"\cdot", "*")
                  .lower())

        expr = parse_latex(limpio)

        e_sym = sp.Symbol("e")
        if e_sym in expr.free_symbols:
            expr = expr.subs(e_sym, sp.E)

        expr = expr.replace(
            lambda node: getattr(node, "func", None) == sp.log
            and len(node.args) == 2 and node.args[1] == sp.E,
            lambda node: sp.log(node.args[0])
        )

        vars_u = [s for s in expr.free_symbols
                  if str(s) not in ("e", "pi", "E")]

        if len(vars_u) == 0:
            return None, None, {
                "error": True,
                "titulo": "🛑 Sin variable",
                "mensaje": "La expresión es constante (no depende de x).",
                "consejo": "Escribe una función que dependa de x.",
            }
        if len(vars_u) > 1:
            return None, None, {
                "error": True,
                "titulo": "🛑 Múltiples variables",
                "mensaje": f"Detectadas: {sorted(str(v) for v in vars_u)}.",
                "consejo": "Solo se admite UNA variable independiente.",
            }

        return expr, vars_u[0], None

    except Exception as exc:
        return None, None, {
            "error": True,
            "titulo": "🛑 Sintaxis inválida",
            "mensaje": str(exc)[:200],
            "consejo": "Usa el teclado virtual. Revisa paréntesis y operadores.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html",
        }


def _error_parametro(mensaje, consejo=None):
    """Error homogéneo para entradas numéricas inválidas."""
    return {
        "error": True,
        "titulo": "⚠️ Entrada inválida",
        "mensaje": mensaje,
        "consejo": consejo or "Revisa los valores numéricos e intenta nuevamente.",
    }


def _validar_reales(**valores):
    """Valida que cada parámetro sea convertible a float finito."""
    for nombre, valor in valores.items():
        try:
            numero = float(valor)
        except (TypeError, ValueError, OverflowError):
            return _error_parametro(f"{nombre} debe ser un número real.")
        if not math.isfinite(numero):
            return _error_parametro(f"{nombre} debe ser finito; no se acepta NaN ni infinito.")
        if abs(numero) > _DIVERGE_THRESH:
            return _error_parametro(
                f"{nombre} supera el rango seguro ({_DIVERGE_THRESH:.0e}).",
                "Usa un valor inicial o intervalo de escala razonable para el método.",
            )
    return None


def _validar_tol_iter(tol, max_iter):
    """Valida tolerancia positiva y cantidad de iteraciones razonable."""
    err = _validar_reales(tol=tol)
    if err:
        return err

    if float(tol) <= 0:
        return _error_parametro(
            "La tolerancia debe ser mayor que 0.",
            "Usa un valor como 0.01 para representar porcentaje de error.",
        )

    try:
        max_iter_f = float(max_iter)
    except (TypeError, ValueError, OverflowError):
        return _error_parametro("Máx. iteraciones debe ser un número entero.")

    if not math.isfinite(max_iter_f) or not max_iter_f.is_integer():
        return _error_parametro("Máx. iteraciones debe ser un entero finito.")

    max_iter_i = int(max_iter_f)
    if max_iter_i < 1:
        return _error_parametro("Máx. iteraciones debe ser al menos 1.")
    if max_iter_i > _MAX_ITER:
        return _error_parametro(
            f"Máx. iteraciones no puede superar {_MAX_ITER}.",
            "Usa un límite menor para mantener la respuesta manejable.",
        )
    return None


def _validar_entero_rango(valor, nombre, minimo, maximo):
    try:
        numero = float(valor)
    except (TypeError, ValueError, OverflowError):
        return _error_parametro(f"{nombre} debe ser un número entero.")

    if not math.isfinite(numero) or not numero.is_integer():
        return _error_parametro(f"{nombre} debe ser un entero finito.")

    numero_i = int(numero)
    if numero_i < minimo or numero_i > maximo:
        return _error_parametro(
            f"{nombre} debe estar entre {minimo} y {maximo}.",
            f"Usa un valor entero del rango {minimo}–{maximo}.",
        )
    return None


def _error_formulario(exc):
    return _error_parametro(
        f"No se pudieron leer todos los campos del formulario: {str(exc)[:120]}",
        "Completa la función y todos los parámetros numéricos requeridos.",
    )


def _validar_intervalo_continuo(f_sim, var, xl, xu):
    """Detecta discontinuidades internas antes de aplicar Bolzano."""
    a, b = sorted((float(xl), float(xu)))
    try:
        singulares = sp.calculus.util.singularities(f_sim, var)
        dentro = singulares.intersect(sp.Interval.open(a, b))
    except Exception:
        return None

    if dentro != sp.EmptySet:
        return {
            "error": True,
            "titulo": "⚠️ Discontinuidad en el intervalo",
            "mensaje": (
                f"La función no es continua en ({a}, {b}); se detectó "
                f"singularidad en {str(dentro)[:120]}."
            ),
            "consejo": "Divide el intervalo y evita asíntotas antes de aplicar Bolzano.",
        }
    return None


def _eval_seguro(f_lam, punto, nombre="x"):
    """
    Evalúa f_lam(punto) con protección completa contra:
      - Excepciones de Python (ZeroDivisionError, OverflowError, etc.)
      - Valores infinitos / NaN
      - Resultados complejos cuando se espera real
    Retorna (float, None) o (None, dict_error).
    """
    try:
        with np.errstate(all="ignore"):
            raw = f_lam(punto)
        val = _safe_float(raw)
        if val is None:
            return None, {
                "error": True,
                "titulo": "🛑 Resultado complejo inesperado",
                "mensaje": (f"f({nombre}={round(float(punto),6) if isinstance(punto,(int,float)) else punto})"
                            f" produjo un número complejo."),
                "consejo": "Ajusta el punto inicial para evitar raíces negativas o log de negativos.",
            }
        if not math.isfinite(val):
            return None, {
                "error": True,
                "titulo": "🛑 Dominio matemático",
                "mensaje": (f"f({nombre}={round(float(punto),6) if isinstance(punto,(int,float)) else punto})"
                            f" = {val} (∞ o NaN). La función no está definida aquí."),
                "consejo": "Evita log(0), 1/0, √(negativo) en el punto evaluado.",
            }
        return val, None
    except Exception as exc:
        return None, {
            "error": True,
            "titulo": "🛑 Error de evaluación",
            "mensaje": f"f({nombre}={punto}) falló: {str(exc)[:150]}",
            "consejo": "Revisa el dominio de la función.",
        }


def _estilo_ax(ax, fig):
    """
    FIX EJES NEGROS: Aplica el estilo oscuro DIRECTAMENTE en el ax/fig,
    no solo en rcParams. Esto garantiza que Flask no resetee los colores
    entre requests.
    """
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    for spine in ax.spines.values():
        spine.set_edgecolor(_AXIS)
        spine.set_linewidth(0.8)

    ax.tick_params(colors=_TICK, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)

    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.axhline(0, color=_AXIS, linewidth=1.2, zorder=1)
    ax.axvline(0, color=_AXIS, linewidth=0.8, zorder=1, alpha=0.5)


def _hacer_figura():
    """Crea fig, ax con estilo oscuro garantizado."""
    fig, ax = plt.subplots(figsize=(9, 4))
    _estilo_ax(ax, fig)
    return fig, ax


def _trazar_funcion(ax, f_lam, x_min, x_max, color, label, n=700):
    """
    Traza f_lam en [x_min, x_max] con protección completa.
    FIX: clip dinámico basado en percentil, no en valor fijo 1e6.
    """
    try:
        xs = np.linspace(x_min, x_max, n)
        with np.errstate(all="ignore"):
            raw = f_lam(xs)

        # Escalar si lambdify devolvió escalar (función constante)
        if isinstance(raw, (int, float, complex)):
            raw = np.full(n, complex(raw).real)

        # Convertir a float real, enmascarar complejos y NaN
        ys = np.array([
            float(complex(v).real) if (not isinstance(v, float) or math.isfinite(v))
            and abs(complex(v).imag) < abs(complex(v).real) * 0.01 + 1
            else np.nan
            for v in raw
        ], dtype=float)

        # FIX: clip dinámico con percentil 99 en vez de valor fijo
        finitos = ys[np.isfinite(ys)]
        if len(finitos) > 10:
            p01, p99 = np.percentile(finitos, 1), np.percentile(finitos, 99)
            rango = max(abs(p99 - p01), 1.0)
            ys = np.where(
                (ys < p01 - 3 * rango) | (ys > p99 + 3 * rango),
                np.nan, ys
            )

        ax.plot(xs, ys, color=color, linewidth=2.2, label=label, zorder=3)
        return True
    except Exception:
        return False


def _grafica_b64(fig):
    """Serializa figura a PNG base64 y cierra la figura."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                    facecolor=_BG)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    finally:
        plt.close(fig)


def _resultado_raiz_extremo(f_lam, var, xl, xu, fxl_v, fxu_v, color, convergencia):
    """Construye una respuesta exitosa cuando un método cerrado recibe la raíz en un extremo."""
    raiz = float(xl) if abs(fxl_v) < _DIV_ZERO_THRESH else float(xu)
    fxr = fxl_v if abs(fxl_v) < _DIV_ZERO_THRESH else fxu_v
    x_min, x_max = min(float(xl), float(xu)), max(float(xl), float(xu))
    margen = max((x_max - x_min) * 0.4, 1.5)

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_lam, x_min - margen, x_max + margen, color, f"f({var})")
    ax.axvline(float(xl), color=_COLOR["limite"], ls="--", lw=1.2, alpha=0.8,
               label=f"xl = {xl}")
    ax.axvline(float(xu), color=_COLOR["limite"], ls="--", lw=1.2, alpha=0.8,
               label=f"xu = {xu}")
    ax.plot(raiz, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Raíz exacta = {round(raiz, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "cerrado",
        "resultados": [{
            "iteracion": 0,
            "xl":  _fmt_num(xl),      "xu":  _fmt_num(xu),
            "xr":  _fmt_num(raiz),    "fxl": _fmt_num(fxl_v),
            "fxr": _fmt_num(fxr),     "ea":  0,
        }],
        "raiz": _fmt_num(raiz),
        "convergencia": f"Raíz exacta detectada en un extremo. {convergencia}",
        "grafica": _grafica_b64(fig),
    }


def _resultado_raiz_inicial(f_lam, var, x0, color, tipo, fila, convergencia):
    """Construye una respuesta exitosa cuando una semilla ya es raíz exacta."""
    x0 = float(x0)
    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_lam, x0 - 2.0, x0 + 2.0, color, f"f({var})")
    ax.axvline(x0, color=_COLOR["limite"], ls=":", lw=1.2, alpha=0.7,
               label=f"x₀ = {x0}")
    ax.plot(x0, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Raíz exacta = {round(x0, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": tipo,
        "resultados": [fila],
        "raiz": _fmt_num(x0),
        "convergencia": f"Raíz exacta detectada en la semilla inicial. {convergencia}",
        "grafica": _grafica_b64(fig),
    }


def _extraer_coeficientes(f_sim, x):
    """Extrae coeficientes usando el motor Poly de SymPy (100% robusto)."""
    try:
        polinomio = sp.Poly(f_sim, x)
        coefs = [float(c) for c in polinomio.all_coeffs()]
        return coefs, polinomio.degree()
    except Exception:
        raise ValueError("La expresión no es un polinomio finito.")


# =============================================================================
# HELPERS — SISTEMAS DE ECUACIONES
# =============================================================================
def _parse_numero_matriz(token):
    token = str(token).strip()
    if not token:
        raise ValueError("valor vacío")
    try:
        val = float(token)
    except ValueError:
        val = float(sp.N(sp.sympify(token.replace("^", "**"))))
    if not math.isfinite(val) or abs(val) > _DIVERGE_THRESH:
        raise ValueError(f"valor fuera de rango: {token}")
    return val


def _parse_matriz_aumentada(texto, max_n=10):
    if not texto or not texto.strip():
        return None, None, _error_parametro(
            "La matriz aumentada está vacía.",
            "Escribe una fila por línea con los coeficientes y el término independiente.",
        )

    filas = []
    for raw in texto.replace(";", "\n").splitlines():
        linea = raw.strip()
        if not linea:
            continue
        linea = linea.replace("|", " ").replace(",", " ")
        try:
            filas.append([_parse_numero_matriz(tok) for tok in linea.split()])
        except Exception as exc:
            return None, None, _error_parametro(
                f"No se pudo leer la fila '{raw.strip()}': {str(exc)[:120]}",
                "Usa números separados por espacios. Ejemplo: 2 1 -1 8",
            )

    if not filas:
        return None, None, _error_parametro("No se detectaron filas numéricas.")

    ancho = len(filas[0])
    if ancho < 2:
        return None, None, _error_parametro("Cada fila debe tener al menos un coeficiente y b.")
    if any(len(f) != ancho for f in filas):
        return None, None, _error_parametro(
            "Todas las filas deben tener la misma cantidad de columnas.",
            "Revisa que no falte ningún coeficiente o término independiente.",
        )

    n = len(filas)
    if n > max_n:
        return None, None, _error_parametro(
            f"El sistema supera el tamaño máximo permitido ({max_n} ecuaciones)."
        )
    if ancho != n + 1:
        return None, None, _error_parametro(
            f"La matriz aumentada debe tener {n + 1} columnas para {n} ecuaciones.",
            "Formato: a11 a12 ... a1n b1",
        )

    aug = np.array(filas, dtype=float)
    return aug[:, :-1], aug[:, -1], None


def _fmt_vec(vec):
    return [_fmt_num(v) for v in np.asarray(vec, dtype=float).tolist()]


def _fmt_matrix(mat):
    return [[_fmt_num(v) for v in fila] for fila in np.asarray(mat, dtype=float).tolist()]


def _residuo_lineal(A, x, b):
    return float(np.linalg.norm(A @ x - b, ord=np.inf))


def _resultado_sistema_lineal(metodo, solucion, A, b, pasos, matriz_final=None, extras=None):
    extras = extras or {}
    return {
        "error": False,
        "tipo": "sistema_lineal",
        "metodo": metodo,
        "solucion": _fmt_vec(solucion),
        "raiz": ", ".join(f"x{i+1}={_fmt_num(v)}" for i, v in enumerate(solucion)),
        "convergencia": (
            f"Sistema resuelto por {metodo}. "
            f"Residuo máximo ||Ax-b||∞ = {_fmt_num(_residuo_lineal(A, solucion, b))}."
        ),
        "pasos": pasos,
        "matriz_final": _fmt_matrix(matriz_final) if matriz_final is not None else None,
        "extras": extras,
    }


def _parse_variables_sistema(texto_vars):
    if not texto_vars or not texto_vars.strip():
        return None, _error_parametro(
            "Debes indicar las variables del sistema no lineal.",
            "Ejemplo: x,y",
        )

    nombres = [
        v.strip().lower()
        for v in texto_vars.replace(";", ",").replace(" ", ",").split(",")
        if v.strip()
    ]
    if len(nombres) != len(set(nombres)):
        return None, _error_parametro("Las variables no deben repetirse.")
    if not nombres or len(nombres) > 6:
        return None, _error_parametro("Usa entre 1 y 6 variables.")

    try:
        simbolos = [sp.Symbol(nombre) for nombre in nombres]
    except Exception:
        return None, _error_parametro("Los nombres de variables no son válidos.")
    return simbolos, None


def _parse_vector_inicial(texto, n):
    if not texto or not texto.strip():
        return None, _error_parametro("La semilla inicial está vacía.")
    try:
        vals = [
            _parse_numero_matriz(tok)
            for tok in texto.replace(";", " ").replace(",", " ").split()
        ]
    except Exception as exc:
        return None, _error_parametro(f"No se pudo leer la semilla: {str(exc)[:120]}")
    if len(vals) != n:
        return None, _error_parametro(
            f"La semilla debe tener {n} valores.",
            "Escribe un valor por cada variable, separados por coma o espacio.",
        )
    return np.array(vals, dtype=float), None


def _parse_expr_sistema(linea, variables):
    local = {
        str(v): v for v in variables
    }
    local.update({
        "e": sp.E, "E": sp.E, "pi": sp.pi,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
        "exp": sp.exp,
    })
    limpio = (linea.strip()
              .replace(r"\mathrm{e}", "e")
              .replace(r"\exponentialE", "e")
              .replace(r"\cdot", "*")
              .lower())
    try:
        expr = parse_latex(limpio)
    except Exception:
        expr = sp.sympify(limpio.replace("^", "**"), locals=local)

    e_sym = sp.Symbol("e")
    if e_sym in expr.free_symbols:
        expr = expr.subs(e_sym, sp.E)
    expr = expr.replace(
        lambda node: getattr(node, "func", None) == sp.log
        and len(node.args) == 2 and node.args[1] == sp.E,
        lambda node: sp.log(node.args[0])
    )
    return expr


def _parse_funciones_sistema(texto_funciones, variables):
    if not texto_funciones or not texto_funciones.strip():
        return None, _error_parametro("Debes escribir las funciones del sistema.")

    lineas = [ln.strip() for ln in texto_funciones.splitlines() if ln.strip()]
    if len(lineas) != len(variables):
        return None, _error_parametro(
            f"Se esperaban {len(variables)} funciones, pero se recibieron {len(lineas)}.",
            "Escribe una ecuación fᵢ(x)=0 por línea.",
        )

    permitidas = set(variables)
    exprs = []
    for i, linea in enumerate(lineas, start=1):
        try:
            expr = _parse_expr_sistema(linea, variables)
        except Exception as exc:
            return None, _error_parametro(
                f"No se pudo interpretar f{i}: {str(exc)[:160]}",
                "Puedes escribir expresiones como x^2 + y^2 - 4.",
            )
        libres = expr.free_symbols - permitidas
        if libres:
            return None, _error_parametro(
                f"f{i} contiene variables no declaradas: {sorted(str(v) for v in libres)}."
            )
        exprs.append(expr)
    return exprs, None


def _grafica_newton_sistemas(exprs, variables, puntos):
    if len(exprs) != 2 or len(variables) != 2 or len(puntos) == 0:
        return None
    try:
        xs = np.array([p[0] for p in puntos], dtype=float)
        ys = np.array([p[1] for p in puntos], dtype=float)
        cx = float(xs[-1])
        cy = float(ys[-1])
        margen = max(float(np.max(np.abs(xs - cx))) * 1.4,
                     float(np.max(np.abs(ys - cy))) * 1.4, 2.0)
        x_min, x_max = cx - margen, cx + margen
        y_min, y_max = cy - margen, cy + margen
        gx, gy = np.meshgrid(np.linspace(x_min, x_max, 250),
                             np.linspace(y_min, y_max, 250))
        f1 = sp.lambdify(variables, exprs[0], "numpy")
        f2 = sp.lambdify(variables, exprs[1], "numpy")
        with np.errstate(all="ignore"):
            z1 = np.asarray(f1(gx, gy), dtype=float)
            z2 = np.asarray(f2(gx, gy), dtype=float)

        fig, ax = _hacer_figura()
        ax.contour(gx, gy, z1, levels=[0], colors=[_COLOR["newton"]], linewidths=2.0)
        ax.contour(gx, gy, z2, levels=[0], colors=[_COLOR["secante"]], linewidths=2.0)
        ax.plot(xs, ys, "o-", color=_COLOR["raiz"], lw=1.4, ms=5, label="Iteraciones")
        ax.plot(xs[-1], ys[-1], "o", color=_COLOR["limite"], ms=9, label="Solución")
        ax.set_xlabel(str(variables[0]))
        ax.set_ylabel(str(variables[1]))
        ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
        fig.tight_layout()
        return _grafica_b64(fig)
    except Exception:
        return None

# =============================================================================
# MÉTODO 1 — BISECCIÓN
# =============================================================================
def metodo_biseccion(latex_str, xl, xu, tol, max_iter):
    """
    Bisección clásica. Requiere Bolzano: f(xl)·f(xu) < 0.
    Convergencia lineal — error se reduce a la mitad por iteración.
    """
    err = _validar_reales(xl=xl, xu=xu)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    xl, xu, tol, max_iter = float(xl), float(xu), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    f = sp.lambdify(var, f_sim, "numpy")
    err = _validar_intervalo_continuo(f_sim, var, xl, xu)
    if err:
        return err

    fxl_v, e = _eval_seguro(f, xl, "xl")
    if e: return e
    fxu_v, e = _eval_seguro(f, xu, "xu")
    if e: return e

    if abs(fxl_v) < _DIV_ZERO_THRESH or abs(fxu_v) < _DIV_ZERO_THRESH:
        return _resultado_raiz_extremo(
            f, var, xl, xu, fxl_v, fxu_v, _COLOR["biseccion"],
            "Bisección no requiere iterar cuando f(x)=0 en el límite.",
        )

    if fxl_v * fxu_v > 0:
        return {
            "error": True,
            "titulo": "⚠️ Intervalo sin cambio de signo",
            "mensaje": (f"f({xl}) = {round(fxl_v,6)},  f({xu}) = {round(fxu_v,6)}. "
                        "Mismo signo → no garantiza raíz (Bolzano)."),
            "consejo": "Ajusta xl y xu para que f(xl)·f(xu) < 0.",
        }

    resultados = []
    xr_prev    = None
    xl0, xu0   = xl, xu
    xr         = (xl + xu) / 2

    for i in range(1, max_iter + 1):
        xr = (xl + xu) / 2

        fxr_v, e = _eval_seguro(f, xr, "xr")
        if e: return e

        # Error aproximado: relativo porcentual respecto a xr
        ea = (abs((xr - xr_prev) / xr) * 100
              if (xr_prev is not None and abs(xr) > _DIV_ZERO_THRESH) else None)

        resultados.append({
            "iteracion": i,
            "xl":  _fmt_num(xl),   "xu":  _fmt_num(xu),
            "xr":  _fmt_num(xr),   "fxl": _fmt_num(fxl_v),
            "fxr": _fmt_num(fxr_v),
            "ea":  _fmt_num(ea) if ea is not None else "---",
        })

        if ea is not None and ea < tol:
            break
        if abs(fxr_v) < _DIV_ZERO_THRESH:
            break  # raíz exacta encontrada

        if fxl_v * fxr_v < 0:
            xu    = xr
            fxu_v = fxr_v
        else:
            xl    = xr
            fxl_v = fxr_v

        xr_prev = xr

    fig, ax = _hacer_figura()
    margen = max((xu0 - xl0) * 0.4, 1.5)
    _trazar_funcion(ax, f, xl0 - margen, xu0 + margen, _COLOR["biseccion"], f"f({var})")
    ax.axvline(xl0, color=_COLOR["limite"], ls="--", lw=1.2, alpha=0.8, label=f"xl = {xl0}")
    ax.axvline(xu0, color=_COLOR["limite"], ls="--", lw=1.2, alpha=0.8, label=f"xu = {xu0}")
    ax.plot(xr, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Raíz ≈ {round(xr, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "cerrado",
        "resultados": resultados,
        "raiz": _fmt_num(xr),
        "convergencia": "Lineal — el error se reduce exactamente a la mitad por iteración.",
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 2 — REGLA FALSA
# =============================================================================
def metodo_falsa_posicion(latex_str, xl, xu, tol, max_iter):
    """
    Regla Falsa: intersección de la recta f(xl)→f(xu) con el eje X.
    Más rápida que bisección en funciones convexas; sigue requiriendo Bolzano.
    """
    err = _validar_reales(xl=xl, xu=xu)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    xl, xu, tol, max_iter = float(xl), float(xu), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    f = sp.lambdify(var, f_sim, "numpy")
    err = _validar_intervalo_continuo(f_sim, var, xl, xu)
    if err:
        return err

    fxl_v, e = _eval_seguro(f, xl, "xl")
    if e: return e
    fxu_v, e = _eval_seguro(f, xu, "xu")
    if e: return e
    xl0, xu0  = xl, xu

    if abs(fxl_v) < _DIV_ZERO_THRESH or abs(fxu_v) < _DIV_ZERO_THRESH:
        return _resultado_raiz_extremo(
            f, var, xl, xu, fxl_v, fxu_v, _COLOR["falsa"],
            "Regla Falsa no requiere interpolar cuando f(x)=0 en el límite.",
        )

    if fxl_v * fxu_v >= 0:
        return {
            "error": True,
            "titulo": "⚠️ Intervalo sin cambio de signo",
            "mensaje": f"f({xl}) = {round(fxl_v,6)},  f({xu}) = {round(fxu_v,6)}.",
            "consejo": "La Regla Falsa requiere f(xl)·f(xu) < 0.",
        }

    resultados = []
    xr_prev    = None
    xr         = xl

    for i in range(1, max_iter + 1):
        # Guard: denominador f(xu)-f(xl)
        den = fxu_v - fxl_v
        if abs(den) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Recta horizontal",
                "mensaje": "f(xl) ≈ f(xu): la recta es horizontal y no corta el eje X.",
                "consejo": "Cambia el intervalo.",
            }

        xr = (xl * fxu_v - xu * fxl_v) / den

        fxr_v, e = _eval_seguro(f, xr, "xr")
        if e: return e

        ea = (abs((xr - xr_prev) / xr) * 100
              if (xr_prev is not None and abs(xr) > _DIV_ZERO_THRESH) else None)

        resultados.append({
            "iteracion": i,
            "xl":  _fmt_num(xl),   "xu":  _fmt_num(xu),
            "xr":  _fmt_num(xr),   "fxl": _fmt_num(fxl_v),
            "fxr": _fmt_num(fxr_v),
            "ea":  _fmt_num(ea) if ea is not None else "---",
        })

        if ea is not None and ea < tol:
            break
        if abs(fxr_v) < _DIV_ZERO_THRESH:
            break

        if fxl_v * fxr_v < 0:
            xu    = xr; fxu_v = fxr_v
        else:
            xl    = xr; fxl_v = fxr_v

        xr_prev = xr

    fig, ax = _hacer_figura()
    margen = max((xu0 - xl0) * 0.4, 1.5)
    _trazar_funcion(ax, f, xl0 - margen, xu0 + margen, _COLOR["falsa"], f"f({var})")
    ax.axvline(xl0, color=_COLOR["limite"], ls="--", lw=1.2, alpha=0.8, label=f"xl = {xl0}")
    ax.axvline(xu0, color=_COLOR["limite"], ls="--", lw=1.2, alpha=0.8, label=f"xu = {xu0}")
    ax.plot(xr, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Raíz ≈ {round(xr, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "cerrado",
        "resultados": resultados,
        "raiz": _fmt_num(xr),
        "convergencia": "Lineal — más rápida que bisección en funciones convexas.",
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 3 — NEWTON-RAPHSON
# =============================================================================
def metodo_newton_raphson(latex_str, x0, tol, max_iter):
    """
    xi+1 = xi − f(xi)/f'(xi).  Derivada analítica via SymPy.
    Convergencia cuadrática — fallla si f'(xi)=0.
    """
    err = _validar_reales(x0=x0)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    x0, tol, max_iter = float(x0), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    try:
        df_sim = sp.diff(f_sim, var)
    except Exception as exc:
        return {"error": True, "titulo": "🛑 No se pudo derivar",
                "mensaje": str(exc)[:200]}

    f  = sp.lambdify(var, f_sim,  "numpy")
    df = sp.lambdify(var, df_sim, "numpy")

    fxi_v, e = _eval_seguro(f,  x0, "x0"); 
    if e: return e
    if abs(fxi_v) < _DIV_ZERO_THRESH:
        return _resultado_raiz_inicial(
            f, var, x0, _COLOR["newton"], "abierto",
            {
                "iteracion": 0,
                "xi": _fmt_num(x0),
                "fxi": _fmt_num(fxi_v),
                "dfxi": "---",
                "x_siguiente": _fmt_num(x0),
                "ea": 0,
            },
            "Newton-Raphson no requiere calcular la tangente.",
        )

    dfxi_v, e = _eval_seguro(df, x0, "x0")
    if e: return e

    resultados = []
    xi = float(x0)

    for i in range(1, max_iter + 1):
        fxi_v,  e = _eval_seguro(f,  xi, f"x{i}");  
        if e: return e
        dfxi_v, e = _eval_seguro(df, xi, f"x{i}")
        if e: return e

        if abs(dfxi_v) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Derivada = 0",
                "mensaje": f"f'(x{i} = {round(xi,6)}) ≈ 0. La tangente es horizontal.",
                "consejo": "Estás en un máximo o mínimo. Prueba otro x₀.",
            }

        x_sig = xi - fxi_v / dfxi_v

        if abs(x_sig) > _DIVERGE_THRESH:
            return {
                "error": True,
                "titulo": "🚀 Divergencia",
                "mensaje": f"xi+1 = {x_sig:.3e} supera el umbral de seguridad.",
                "consejo": "Prueba un x₀ más cercano a la raíz.",
            }

        ea = (abs((x_sig - xi) / x_sig) * 100
              if abs(x_sig) > _DIV_ZERO_THRESH else 100.0)

        resultados.append({
            "iteracion":   i,
            "xi":          _fmt_num(xi),
            "fxi":         _fmt_num(fxi_v),
            "dfxi":        _fmt_num(dfxi_v),
            "x_siguiente": _fmt_num(x_sig),
            "ea":          _fmt_num(ea) if i > 1 else "---",
        })

        xi = x_sig
        if i > 1 and ea < tol:
            break

    fig, ax = _hacer_figura()
    margen = max(abs(xi - float(x0)) * 1.5, 2.0)
    _trazar_funcion(ax, f, min(float(x0), xi) - margen,
                    max(float(x0), xi) + margen, _COLOR["newton"], f"f({var})")
    ax.axvline(float(x0), color=_COLOR["limite"], ls="--", lw=1.2, alpha=0.8,
               label=f"x₀ = {x0}")
    ax.plot(xi, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Raíz ≈ {round(xi, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "abierto",
        "resultados": resultados,
        "raiz": _fmt_num(xi),
        "convergencia": "Cuadrática — los dígitos correctos se duplican por iteración.",
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 4 — SECANTE
# =============================================================================
def metodo_secante(latex_str, x0, x1, tol, max_iter):
    """
    xi+1 = xi − f(xi)·(xi-1 − xi) / (f(xi-1) − f(xi)).
    No requiere derivada. Convergencia superlineal (orden φ ≈ 1.618).

    FIX vs versión anterior:
      - _DIVERGE_THRESH subido a 1e15 para no abortar antes de converger
      - Gráfica usa el rango real de convergencia, no el rango divergente
      - ea se guarda y compara como float, no como posible complejo
    """
    err = _validar_reales(x0=x0, x1=x1)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    x0, x1, tol, max_iter = float(x0), float(x1), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    f = sp.lambdify(var, f_sim, "numpy")

    fx0_v, e = _eval_seguro(f, x0, "x0");
    if e: return e
    fx1_v, e = _eval_seguro(f, x1, "x1")
    if e: return e

    if abs(fx0_v) < _DIV_ZERO_THRESH or abs(fx1_v) < _DIV_ZERO_THRESH:
        raiz = x0 if abs(fx0_v) < _DIV_ZERO_THRESH else x1
        fx_raiz = fx0_v if abs(fx0_v) < _DIV_ZERO_THRESH else fx1_v
        return _resultado_raiz_inicial(
            f, var, raiz, _COLOR["secante"], "secante",
            {
                "iteracion": 0,
                "x_previo": _fmt_num(x0),
                "x_actual": _fmt_num(x1),
                "fx_previo": _fmt_num(fx0_v),
                "fx_actual": _fmt_num(fx1_v),
                "x_siguiente": _fmt_num(raiz),
                "ea": 0,
            },
            f"f({raiz}) = {_fmt_num(fx_raiz)}; Secante no requiere iterar.",
        )

    if abs(x1 - x0) < _DIV_ZERO_THRESH:
        return {
            "error": True,
            "titulo": "⚠️ Semillas idénticas",
            "mensaje": "x0 y x1 son prácticamente iguales.",
            "consejo": "Usa dos valores iniciales distintos.",
        }

    resultados     = []
    x_prev, fx_prev = float(x0), fx0_v
    x_curr, fx_curr = float(x1), fx1_v

    for i in range(1, max_iter + 1):
        den = fx_curr - fx_prev

        if abs(den) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Secante horizontal",
                "mensaje": (f"f(xi-1) ≈ f(xi) = {round(fx_curr,8)}. "
                            "La recta secante es horizontal → división por cero."),
                "consejo": "Cambia las semillas x0 y x1.",
            }

        x_sig = x_curr - fx_curr * (x_curr - x_prev) / den

        # Guard divergencia: solo abortamos si REALMENTE se disparó
        if not math.isfinite(x_sig) or abs(x_sig) > _DIVERGE_THRESH:
            return {
                "error": True,
                "titulo": "🚀 Divergencia detectada",
                "mensaje": f"xi+1 = {x_sig:.3e} supera el umbral de seguridad ({_DIVERGE_THRESH:.0e}).",
                "consejo": "Prueba semillas más cercanas a la raíz real.",
            }

        fx_sig, e = _eval_seguro(f, x_sig, f"x{i+1}")
        if e: return e

        ea = float(abs((x_sig - x_curr) / x_sig) * 100
                   if abs(x_sig) > _DIV_ZERO_THRESH else 100.0)

        resultados.append({
            "iteracion":   i,
            "x_previo":    _fmt_num(x_prev),
            "x_actual":    _fmt_num(x_curr),
            "fx_previo":   _fmt_num(fx_prev),
            "fx_actual":   _fmt_num(fx_curr),
            "x_siguiente": _fmt_num(x_sig),
            "ea":          _fmt_num(ea) if i > 1 else "---",
        })

        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_sig,  fx_sig

        if i > 1 and ea < tol:
            break

    # Usamos el rango de convergencia para la gráfica, no el rango de divergencia
    x_fin  = x_curr
    margen = max(abs(x_fin - float(x0)) * 1.2, 2.0)
    x_min  = min(float(x0), float(x1), x_fin) - margen
    x_max  = max(float(x0), float(x1), x_fin) + margen

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f, x_min, x_max, _COLOR["secante"], f"f({var})")
    ax.axvline(float(x0), color=_COLOR["limite"], ls=":", lw=1.2, alpha=0.7,
               label=f"x₀ = {x0}")
    ax.axvline(float(x1), color=_COLOR["limite"], ls=":", lw=1.2, alpha=0.7,
               label=f"x₁ = {x1}")
    ax.plot(x_fin, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Raíz ≈ {round(x_fin, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "secante",
        "resultados": resultados,
        "raiz": _fmt_num(x_fin),
        "convergencia": "Superlineal (orden φ ≈ 1.618) — sin derivada analítica.",
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 5 — SERIE DE TAYLOR
# =============================================================================
def metodo_taylor(latex_str, x0, x_eval, n_terminos):
    """
    P(x) = Σ_{n=0}^{N-1} f^(n)(x0)/n! · (x−x0)^n
    Compara con el valor real f(x_eval) y grafica ambas curvas.
    """
    err = _validar_reales(x0=x0, x_eval=x_eval)
    if err:
        return err
    err = _validar_entero_rango(n_terminos, "Número de términos", 1, 20)
    if err:
        return err
    x0, x_eval, n_terminos = float(x0), float(x_eval), int(float(n_terminos))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    f_num = sp.lambdify(var, f_sim, "numpy")

    val_real, e = _eval_seguro(f_num, x_eval, "x_eval")
    if e: return e

    h = sp.Symbol("_h")
    x0_sym = sp.nsimplify(x0)
    desplazada = f_sim.subs(var, x0_sym + h)
    orden_limite = max(30, min(100, 5 * n_terminos + 5))
    try:
        serie_h = sp.series(desplazada, h, 0, orden_limite + 1).removeO().expand()
    except Exception as exc:
        return {
            "error": True,
            "titulo": "🛑 Taylor no pudo expandirse",
            "mensaje": str(exc)[:200],
            "consejo": "Prueba menos términos o un centro x0 dentro del dominio.",
        }

    terminos = []
    for potencia in range(orden_limite + 1):
        coef = sp.simplify(serie_h.coeff(h, potencia))
        coef_real = _safe_float(coef.evalf())
        if coef_real is None:
            return {
                "error": True,
                "titulo": "🛑 Coeficiente complejo",
                "mensaje": f"El coeficiente de orden {potencia} no es real en x0 = {x0}.",
                "consejo": "Elige un centro x0 donde la serie sea real.",
            }
        if abs(coef_real) > _DIV_ZERO_THRESH:
            terminos.append((potencia, coef, coef_real))
            if len(terminos) >= n_terminos:
                break

    if len(terminos) < n_terminos:
        return {
            "error": True,
            "titulo": "⚠️ Serie insuficiente",
            "mensaje": (
                f"Solo se encontraron {len(terminos)} términos no nulos "
                f"hasta el orden {orden_limite}."
            ),
            "consejo": "Usa menos términos o una función con expansión menos dispersa.",
        }

    resultados = []
    aprox_acum = 0.0
    polinomio_taylor = sp.Integer(0)

    for idx, (n, coef_simb, coef_real) in enumerate(terminos, start=1):
        val_deriv = coef_real * math.factorial(n)
        termino_val = coef_real * ((x_eval - x0) ** n)
        termino_simb = coef_simb * ((var - x0_sym) ** n)

        aprox_acum += termino_val
        polinomio_taylor += termino_simb

        et = (abs((val_real - aprox_acum) / val_real) * 100
              if abs(val_real) > _DIV_ZERO_THRESH else abs(aprox_acum))

        termino_str = str(sp.expand(termino_simb)).replace("**","^").replace("*","·")
        resultados.append({
            "orden":             n,
            "derivada":          termino_str,
            "derivada_evaluada": round(val_deriv, _ROUND_DIGITS),
            "termino_calculado": round(termino_val, _ROUND_DIGITS),
            "aproximacion":      round(aprox_acum, _ROUND_DIGITS),
            "et":                round(et, _ROUND_DIGITS) if idx > 1 else "---",
        })

    p_num   = sp.lambdify(var, polinomio_taylor, "numpy")
    pol_str = (str(sp.expand(polinomio_taylor))
               .replace("**","^").replace("*","·").replace("sqrt","√"))
    margen  = max(abs(x_eval - x0) * 1.3, 2.0)

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_num, min(x0,x_eval)-margen, max(x0,x_eval)+margen,
                    _COLOR["taylor_f"], f"f({var}) original")
    _trazar_funcion(ax, p_num, min(x0,x_eval)-margen, max(x0,x_eval)+margen,
                    _COLOR["taylor_p"], f"Taylor {n_terminos} términos")
    ax.axvline(x0, color=_COLOR["limite"], ls=":", lw=1.2, alpha=0.7,
               label=f"Centro x₀={x0}")
    ax.plot(x_eval, val_real,   "o", color=_COLOR["taylor_f"], ms=8, zorder=5,
            label="Valor real")
    ax.plot(x_eval, aprox_acum, "D", color=_COLOR["taylor_p"], ms=8, zorder=5,
            label="Aproximación")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "taylor",
        "resultados": resultados,
        "valor_verdadero": round(val_real, _ROUND_DIGITS),
        "aprox_final":     round(aprox_acum, _ROUND_DIGITS),
        "polinomio_final": pol_str,
        "grafica":         _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 6 — PUNTO FIJO (DESPEJE AUTOMÁTICO)
# =============================================================================
def metodo_punto_fijo(latex_fx_str, x0, tol, max_iter):
    """
    Genera candidatos g(x) de f(x)=0 y selecciona el que cumple |g'(x0)|<1.
    """
    err = _validar_reales(x0=x0)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    x0, tol, max_iter = float(x0), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_fx_str)
    if err:
        return err

    x = var
    candidatos = [x + f_sim, x - f_sim]

    for termino in sp.Add.make_args(f_sim):
        if termino.has(x):
            resto = f_sim - termino
            y_tmp = sp.Symbol("_y")
            try:
                for sol in sp.solve(termino - y_tmp, x):
                    candidatos.append(sol.subs(y_tmp, -resto))
            except Exception:
                pass

    candidatos_unicos = []
    vistos = set()
    for g_expr in candidatos:
        try:
            g_s = sp.simplify(g_expr)
        except Exception:
            g_s = g_expr
        clave = str(g_s)
        if clave not in vistos:
            vistos.add(clave)
            candidatos_unicos.append(g_s)

    f_num = sp.lambdify(x, f_sim, "numpy")
    residual_inicial, e = _eval_seguro(f_num, x0, "x0")
    if e:
        return e

    g_elegida = None
    menor_score = (float("inf"), float("inf"))
    deriv_elegida = float("inf")
    complejo_detectado = False

    def _salto_complejo(punto, expr="g(x)"):
        return {
            "error": True,
            "titulo": "🛑 Salto al plano complejo",
            "mensaje": f"{expr} produjo un valor no real al evaluar desde x = {_fmt_num(punto)}.",
            "consejo": "Elige otra semilla o usa un método que admita raíces complejas, como Müller.",
        }

    for g_expr in candidatos_unicos:
        try:
            g_num_tmp = sp.lambdify(x, g_expr, "numpy")
            dg = sp.diff(g_expr, x)
            dg_f = sp.lambdify(x, dg, "numpy")

            g0 = _safe_float(g_num_tmp(x0))
            dg0 = _safe_float(dg_f(x0))
            if g0 is None or dg0 is None:
                complejo_detectado = True
                continue
            if not math.isfinite(g0) or not math.isfinite(dg0):
                continue

            xi_prueba = float(x0)
            residuos = []
            estable = True
            for _ in range(min(6, max_iter)):
                gxi_prueba = _safe_float(g_num_tmp(xi_prueba))
                if gxi_prueba is None:
                    complejo_detectado = True
                    estable = False
                    break
                if not math.isfinite(gxi_prueba) or abs(gxi_prueba) > _DIVERGE_THRESH:
                    estable = False
                    break

                f_prueba = _safe_float(f_num(gxi_prueba))
                if f_prueba is None:
                    complejo_detectado = True
                    estable = False
                    break
                if not math.isfinite(f_prueba):
                    estable = False
                    break

                residuos.append(abs(f_prueba))
                xi_prueba = float(gxi_prueba)

            if not estable or not residuos:
                continue

            mejora = residuos[-1] < max(abs(residual_inicial), _DIV_ZERO_THRESH)
            contraccion_suave = abs(dg0) <= 1.05
            if mejora and contraccion_suave:
                score = (residuos[-1], abs(dg0))
                if score < menor_score:
                    menor_score = score
                    deriv_elegida = abs(dg0)
                    g_elegida = g_expr
        except Exception:
            continue

    if g_elegida is None:
        if complejo_detectado:
            return _salto_complejo(x0)
        return {
            "error": True,
            "titulo": "🛑 Sin despeje convergente",
            "mensaje": (f"Se probaron {len(candidatos_unicos)} despejes. "
                        f"Ninguno mantuvo iteraciones reales y convergentes desde x0 = {x0}."),
            "consejo": "Prueba un x0 más cercano a la raíz real.",
        }

    g_num  = sp.lambdify(x, g_elegida, "numpy")
    g_str  = str(g_elegida).replace("**","^")
    conv_s = (f"g({x}) = {g_str}  |  |g'(x₀)| ≈ {round(deriv_elegida,5)} "
              f"→ despeje estable para esta semilla.")

    resultados = []
    xi         = float(x0)

    for i in range(1, max_iter + 1):
        try:
            gxi = _safe_float(g_num(xi))
        except Exception as exc:
            return {"error": True, "titulo": "🛑 Error en g(xi)",
                    "mensaje": str(exc)[:200]}
        if gxi is None:
            return _salto_complejo(xi)

        if not math.isfinite(gxi) or abs(gxi) > _DIVERGE_THRESH:
            return {"error": True, "titulo": "🚀 Divergencia",
                    "mensaje": f"g(x{i}) = {gxi:.3e}.",
                    "consejo": "Cambia x0."}

        ea = float(abs((gxi - xi) / gxi) * 100
                   if abs(gxi) > _DIV_ZERO_THRESH else 100.0)

        resultados.append({
            "iteracion": i,
            "xi":        _fmt_num(xi),
            "gxi":       _fmt_num(gxi),
            "ea":        _fmt_num(ea) if i > 1 else "---",
        })

        xi = gxi
        if i > 1 and ea < tol:
            break

    margen  = max(abs(xi - float(x0)) * 1.3, 2.0)
    x_min_g = min(float(x0), xi) - margen
    x_max_g = max(float(x0), xi) + margen
    xs_id   = np.linspace(x_min_g, x_max_g, 300)

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, g_num, x_min_g, x_max_g, _COLOR["punto_fijo_g"], f"g({x})")
    ax.plot(xs_id, xs_id, color=_COLOR["punto_fijo_id"], lw=1.5, ls="--",
            label=f"y = {x}")
    ax.axvline(float(x0), color=_COLOR["limite"], ls=":", lw=1.2, alpha=0.7,
               label=f"x₀ = {x0}")
    ax.plot(xi, xi, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Punto fijo ≈ {round(xi, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "punto_fijo",
        "resultados": resultados,
        "raiz": _fmt_num(xi),
        "convergencia": conv_s,
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 7 — HORNER
# =============================================================================
def metodo_horner(latex_str, x0):
    """
    División sintética para evaluar P(x0) en O(n) multiplicaciones.
    """
    err = _validar_reales(x0=x0)
    if err:
        return err
    x0 = float(x0)

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    try:
        expandida       = sp.expand(f_sim)
        coefs, grado    = _extraer_coeficientes(expandida, var)
    except Exception as exc:
        return {
            "error": True,
            "titulo": "⚠️ No es un polinomio",
            "mensaje": str(exc)[:200],
            "consejo": "Horner solo acepta polinomios (ej. x³−2x+1). Sin sen, log, etc.",
        }

    if grado < 1:
        return {"error": True, "titulo": "⚠️ Polinomio constante",
                "mensaje": "El grado es 0. Ingresa un polinomio de grado ≥ 1."}

    resultados = []
    b = coefs[0]
    resultados.append({
        "grado": grado, "a": _fmt_num(coefs[0]),
        "operacion": "—", "b": _fmt_num(b),
    })

    for k in range(1, len(coefs)):
        op = b * float(x0)
        b  = coefs[k] + op
        resultados.append({
            "grado":     grado - k,
            "a":         _fmt_num(coefs[k]),
            "operacion": _fmt_num(op),
            "b":         _fmt_num(b),
        })

    residuo = b   # P(x0)
    f_num   = sp.lambdify(var, expandida, "numpy")
    margen  = 3.0

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_num, float(x0)-margen, float(x0)+margen,
                    _COLOR["horner"], f"P({var})")
    ax.axvline(float(x0), color=_COLOR["limite"], ls=":", lw=1.2, alpha=0.8,
               label=f"x₀ = {x0}")
    ax.plot(float(x0), residuo, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"P({x0}) = {round(residuo,5)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "horner",
        "resultados": resultados,
        "raiz": _fmt_num(residuo),
        "convergencia": f"P({x0}) = {round(residuo, _ROUND_DIGITS)} (evaluación directa).",
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 8 — HORNER-NEWTON (BIRGE-VIETA)
# =============================================================================
def metodo_horner_newton(latex_str, x0, tol, max_iter):
    """
    Newton-Raphson donde P(xi) y P'(xi) se calculan con doble síntesis de Horner.
    Evita derivar simbólicamente. Convergencia cuadrática.
    """
    err = _validar_reales(x0=x0)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    x0, tol, max_iter = float(x0), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    try:
        expandida    = sp.expand(f_sim)
        coefs_orig, _= _extraer_coeficientes(expandida, var)
    except Exception as exc:
        return {"error": True, "titulo": "⚠️ No es un polinomio", "mensaje": str(exc)[:200]}

    def _horner(coefs, val):
        """División sintética pura. Devuelve (residuo, coefs_cociente)."""
        b = [coefs[0]]
        for k in range(1, len(coefs)):
            b.append(coefs[k] + b[-1] * val)
        return b[-1], b[:-1]

    resultados = []
    xi         = float(x0)

    for i in range(1, max_iter + 1):
        pxi,  q  = _horner(coefs_orig, xi)  # P(xi)
        dpxi, _  = _horner(q, xi)           # P'(xi)

        if abs(pxi) < _DIV_ZERO_THRESH:
            f_num = sp.lambdify(var, expandida, "numpy")
            return _resultado_raiz_inicial(
                f_num, var, xi, _COLOR["horner_newton"], "horner_newton",
                {
                    "iteracion": 0,
                    "xi": _fmt_num(xi),
                    "pxi": _fmt_num(pxi),
                    "dpxi": "---",
                    "x_siguiente": _fmt_num(xi),
                    "ea": 0,
                },
                "Horner-Newton no requiere dividir entre P'(xi).",
            )

        if abs(dpxi) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ P'(xi) ≈ 0",
                "mensaje": f"La derivada en xi={round(xi,6)} es prácticamente cero.",
                "consejo": "Prueba un x0 diferente.",
            }

        x_sig = xi - pxi / dpxi

        if not math.isfinite(x_sig) or abs(x_sig) > _DIVERGE_THRESH:
            return {"error": True, "titulo": "🚀 Divergencia",
                    "mensaje": f"xi+1 = {x_sig:.3e}.",
                    "consejo": "Prueba un x0 más cercano a la raíz."}

        ea = float(abs((x_sig - xi) / x_sig) * 100
                   if abs(x_sig) > _DIV_ZERO_THRESH else 100.0)

        resultados.append({
            "iteracion":   i,
            "xi":          _fmt_num(xi),
            "pxi":         _fmt_num(pxi),
            "dpxi":        _fmt_num(dpxi),
            "x_siguiente": _fmt_num(x_sig),
            "ea":          _fmt_num(ea) if i > 1 else "---",
        })

        xi = x_sig
        if i > 1 and ea < tol:
            break

    f_num  = sp.lambdify(var, expandida, "numpy")
    margen = max(abs(xi - float(x0)) * 1.3, 2.0)

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_num,
                    min(float(x0), xi) - margen,
                    max(float(x0), xi) + margen,
                    _COLOR["horner_newton"], f"P({var})")
    ax.axvline(float(x0), color=_COLOR["limite"], ls=":", lw=1.2, alpha=0.7,
               label=f"x₀ = {x0}")
    ax.plot(xi, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
            label=f"Raíz ≈ {round(xi, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "horner_newton",
        "resultados": resultados,
        "raiz": _fmt_num(xi),
        "convergencia": "Cuadrática — Newton acelerado por doble síntesis de Horner.",
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 9 — MÜLLER
# =============================================================================
def metodo_muller(latex_str, x0, x1, x2, tol, max_iter):
    """
    Parábola por tres puntos. Usa cmath en todos los pasos.
    Puede encontrar raíces complejas.

    FIX: ea se castea a float() explícitamente antes de comparar con tol.
    """
    err = _validar_reales(x0=x0, x1=x1, x2=x2)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    x0, x1, x2, tol, max_iter = float(x0), float(x1), float(x2), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    try:
        if sp.Poly(f_sim, var).degree() <= 1:
            return {
                "error": True,
                "titulo": "⚠️ Denominador Müller = 0",
                "mensaje": "Los puntos pertenecen a una recta: el coeficiente cuadrático a es 0.",
                "consejo": "Para funciones lineales usa Secante o resuelve ax+b=0 directamente.",
            }
    except Exception:
        pass
          #este sirve para evalua funciones
    def _ev(val):
        try:
            return complex(f_sim.subs(var, val).evalf())
        except Exception as exc:
            raise ValueError(f"f({val}): {exc}")

    def _fc(num):
        c = complex(num)
        if abs(c.imag) < 1e-8:
            return round(c.real, 6)
        s = "+" if c.imag >= 0 else "−"
        return f"{round(c.real,4)} {s} {round(abs(c.imag),4)}i"

    try:
        valores_semilla = [_ev(x0), _ev(x1), _ev(x2)]
    except ValueError as exc:
        return {"error": True, "titulo": "🛑 Error de evaluación", "mensaje": str(exc)}

    for semilla, valor in zip([x0, x1, x2], valores_semilla):
        if abs(valor) < _DIV_ZERO_THRESH:
            f_num = sp.lambdify(var, f_sim, "numpy")
            return _resultado_raiz_inicial(
                f_num, var, semilla, _COLOR["muller"], "muller",
                {
                    "iteracion": 0,
                    "x0": _fc(x0),
                    "x1": _fc(x1),
                    "x2": _fc(x2),
                    "xr": _fc(semilla),
                    "fxr": _fc(valor),
                    "ea": 0,
                },
                f"Müller no requiere interpolar porque f({semilla}) = 0.",
            )

    if len({x0, x1, x2}) < 3:
        return {
            "error": True,
            "titulo": "⚠️ Semillas repetidas",
            "mensaje": "x0, x1 y x2 deben ser tres valores distintos.",
            "consejo": "Separa más las semillas.",
        }

    cx0, cx1, cx2 = complex(x0), complex(x1), complex(x2)
    resultados     = []
    xr             = cx2

    for i in range(1, max_iter + 1):
        try:
            f0, f1, f2 = _ev(cx0), _ev(cx1), _ev(cx2)
        except ValueError as exc:
            return {"error": True, "titulo": "🛑 Error de evaluación", "mensaje": str(exc)}

        h0, h1 = cx1 - cx0, cx2 - cx1

        if abs(h0) < _DIV_ZERO_THRESH or abs(h1) < _DIV_ZERO_THRESH:
            return {"error": True, "titulo": "⚠️ Puntos colapsados",
                    "mensaje": "Dos semillas convergieron al mismo valor.",
                    "consejo": "Usa semillas más separadas."}

        d0 = (f1 - f0) / h0
        d1 = (f2 - f1) / h1
        da = h1 + h0

        if abs(da) < _DIV_ZERO_THRESH:
            return {"error": True, "titulo": "⚠️ h0 + h1 ≈ 0",
                    "mensaje": "Semillas simétricas respecto a x2.",
                    "consejo": "Usa semillas no simétricas."}

        a = (d1 - d0) / da
        b = a * h1 + d1
        c = f2

        try:
            disc = cmath.sqrt(b**2 - 4 * a * c)
        except Exception as exc:
            return {"error": True, "titulo": "🛑 Discriminante", "mensaje": str(exc)}

        den_pos, den_neg = b + disc, b - disc
        den = den_pos if abs(den_pos) >= abs(den_neg) else den_neg

        if abs(den) < _DIV_ZERO_THRESH:
            return {"error": True, "titulo": "⚠️ Denominador Müller = 0",
                    "mensaje": "b±√D colapsó a cero.", "consejo": "Cambia las semillas."}

        xr = cx2 + (-2 * c / den)

        if not cmath.isfinite(xr) or abs(xr) > _DIVERGE_THRESH:
            return {"error": True, "titulo": "🚀 Divergencia",
                    "mensaje": f"|xr| = {abs(xr):.3e}.",
                    "consejo": "Usa semillas más cercanas a la raíz."}

        # FIX: abs() de complejo da float, pero lo casteamos explícitamente
        ea = float(abs((xr - cx2) / xr) * 100
                   if abs(xr) > _DIV_ZERO_THRESH else 100.0)

        try:
            fxr = _ev(xr)
        except ValueError as exc:
            return {"error": True, "titulo": "🛑 Error al evaluar xr", "mensaje": str(exc)}

        resultados.append({
            "iteracion": i,
            "x0":  _fc(cx0), "x1": _fc(cx1), "x2": _fc(cx2),
            "xr":  _fc(xr),  "fxr": _fc(fxr),
            "ea":  round(ea, 6) if i > 1 else "---",
        })

        if i > 1 and ea < tol:
            break

        cx0, cx1, cx2 = cx1, cx2, xr

    es_compleja = abs(xr.imag) > _COMPLEX_THRESH
    f_num       = sp.lambdify(var, f_sim, "numpy")
    margen      = 3.0

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_num, xr.real - margen, xr.real + margen,
                    _COLOR["muller"], f"f({var})")
    if es_compleja:
        ax.axvline(xr.real, color=_COLOR["raiz_compleja"], ls=":", lw=2,
                   label=f"Re(xr) = {round(xr.real,5)} (raíz ℂ)")
    else:
        ax.plot(xr.real, 0, "o", color=_COLOR["raiz"], ms=10, zorder=5,
                label=f"Raíz ≈ {round(xr.real, _ROUND_DIGITS)}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "muller",
        "resultados": resultados,
        "raiz": _fc(xr),
        "convergencia": "Orden ≈ 1.84 — Encuentra raíces reales Y complejas.",
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 10 — BAIRSTOW (DEFLACIÓN COMPLETA)
# =============================================================================
def metodo_bairstow(latex_str, r0, s0, tol, max_iter):
    """
    Extrae factores cuadráticos (x²−rx−s) iterativamente hasta encontrar
    TODAS las raíces del polinomio.

    B
      1. _extraer_coeficientes: término constante calculado UNA sola vez
      2. Deflación: cociente b[:n-1] producía grado n-2, pero si n=3 daba
         solo 2 coefs (lineal en vez de cuadrático). Ahora verificamos longitud
         correctamente antes de cada vuelta del while
      3. Threshold Im para distinguir raíz real/compleja: subido a _COMPLEX_THRESH
         para absorber errores de punto flotante en raíces que matemáticamente
         son reales pero acumularon Im pequeña
      4. Tabla: se muestran TODAS las filas (no truncadas) para transparencia
      5. Gráfica con clip dinámico (percentil) en vez de valor fijo
    """
    err = _validar_reales(r0=r0, s0=s0)
    if err:
        return err
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    r0, s0, tol, max_iter = float(r0), float(s0), float(tol), int(float(max_iter))

    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return err

    try:
        expandida       = sp.expand(f_sim)
        coefs_orig, grado = _extraer_coeficientes(expandida, var)
    except Exception as exc:
        return {"error": True, "titulo": "🛑 No es un polinomio", "mensaje": str(exc)[:200]}

    if grado < 3:
        return {
            "error": True,
            "titulo": "⚠️ Grado insuficiente",
            "mensaje": f"Grado actual: {grado}. Bairstow requiere grado ≥ 3.",
            "consejo": "Para cuadráticas usa la fórmula general.",
        }

    def _fmt_root(c):
        # FIX: threshold más tolerante para errores de punto flotante
        if abs(c.imag) < _COMPLEX_THRESH:
            return f"{round(c.real, 6)}"
        s = "+" if c.imag >= 0 else "−"
        return f"{round(c.real,6)} {s} {round(abs(c.imag),6)}i"

    def _sintetica(a_coefs, r, s):
        """Doble división sintética de Bairstow. Retorna (b, c)."""
        n = len(a_coefs) - 1
        b = [0.0] * (n + 1)
        c = [0.0] * (n + 1)

        b[0] = a_coefs[0]
        if n >= 1:
            b[1] = a_coefs[1] + r * b[0]
        for k in range(2, n + 1):
            b[k] = a_coefs[k] + r * b[k-1] + s * b[k-2]

        c[0] = b[0]
        if n >= 2:
            c[1] = b[1] + r * c[0]
        for k in range(2, n):   # Solo hasta n-1
            c[k] = b[k] + r * c[k-1] + s * c[k-2]

        return b, c

    def _extraer_factor(a_coefs, r_ini, s_ini):
        """
        Itera hasta encontrar (r*, s*) tal que (x²−r*x−s*) | P(x).7
        Retorna (r, s, coefs_cociente, todas_filas, error_msg).
        """
        n     = len(a_coefs) - 1
        r, s  = float(r_ini), float(s_ini)
        filas = []

        for it in range(1, max_iter + 1):
            b, c = _sintetica(a_coefs, r, s)

            if n < 3:
                break

            det = c[n-2]**2 - c[n-1] * c[n-3]
            if abs(det) < _DIV_ZERO_THRESH:
                return None, None, None, filas, (
                    "Determinante Jacobiano ≈ 0. Prueba otras semillas r0/s0.")

            dr = (-b[n-1] * c[n-2] + b[n]   * c[n-3]) / det
            ds = (-b[n]   * c[n-2] + b[n-1] * c[n-1]) / det

            r += dr
            s += ds

            ear = abs(dr / r) * 100 if abs(r) > _DIV_ZERO_THRESH else 100.0
            eas = abs(ds / s) * 100 if abs(s) > _DIV_ZERO_THRESH else 100.0
            ea  = max(ear, eas)

            filas.append({"r": round(r,6), "s": round(s,6), "ea": round(ea,6)})

            if ea < tol:
                break
        else:
            # El for-else de Python: entra aquí si el for agotó max_iter sin hacer 'break'
            return None, None, None, filas, f"No convergió tras {max_iter} iteraciones (ea={ea:.2f}%)."

        # Si el ciclo hizo 'break' (sí convergió), el código salta directamente aquí:
        cociente = b[:n-1]
        return r, s, cociente, filas, None

    # ── Bucle de deflación ────────────────────────────────────────────────────
    todas_raices = []
    grupos_datos = []
    coefs_act    = coefs_orig[:]
    n_factor     = 1

    while True:
        grado_act = len(coefs_act) - 1

        if grado_act < 1:
            break

        if grado_act == 1:
            # Lineal: ax+b=0  →  x = -b/a
            raiz_lin = complex(-coefs_act[1] / coefs_act[0])
            todas_raices.append(raiz_lin)
            grupos_datos.append({
                "titulo":           f"Factor {n_factor} — Lineal",
                "raices_complejas": [raiz_lin],
                "filas_raw":        [],
            })
            break

        if grado_act == 2:
            # Cuadrática residual: fórmula general directa
            a2, b2, c2 = coefs_act[0], coefs_act[1], coefs_act[2]
            D  = complex(b2**2 - 4*a2*c2)
            r1 = (-b2 + cmath.sqrt(D)) / (2*a2)
            r2 = (-b2 - cmath.sqrt(D)) / (2*a2)
            todas_raices += [r1, r2]
            grupos_datos.append({
                "titulo":           f"Factor {n_factor} — Cuadrático directo",
                "raices_complejas": [r1, r2],
                "filas_raw":        [],
            })
            break

        # Grado ≥ 3: iterar Bairstow
        r_f, s_f, cociente, filas, emsg = _extraer_factor(coefs_act, r0, s0)
        if emsg:
            return {
                "error": True,
                "titulo": "⚠️ Bairstow no convergió",
                "mensaje": emsg,
                "consejo": "Cambia r0 y s0.",
            }

        # Raíces del factor: x² − r*x − s = 0
        D_f  = complex(r_f**2 + 4*s_f)
        xq1  = (r_f + cmath.sqrt(D_f)) / 2
        xq2  = (r_f - cmath.sqrt(D_f)) / 2
        todas_raices += [xq1, xq2]

        grupos_datos.append({
            "titulo":           (f"Factor {n_factor} — "
                                 f"x² − ({round(r_f,4)})x − ({round(s_f,4)})"),
            "raices_complejas": [xq1, xq2],
            "filas_raw":        filas,
        })

        # FIX: verificar que el cociente tenga longitud correcta
        if len(cociente) < 2:
            break   # Quedó grado 0 (constante), terminar

        coefs_act = cociente
        n_factor += 1

    # ── Post-procesar: formatear raíces y preparar tabla COMPLETA ────────────
    idx_raiz = 1
    for g in grupos_datos:
        g["raices"] = []
        for rc in g["raices_complejas"]:
            g["raices"].append(f"x<sub>{idx_raiz}</sub> = {_fmt_root(rc)}")
            idx_raiz += 1

        raw   = g["filas_raw"]
        n_tot = len(raw)

        # FIX: mostramos TODAS las filas (era raw[-5:] antes, perdía info)
        g["iteraciones"] = [
            {"num": k + 1, "r": f["r"], "s": f["s"], "ea": f["ea"]}
            for k, f in enumerate(raw)
        ]
        g["n_iter"] = n_tot

    # ── Gráfica ──────────────────────────────────────────────────────────────
    raices_reales = [rc.real for rc in todas_raices if abs(rc.imag) < _COMPLEX_THRESH]
    x_min_g = (min(raices_reales) - 3) if raices_reales else -5.0
    x_max_g = (max(raices_reales) + 3) if raices_reales else  5.0
    f_num   = sp.lambdify(var, expandida, "numpy")

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_num, x_min_g, x_max_g, _COLOR["bairstow"], f"P({var})")

    for idx, rc in enumerate(todas_raices):
        col = _COLORES_RAICES[idx % len(_COLORES_RAICES)]
        lbl = f"x{idx+1} = {_fmt_root(rc)}"
        if abs(rc.imag) < _COMPLEX_THRESH:
            ax.plot(rc.real, 0, "o", color=col, ms=9, zorder=5, label=lbl)
        else:
            ax.axvline(rc.real, color=col, ls=":", alpha=0.7,
                       label=f"{lbl} (ℂ)")

    ax.legend(fontsize=7, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False, "tipo": "bairstow",
        "grupos":      grupos_datos,
        "resultados":  [],
        "raiz":        "",
        "convergencia": (f"Se encontraron {len(todas_raices)} raíces en "
                         f"{n_factor - 1} extracción(es) cuadrática(s)."),
        "grafica": _grafica_b64(fig),
    }


# =============================================================================
# MÉTODO 11 — ELIMINACIÓN DE GAUSS (SEC LINEALES)
# =============================================================================
def metodo_gauss(matriz_texto):
    A, b, err = _parse_matriz_aumentada(matriz_texto)
    if err:
        return err

    n = len(b)
    M = np.column_stack((A.copy(), b.copy())).astype(float)
    pasos = [{"titulo": "Matriz aumentada inicial", "matriz": _fmt_matrix(M)}]

    for k in range(n - 1):
        pivote_fila = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[pivote_fila, k]) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Sistema singular",
                "mensaje": f"No hay pivote válido en la columna {k + 1}.",
                "consejo": "Revisa si las ecuaciones son dependientes o incompatibles.",
            }
        if pivote_fila != k:
            M[[k, pivote_fila]] = M[[pivote_fila, k]]
            pasos.append({
                "titulo": f"Intercambio F{k+1} ↔ F{pivote_fila+1}",
                "matriz": _fmt_matrix(M),
            })

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]
            M[i, k] = 0.0
            pasos.append({
                "titulo": f"F{i+1} ← F{i+1} − ({_fmt_num(factor)})F{k+1}",
                "matriz": _fmt_matrix(M),
            })

    if abs(M[-1, -2]) < _DIV_ZERO_THRESH:
        return {
            "error": True,
            "titulo": "⚠️ Sistema singular",
            "mensaje": "El último pivote es cero; no hay solución única.",
            "consejo": "Usa otro método de análisis o revisa el sistema.",
        }

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        pivote = M[i, i]
        if abs(pivote) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Sustitución imposible",
                "mensaje": f"El pivote de la fila {i + 1} es cero.",
            }
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / pivote

    return _resultado_sistema_lineal(
        "Eliminación de Gauss",
        x, A, b, pasos, matriz_final=M,
    )


# =============================================================================
# MÉTODO 12 — GAUSS-JORDAN (SEC LINEALES)
# =============================================================================
def metodo_gauss_jordan(matriz_texto):
    A, b, err = _parse_matriz_aumentada(matriz_texto)
    if err:
        return err

    n = len(b)
    M = np.column_stack((A.copy(), b.copy())).astype(float)
    pasos = [{"titulo": "Matriz aumentada inicial", "matriz": _fmt_matrix(M)}]

    for k in range(n):
        pivote_fila = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[pivote_fila, k]) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Sistema singular",
                "mensaje": f"No hay pivote válido en la columna {k + 1}.",
                "consejo": "Gauss-Jordan requiere una solución única.",
            }
        if pivote_fila != k:
            M[[k, pivote_fila]] = M[[pivote_fila, k]]
            pasos.append({
                "titulo": f"Intercambio F{k+1} ↔ F{pivote_fila+1}",
                "matriz": _fmt_matrix(M),
            })

        pivote = M[k, k]
        M[k, :] = M[k, :] / pivote
        pasos.append({
            "titulo": f"F{k+1} ← F{k+1} / ({_fmt_num(pivote)})",
            "matriz": _fmt_matrix(M),
        })

        for i in range(n):
            if i == k:
                continue
            factor = M[i, k]
            if abs(factor) < _DIV_ZERO_THRESH:
                continue
            M[i, :] -= factor * M[k, :]
            M[i, k] = 0.0
            pasos.append({
                "titulo": f"F{i+1} ← F{i+1} − ({_fmt_num(factor)})F{k+1}",
                "matriz": _fmt_matrix(M),
            })

    x = M[:, -1].copy()
    return _resultado_sistema_lineal(
        "Gauss-Jordan",
        x, A, b, pasos, matriz_final=M,
    )


# =============================================================================
# MÉTODO 13 — FACTORIZACIÓN LU (PA = LU)
# =============================================================================
def metodo_lu(matriz_texto):
    A, b, err = _parse_matriz_aumentada(matriz_texto)
    if err:
        return err

    n = len(b)
    U = A.copy().astype(float)
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)
    pasos = [{"titulo": "Matrices iniciales", "L": _fmt_matrix(L),
              "U": _fmt_matrix(U), "P": _fmt_matrix(P)}]

    for k in range(n - 1):
        pivote_fila = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivote_fila, k]) < _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Matriz singular",
                "mensaje": f"No hay pivote válido en la columna {k + 1}.",
                "consejo": "LU requiere una matriz de coeficientes invertible.",
            }
        if pivote_fila != k:
            U[[k, pivote_fila]] = U[[pivote_fila, k]]
            P[[k, pivote_fila]] = P[[pivote_fila, k]]
            if k > 0:
                L[[k, pivote_fila], :k] = L[[pivote_fila, k], :k]
            pasos.append({
                "titulo": f"Pivoteo: F{k+1} ↔ F{pivote_fila+1}",
                "L": _fmt_matrix(L), "U": _fmt_matrix(U), "P": _fmt_matrix(P),
            })

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
            U[i, k] = 0.0
            pasos.append({
                "titulo": f"l{i+1}{k+1} = {_fmt_num(factor)}",
                "L": _fmt_matrix(L), "U": _fmt_matrix(U), "P": _fmt_matrix(P),
            })

    if abs(U[-1, -1]) < _DIV_ZERO_THRESH:
        return {
            "error": True,
            "titulo": "⚠️ Matriz singular",
            "mensaje": "U tiene un pivote final cero.",
        }

    pb = P @ b
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = pb[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < _DIV_ZERO_THRESH:
            return {"error": True, "titulo": "⚠️ Sustitución LU imposible",
                    "mensaje": f"U[{i+1},{i+1}] = 0."}
        x[i] = (y[i] - np.dot(U[i, i+1:n], x[i+1:n])) / U[i, i]

    return _resultado_sistema_lineal(
        "LU con pivoteo parcial",
        x, A, b, pasos,
        extras={
            "P": _fmt_matrix(P),
            "L": _fmt_matrix(L),
            "U": _fmt_matrix(U),
            "y": _fmt_vec(y),
            "pb": _fmt_vec(pb),
        },
    )


# =============================================================================
# MÉTODO 14 — NEWTON-RAPHSON PARA SISTEMAS NO LINEALES
# =============================================================================
def metodo_newton_sistemas(funciones_texto, variables_texto, inicial_texto, tol, max_iter):
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    tol, max_iter = float(tol), int(float(max_iter))

    variables, err = _parse_variables_sistema(variables_texto)
    if err:
        return err
    exprs, err = _parse_funciones_sistema(funciones_texto, variables)
    if err:
        return err
    x_actual, err = _parse_vector_inicial(inicial_texto, len(variables))
    if err:
        return err

    F_sim = sp.Matrix(exprs)
    J_sim = F_sim.jacobian(variables)
    try:
        F_num = sp.lambdify(variables, F_sim, "numpy")
        J_num = sp.lambdify(variables, J_sim, "numpy")
    except Exception as exc:
        return {
            "error": True,
            "titulo": "🛑 No se pudo preparar el sistema",
            "mensaje": str(exc)[:200],
        }

    resultados = []
    puntos = [x_actual.copy()]

    for i in range(1, max_iter + 1):
        try:
            with np.errstate(all="ignore"):
                F_raw = np.array(F_num(*x_actual), dtype=complex).reshape(-1)
                J_raw = np.array(J_num(*x_actual), dtype=complex)
        except Exception as exc:
            return {
                "error": True,
                "titulo": "🛑 Error de evaluación",
                "mensaje": f"F/J falló en la iteración {i}: {str(exc)[:160]}",
                "consejo": "Revisa el dominio de las funciones y la semilla inicial.",
            }

        if np.max(np.abs(F_raw.imag)) > 1e-8 or np.max(np.abs(J_raw.imag)) > 1e-8:
            return {
                "error": True,
                "titulo": "🛑 Salto al plano complejo",
                "mensaje": f"El sistema produjo valores complejos en la iteración {i}.",
                "consejo": "Cambia la semilla o revisa raíces/logaritmos en las funciones.",
            }

        F_val = F_raw.real.astype(float)
        J_val = J_raw.real.astype(float)
        if (not np.all(np.isfinite(F_val))) or (not np.all(np.isfinite(J_val))):
            return {
                "error": True,
                "titulo": "🛑 Dominio matemático",
                "mensaje": f"F o J tiene NaN/∞ en la iteración {i}.",
                "consejo": "La semilla está fuera del dominio o cerca de una singularidad.",
            }

        norma_f = float(np.linalg.norm(F_val, ord=np.inf))
        if norma_f < tol:
            resultados.append({
                "iteracion": i,
                "x": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_actual)),
                "F": _fmt_vec(F_val),
                "delta": _fmt_vec(np.zeros_like(x_actual)),
                "x_siguiente": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_actual)),
                "norma": _fmt_num(norma_f),
                "ea": 0,
            })
            break

        try:
            delta = np.linalg.solve(J_val, -F_val)
        except np.linalg.LinAlgError:
            return {
                "error": True,
                "titulo": "⚠️ Jacobiano singular",
                "mensaje": f"det(J) ≈ 0 en la iteración {i}; no se puede calcular Δx.",
                "consejo": "Prueba otra semilla inicial.",
            }

        x_siguiente = x_actual + delta
        if (not np.all(np.isfinite(x_siguiente))) or np.linalg.norm(x_siguiente, ord=np.inf) > _DIVERGE_THRESH:
            return {
                "error": True,
                "titulo": "🚀 Divergencia",
                "mensaje": f"La iteración {i} salió del rango seguro.",
                "consejo": "Usa una semilla más cercana a la solución.",
            }

        ea = float(np.linalg.norm(delta, ord=np.inf) /
                   max(np.linalg.norm(x_siguiente, ord=np.inf), _DIV_ZERO_THRESH) * 100)

        resultados.append({
            "iteracion": i,
            "x": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_actual)),
            "F": _fmt_vec(F_val),
            "delta": _fmt_vec(delta),
            "x_siguiente": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_siguiente)),
            "norma": _fmt_num(norma_f),
            "ea": _fmt_num(ea),
        })

        x_actual = x_siguiente
        puntos.append(x_actual.copy())
        if norma_f < tol or ea < tol:
            break

    grafica = _grafica_newton_sistemas(exprs, variables, puntos)
    return {
        "error": False,
        "tipo": "newton_sistemas",
        "variables": [str(v) for v in variables],
        "funciones": [str(e).replace("**", "^").replace("*", "·") for e in exprs],
        "jacobiano": [[str(J_sim[i, j]).replace("**", "^").replace("*", "·")
                       for j in range(J_sim.shape[1])]
                      for i in range(J_sim.shape[0])],
        "solucion": _fmt_vec(x_actual),
        "raiz": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_actual)),
        "convergencia": (
            f"Newton para sistemas finalizó con ||F(x)||∞ = "
            f"{_fmt_num(float(np.linalg.norm(np.array(F_num(*x_actual), dtype=float).reshape(-1), ord=np.inf)))}."
        ),
        "resultados": resultados,
        "grafica": grafica,
    }


# =============================================================================
# RUTAS FLASK
# =============================================================================
def _ruta(metodo_fn, template, **kwargs):
    """Helper genérico para rutas POST/GET."""
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_fn(**kwargs())
        except Exception as exc:
            datos = {"error": True,
                     "titulo": "🛑 Error inesperado en el servidor",
                     "mensaje": str(exc)[:300]}
    return render_template(template, datos=datos)


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/biseccion", methods=["GET","POST"])
def biseccion():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_biseccion(
                request.form["ecuacion_latex"],
                float(request.form["xl"]),
                float(request.form["xu"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("biseccion.html", datos=datos)


@app.route("/falsa_posicion", methods=["GET","POST"])
def falsa_posicion():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_falsa_posicion(
                request.form["ecuacion_latex"],
                float(request.form["xl"]),
                float(request.form["xu"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("falsa_posicion.html", datos=datos)


@app.route("/newton", methods=["GET","POST"])
def newton():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_newton_raphson(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("newton.html", datos=datos)


@app.route("/secante", methods=["GET","POST"])
def secante():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_secante(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
                float(request.form["x1"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("secante.html", datos=datos)


@app.route("/taylor", methods=["GET","POST"])
def taylor():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_taylor(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
                float(request.form["x_eval"]),
                int(request.form["n_terminos"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("taylor.html", datos=datos)


@app.route("/punto_fijo", methods=["GET","POST"])
def punto_fijo():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_punto_fijo(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("punto_fijo.html", datos=datos)


@app.route("/horner", methods=["GET","POST"])
def horner():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_horner(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("horner.html", datos=datos)


@app.route("/horner_newton", methods=["GET","POST"])
def horner_newton():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_horner_newton(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("horner_newton.html", datos=datos)


@app.route("/muller", methods=["GET","POST"])
def muller():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_muller(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
                float(request.form["x1"]),
                float(request.form["x2"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("muller.html", datos=datos)


@app.route("/bairstow", methods=["GET","POST"])
def bairstow():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_bairstow(
                request.form["ecuacion_latex"],
                float(request.form["r0"]),
                float(request.form["s0"]),
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("bairstow.html", datos=datos)


@app.route("/gauss", methods=["GET","POST"])
def gauss():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_gauss(request.form["matriz"])
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("gauss.html", datos=datos)


@app.route("/gauss_jordan", methods=["GET","POST"])
def gauss_jordan():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_gauss_jordan(request.form["matriz"])
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("gauss_jordan.html", datos=datos)


@app.route("/lu", methods=["GET","POST"])
def lu():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_lu(request.form["matriz"])
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("lu.html", datos=datos)


@app.route("/newton_sistemas", methods=["GET","POST"])
def newton_sistemas():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_newton_sistemas(
                request.form["funciones"],
                request.form["variables"],
                request.form["inicial"],
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("newton_sistemas.html", datos=datos)


if __name__ == "__main__":
    app.run(debug=True)
