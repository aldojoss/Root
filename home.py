

from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, cmath, math, itertools
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import analisis_inteligente
from teoria_metodos import obtener_familias_teoria, obtener_teoria_metodos

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES DE SEGURIDAD
# ─────────────────────────────────────────────────────────────────────────────
_DIV_ZERO_THRESH = 1e-12    # Denominador mínimo tolerable
_DIVERGE_THRESH  = 1e15     # FIX: era 1e10, demasiado estricto para algunas funciones
_ROUND_DIGITS    = 8
_COMPLEX_THRESH  = 1e-4     # FIX: era 1e-6, ahora más tolerante con errores de float
_MAX_ITER        = 1000     # Limite operativo para evitar respuestas enormes o cuelgues
_PARSER_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────────────────────────────────────
_BG     = "#0d1117"
_BG2    = "#181b20"
_GRID   = "#2a313d"
_AXIS   = "#5e6a7d"     
_TICK   = "#a0acc0"
_TEXT   = "#f4f7fb"
_LEGEND = "#12161d"

_COLOR = {
    "biseccion":     "#6ea8ff",
    "falsa":         "#2ee59d",
    "newton":        "#ff9b6a",
    "secante":       "#c792ff",
    "taylor_f":      "#6ea8ff",
    "taylor_p":      "#f8d66d",
    "punto_fijo_g":  "#c792ff",
    "punto_fijo_id": "#5e6a7d",
    "horner":        "#f8d66d",
    "horner_newton": "#2ee59d",
    "muller":        "#ff9b6a",
    "bairstow":      "#c792ff",
    "jacobi":        "#4ddde0",
    "gauss_seidel":  "#f8d66d",
    "newton_diff":   "#4ddde0",
    "lagrange":      "#f8d66d",
    "spline":        "#6ea8ff",
    "regresion":     "#ff6f91",
    "trapecio":      "#2ee59d",
    "romberg":       "#6ea8ff",
    "diferenciacion":"#c792ff",
    "raiz":          "#2ee59d",
    "raiz_compleja": "#ff9b6a",
    "limite":        "#f8d66d",
}

_COLORES_RAICES = [
    "#2ee59d", "#f8d66d", "#ff9b6a", "#6ea8ff",
    "#c792ff", "#4ddde0", "#ff6f91", "#b7f7d7",
]


# =============================================================================
# funciones compartidas para los metodos que se reciclaran
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

        expr = _normalizar_constantes_expr(expr)

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


def _normalizar_constantes_expr(expr):
    """Convierte símbolos comunes escritos por MathLive a constantes de SymPy."""
    reemplazos = {
        sp.Symbol("e"): sp.E,
        sp.Symbol("E"): sp.E,
        sp.Symbol("pi"): sp.pi,
        sp.Symbol("Pi"): sp.pi,
        sp.Symbol("π"): sp.pi,
    }
    return expr.subs({simbolo: valor for simbolo, valor in reemplazos.items()
                      if simbolo in expr.free_symbols})


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

        
        if isinstance(raw, (int, float, complex)):
            raw = np.full(n, complex(raw).real)

        
        ys = np.array([
            float(complex(v).real) if (not isinstance(v, float) or math.isfinite(v))
            and abs(complex(v).imag) < abs(complex(v).real) * 0.01 + 1
            else np.nan
            for v in raw
        ], dtype=float)

        
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
# HELPERS para las inter y extrapolaciones, regresión y splines
# =============================================================================
def _fmt_expr(expr, digits=_ROUND_DIGITS):
    try:
        expr = sp.expand(expr)
        return (str(sp.N(expr, digits))
                .replace("**", "^")
                .replace("*", "·")
                .replace("sqrt", "√"))
    except Exception:
        return str(expr).replace("**", "^").replace("*", "·")


def _latex_expr(expr, digits=_ROUND_DIGITS, expand=True):
    try:
        expr = sp.expand(expr) if expand else sp.simplify(expr)
        return sp.latex(sp.N(expr, digits))
    except Exception:
        try:
            return sp.latex(expr)
        except Exception:
            return str(expr)


def _math_inline(expr_latex):
    return f"\\({expr_latex}\\)"


def _parse_x_eval(x_eval, nombre="x a evaluar"):
    if x_eval is None or str(x_eval).strip() == "":
        return None, None
    try:
        valor = _parse_numero_matriz(x_eval)
    except Exception:
        return None, _error_parametro(
            f"{nombre} debe ser un número real.",
            "Puedes dejarlo vacío si solo quieres construir el modelo.",
        )
    return valor, None


def _parse_puntos(texto, min_n=2, max_n=50, ordenar=True):
    if not texto or not str(texto).strip():
        return None, _error_parametro(
            "La tabla de puntos está vacía.",
            "Ingresa al menos dos pares x y, uno por fila.",
        )

    puntos = []
    for raw in str(texto).replace(";", "\n").splitlines():
        linea = raw.strip()
        if not linea:
            continue
        limpia = (linea.strip("()[]{}")
                  .replace("|", " ")
                  .replace(",", " "))
        partes = limpia.split()
        if len(partes) != 2:
            return None, _error_parametro(
                f"No se pudo leer el punto '{linea}'.",
                "Usa exactamente dos valores por fila: x y.",
            )
        try:
            x_val = _parse_numero_matriz(partes[0])
            y_val = _parse_numero_matriz(partes[1])
        except Exception as exc:
            return None, _error_parametro(
                f"No se pudo leer el punto '{linea}': {str(exc)[:120]}",
                "Usa números reales, fracciones o decimales con punto.",
            )
        puntos.append((x_val, y_val))

    if len(puntos) < min_n:
        return None, _error_parametro(
            f"Se requieren al menos {min_n} puntos.",
            "Agrega más pares (x, y) antes de calcular.",
        )
    if len(puntos) > max_n:
        return None, _error_parametro(
            f"El máximo permitido es {max_n} puntos.",
            "Reduce la tabla para mantener la respuesta manejable.",
        )

    puntos_ordenados = sorted(puntos, key=lambda p: p[0]) if ordenar else puntos
    for i in range(1, len(puntos_ordenados)):
        if abs(puntos_ordenados[i][0] - puntos_ordenados[i - 1][0]) <= _DIV_ZERO_THRESH:
            return None, _error_parametro(
                "Los valores de x deben ser únicos.",
                "No puede haber dos puntos con la misma abscisa.",
            )
    return puntos_ordenados, None


def _puntos_para_tabla(puntos):
    return [
        {"i": i, "x": _fmt_num(x_val), "y": _fmt_num(y_val)}
        for i, (x_val, y_val) in enumerate(puntos)
    ]


def _rango_desde_x(xs, margen_rel=0.16):
    x_min, x_max = min(xs), max(xs)
    ancho = max(abs(x_max - x_min), 1.0)
    margen = ancho * margen_rel
    return x_min - margen, x_max + margen


def _evaluar_polinomio(expr, var, x_eval):
    val = _safe_float(sp.N(expr.subs(var, x_eval)))
    if val is None or not math.isfinite(val):
        return None, {
            "error": True,
            "titulo": "🛑 Evaluación inválida",
            "mensaje": f"El modelo produjo un valor no real en x = {_fmt_num(x_eval)}.",
            "consejo": "Revisa los puntos o usa otro valor de evaluación.",
        }
    return val, None


def _grafica_interpolacion(puntos, expr, var, color, etiqueta, x_eval=None, y_eval=None):
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    x_min, x_max = _rango_desde_x(xs)
    f_num = sp.lambdify(var, expr, "numpy")

    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_num, x_min, x_max, color, etiqueta)
    ax.scatter(xs, ys, s=55, color=_COLOR["raiz"], edgecolor=_BG,
               linewidth=1.0, zorder=5, label="Puntos")
    if x_eval is not None and y_eval is not None:
        ax.plot(x_eval, y_eval, "D", color=_COLOR["limite"], ms=8,
                zorder=6, label=f"P({_fmt_num(x_eval)}) = {_fmt_num(y_eval)}")
        ax.axvline(x_eval, color=_COLOR["limite"], ls=":", lw=1.1, alpha=0.75)
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()
    return _grafica_b64(fig)


# =============================================================================
# HELPERS para sistemas de ecuaciones
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


def _latex_num(num):
    return str(_fmt_num(num)).replace("−", "-")


def _latex_vector_numeric(vec):
    valores = [_latex_num(v) for v in np.asarray(vec, dtype=float).reshape(-1)]
    return r"\begin{bmatrix}" + r"\\ ".join(valores) + r"\end{bmatrix}"


def _latex_matrix_numeric(mat):
    filas = []
    for fila in np.asarray(mat, dtype=float):
        filas.append(" & ".join(_latex_num(v) for v in fila))
    return r"\begin{bmatrix}" + r"\\ ".join(filas) + r"\end{bmatrix}"


def _latex_vector_simbolico(n, nombre="y"):
    return r"\begin{bmatrix}" + r"\\ ".join(f"{nombre}_{i + 1}" for i in range(n)) + r"\end{bmatrix}"


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


def _matriz_iteracion_lineal(A, b, metodo):
    D = np.diag(np.diag(A))
    if np.any(np.abs(np.diag(A)) < _DIV_ZERO_THRESH):
        raise ValueError("diagonal_cero")

    if metodo == "jacobi":
        R = A - D
        B = -np.linalg.solve(D, R)
        c = np.linalg.solve(D, b)
    elif metodo == "gauss_seidel":
        DL = np.tril(A)
        U = np.triu(A, 1)
        B = -np.linalg.solve(DL, U)
        c = np.linalg.solve(DL, b)
    else:
        raise ValueError("método iterativo desconocido")

    if (not np.all(np.isfinite(B))) or (not np.all(np.isfinite(c))):
        raise ValueError("matriz_iteracion_no_finita")
    return B, c


def _radio_espectral(B):
    vals = np.linalg.eigvals(B)
    rho = float(np.max(np.abs(vals))) if len(vals) else 0.0
    if not math.isfinite(rho):
        raise ValueError("radio_espectral_no_finito")
    return rho


def _autovalores_iteracion(B):
    vals = np.linalg.eigvals(B)
    return [
        {"valor": _fmt_num(v), "modulo": _fmt_num(abs(v))}
        for v in vals
    ]


def _diagnostico_dominancia(A):
    """Analiza dominancia diagonal fila por fila para explicar convergencia."""
    A = np.asarray(A, dtype=float)
    filas = []
    ok_count = 0
    strict_count = 0
    for i, fila in enumerate(A):
        diagonal = abs(float(fila[i]))
        resto = float(np.sum(np.abs(fila)) - diagonal)
        margen = diagonal - resto
        if diagonal > resto + _DIV_ZERO_THRESH:
            estado = "Estricta"
            ok_count += 1
            strict_count += 1
        elif diagonal + _DIV_ZERO_THRESH >= resto:
            estado = "Débil"
            ok_count += 1
        else:
            estado = "No dominante"
        filas.append({
            "fila": i + 1,
            "diagonal": _fmt_num(diagonal),
            "resto": _fmt_num(resto),
            "margen": _fmt_num(margen),
            "estado": estado,
            "ok": diagonal + _DIV_ZERO_THRESH >= resto,
            "estricta": diagonal > resto + _DIV_ZERO_THRESH,
            "margen_raw": margen,
        })

    dominante = ok_count == len(filas) and strict_count > 0
    debil = ok_count == len(filas)
    if dominante:
        resumen = "Diagonal dominante estricta al menos en una fila."
    elif debil:
        resumen = "Diagonal dominante débil."
    else:
        resumen = f"{ok_count}/{len(filas)} filas cumplen dominancia diagonal."
    return {
        "filas": filas,
        "dominante": dominante,
        "debil": debil,
        "ok_count": ok_count,
        "strict_count": strict_count,
        "resumen": resumen,
    }


def _perm_texto(perm):
    return " → ".join(f"F{p + 1}" for p in perm)


def _fila_score_para_columna(A, fila_idx, col_idx):
    fila = A[fila_idx]
    diagonal = abs(float(fila[col_idx]))
    resto = float(np.sum(np.abs(fila)) - diagonal)
    escala = max(float(np.sum(np.abs(fila))), _DIV_ZERO_THRESH)
    dominante_bonus = 2.0 if diagonal + _DIV_ZERO_THRESH >= resto else 0.0
    estricta_bonus = 1.0 if diagonal > resto + _DIV_ZERO_THRESH else 0.0
    return dominante_bonus + estricta_bonus + (diagonal - resto) / escala + diagonal / escala


def _buscar_perm_dominante(A):
    """Busca por backtracking una permutación de filas con dominancia diagonal."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    candidatos = []
    for col in range(n):
        rows = []
        for row in range(n):
            diag = abs(float(A[row, col]))
            resto = float(np.sum(np.abs(A[row])) - diag)
            if diag + _DIV_ZERO_THRESH >= resto:
                rows.append(row)
        rows.sort(key=lambda r: _fila_score_para_columna(A, r, col), reverse=True)
        candidatos.append((col, rows))

    if any(not rows for _, rows in candidatos):
        return None

    orden_columnas = sorted(candidatos, key=lambda item: len(item[1]))
    asignacion = [None] * n
    usadas = set()

    def backtrack(pos):
        if pos == len(orden_columnas):
            diag = _diagnostico_dominancia(A[asignacion, :])
            return diag["debil"]
        col, rows = orden_columnas[pos]
        for row in rows:
            if row in usadas:
                continue
            asignacion[col] = row
            usadas.add(row)
            if backtrack(pos + 1):
                return True
            usadas.remove(row)
            asignacion[col] = None
        return False

    if backtrack(0):
        return tuple(asignacion)
    return None


def _permutaciones_beam(A, ancho=384):
    """Genera buenas permutaciones candidatas sin factorial completo para sistemas grandes."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    beam = [(0.0, tuple(), frozenset())]
    for col in range(n):
        siguiente = []
        for score, perm, usadas in beam:
            for row in range(n):
                if row in usadas:
                    continue
                nuevo_score = score + _fila_score_para_columna(A, row, col)
                siguiente.append((nuevo_score, perm + (row,), usadas | {row}))
        siguiente.sort(key=lambda item: item[0], reverse=True)
        beam = siguiente[:ancho]
    return [perm for _, perm, _ in beam]


def _preparar_iterativo(A, b, metodo):
    n = len(b)

    def _candidato(perm):
        perm = tuple(perm)
        PA = A[list(perm), :].astype(float)
        Pb = b[list(perm)].astype(float)
        B, c = _matriz_iteracion_lineal(PA, Pb, metodo)
        return {
            "A": PA,
            "b": Pb,
            "B": B,
            "c": c,
            "rho": _radio_espectral(B),
            "perm": perm,
            "dominancia": _diagnostico_dominancia(PA),
            "autovalores": _autovalores_iteracion(B),
        }

    original = None
    candidatos_validos = []
    try:
        original = _candidato(range(n))
        candidatos_validos.append(original)
    except Exception:
        pass

    perm_dominante = _buscar_perm_dominante(A)
    if perm_dominante is not None:
        try:
            candidatos_validos.append(_candidato(perm_dominante))
        except Exception:
            pass

    perms_revisadas = {cand["perm"] for cand in candidatos_validos}
    if n <= 7:
        for perm in itertools.permutations(range(n)):
            perm = tuple(perm)
            if perm in perms_revisadas:
                continue
            try:
                cand = _candidato(perm)
            except Exception:
                continue
            candidatos_validos.append(cand)
            perms_revisadas.add(perm)
    else:
        for perm in _permutaciones_beam(A):
            if perm in perms_revisadas:
                continue
            try:
                cand = _candidato(perm)
            except Exception:
                continue
            candidatos_validos.append(cand)
            perms_revisadas.add(perm)

    if not candidatos_validos and np.any(np.abs(np.diag(A)) < _DIV_ZERO_THRESH):
        return None, None, None, None, None, False, {
            "error": True,
            "titulo": "⚠️ Diagonal inválida",
            "mensaje": "Jacobi y Gauss-Seidel requieren coeficientes diagonales no nulos.",
            "consejo": "Reordena las ecuaciones o usa un método directo como Gauss o LU.",
        }

    if not candidatos_validos:
        return None, None, None, None, None, False, {
            "error": True,
            "titulo": "⚠️ Sistema iterativo inválido",
            "mensaje": "No se pudo construir una matriz de iteración válida con ningún orden de filas probado.",
            "consejo": "Revisa pivotes diagonales, dependencias entre ecuaciones y escala del sistema.",
        }

    mejor = min(candidatos_validos, key=lambda cand: cand["rho"])
    identidad = tuple(range(n))
    reordenado = mejor["perm"] != identidad
    rho_original = original["rho"] if original is not None else None
    analisis = {
        "reordenado": reordenado,
        "perm": mejor["perm"],
        "perm_texto": _perm_texto(mejor["perm"]),
        "perm_original": identidad,
        "perm_original_texto": _perm_texto(identidad),
        "rho_original": rho_original,
        "rho_final": mejor["rho"],
        "dominancia_original": _diagnostico_dominancia(A),
        "dominancia_final": mejor["dominancia"],
        "autovalores": mejor["autovalores"],
        "candidatos_revisados": len(perms_revisadas),
        "matriz_original": _fmt_matrix(A),
        "matriz_reordenada": _fmt_matrix(mejor["A"]),
        "b_original": _fmt_vec(b),
        "b_reordenado": _fmt_vec(mejor["b"]),
        "diagnostico": (
            "Se reordenaron las ecuaciones para obtener el menor radio espectral encontrado."
            if reordenado else
            "El orden ingresado ya fue el mejor entre los candidatos revisados."
        ),
    }
    if rho_original is not None and reordenado:
        analisis["mejora_rho"] = rho_original - mejor["rho"]
    else:
        analisis["mejora_rho"] = 0.0

    return mejor["A"], mejor["b"], mejor["B"], mejor["c"], mejor["rho"], analisis, None


def _grafica_convergencia_sistema(resultados, color, label):
    fig, ax = _hacer_figura()
    xs = [fila["iteracion"] for fila in resultados]
    errores = [float(fila["ea_raw"]) for fila in resultados]
    residuos = [float(fila["residuo_raw"]) for fila in resultados]
    ax.plot(xs, errores, "o-", color=color, linewidth=2.0, ms=4,
            label="Error relativo (%)")
    ax.plot(xs, residuos, "s--", color=_COLOR["raiz"], linewidth=1.8, ms=3,
            label="Residuo ||Ax-b||∞")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Magnitud")
    ax.set_title(f"Convergencia de {label}")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()
    return _grafica_b64(fig)


def _extras_iterativo(B, c, rho, tol, analisis, iteraciones=0, error_final=None, residuo_final=None):
    analisis = analisis or {}
    criterio = "Convergencia esperada: ρ(B) < 1" if rho < 1.0 else "Convergencia no garantizada: ρ(B) ≥ 1"
    extras = {
        "radio_espectral": _fmt_num(rho),
        "radio_espectral_raw": rho,
        "criterio_rho": criterio,
        "matriz_iteracion": _fmt_matrix(B),
        "vector_c": _fmt_vec(c),
        "tolerancia": _fmt_num(tol),
        "reordenado": bool(analisis.get("reordenado", False)),
        "perm_texto": analisis.get("perm_texto", "F1"),
        "perm_original_texto": analisis.get("perm_original_texto", "F1"),
        "rho_original": "---" if analisis.get("rho_original") is None else _fmt_num(analisis.get("rho_original")),
        "rho_final": _fmt_num(rho),
        "mejora_rho": _fmt_num(analisis.get("mejora_rho", 0.0)),
        "dominancia_original": analisis.get("dominancia_original"),
        "dominancia_final": analisis.get("dominancia_final"),
        "autovalores": analisis.get("autovalores", []),
        "candidatos_revisados": analisis.get("candidatos_revisados", 0),
        "matriz_original": analisis.get("matriz_original"),
        "matriz_reordenada": analisis.get("matriz_reordenada"),
        "b_original": analisis.get("b_original"),
        "b_reordenado": analisis.get("b_reordenado"),
        "diagnostico": analisis.get("diagnostico", ""),
        "iteraciones": iteraciones,
        "error_final": "---" if error_final is None else _fmt_num(error_final),
        "residuo_final": "---" if residuo_final is None else _fmt_num(residuo_final),
    }
    return extras


def _resultado_sistema_iterativo(metodo, A_original, b_original, A, b, B, c, rho,
                                 inicial, tol, max_iter, analisis, color):
    residuo_inicial = _residuo_lineal(A_original, inicial, b_original)
    if residuo_inicial <= _DIV_ZERO_THRESH:
        resultados = [{
            "iteracion": 0,
            "x_previo": "---",
            "x_siguiente": ", ".join(f"x{i+1}={_fmt_num(v)}" for i, v in enumerate(inicial)),
            "ea": 0,
            "ea_raw": 0.0,
            "delta": 0,
            "residuo": _fmt_num(residuo_inicial),
            "residuo_raw": residuo_inicial,
        }]
        grafica = _grafica_convergencia_sistema(resultados, color, metodo)
        extras = _extras_iterativo(B, c, rho, tol, analisis, 0, 0.0, residuo_inicial)
        return {
            "error": False,
            "tipo": "sistema_iterativo",
            "metodo": metodo,
            "solucion": _fmt_vec(inicial),
            "raiz": ", ".join(f"x{i+1}={_fmt_num(v)}" for i, v in enumerate(inicial)),
            "convergencia": "La semilla inicial ya satisface el sistema.",
            "resultados": resultados,
            "pasos": [],
            "grafica": grafica,
            "extras": extras,
        }

    if rho >= 1.0:
        return {
            "error": True,
            "titulo": "⚠️ Convergencia no garantizada",
            "mensaje": (
                f"El radio espectral de la matriz de iteración es {_fmt_num(rho)} ≥ 1; "
                f"{metodo} puede divergir para este sistema."
            ),
            "consejo": "Reordena o escala las ecuaciones, usa una semilla distinta, o resuelve con Gauss/LU.",
            "extras": _extras_iterativo(B, c, rho, tol, analisis),
        }

    x_prev = np.asarray(inicial, dtype=float)
    resultados = []
    for i in range(1, max_iter + 1):
        try:
            x_next = B @ x_prev + c
        except Exception as exc:
            return {
                "error": True,
                "titulo": "🛑 Iteración fallida",
                "mensaje": f"No se pudo calcular la iteración {i}: {str(exc)[:160]}",
                "consejo": "Revisa la matriz y el vector inicial.",
            }

        if (not np.all(np.isfinite(x_next))) or np.linalg.norm(x_next, ord=np.inf) > _DIVERGE_THRESH:
            return {
                "error": True,
                "titulo": "🚀 Divergencia",
                "mensaje": f"La iteración {i} salió del rango seguro.",
                "consejo": "Usa otra semilla, reordena el sistema o aplica un método directo.",
            }

        ea = float(np.linalg.norm(x_next - x_prev, ord=np.inf) /
                   max(np.linalg.norm(x_next, ord=np.inf), _DIV_ZERO_THRESH) * 100)
        delta = float(np.linalg.norm(x_next - x_prev, ord=np.inf))
        residuo = _residuo_lineal(A_original, x_next, b_original)
        resultados.append({
            "iteracion": i,
            "x_previo": ", ".join(f"x{j+1}={_fmt_num(v)}" for j, v in enumerate(x_prev)),
            "x_siguiente": ", ".join(f"x{j+1}={_fmt_num(v)}" for j, v in enumerate(x_next)),
            "ea": _fmt_num(ea),
            "ea_raw": ea,
            "delta": _fmt_num(delta),
            "residuo": _fmt_num(residuo),
            "residuo_raw": residuo,
        })

        x_prev = x_next
        if ea <= tol or residuo <= _DIV_ZERO_THRESH:
            grafica = _grafica_convergencia_sistema(resultados, color, metodo)
            extras = _extras_iterativo(B, c, rho, tol, analisis, i, ea, residuo)
            return {
                "error": False,
                "tipo": "sistema_iterativo",
                "metodo": metodo,
                "solucion": _fmt_vec(x_prev),
                "raiz": ", ".join(f"x{j+1}={_fmt_num(v)}" for j, v in enumerate(x_prev)),
                "convergencia": (
                    f"{metodo} convergió en {i} iteración(es). "
                    f"Error = {_fmt_num(ea)}%, residuo ||Ax-b||∞ = {_fmt_num(residuo)}, "
                    f"radio espectral = {_fmt_num(rho)}."
                ),
                "resultados": resultados,
                "pasos": [],
                "grafica": grafica,
                "extras": extras,
            }

    extras = _extras_iterativo(
        B, c, rho, tol, analisis,
        len(resultados),
        resultados[-1]["ea_raw"] if resultados else None,
        resultados[-1]["residuo_raw"] if resultados else None,
    )
    return {
        "error": True,
        "titulo": "⚠️ No convergió",
        "mensaje": (
            f"{metodo} no alcanzó la tolerancia de {_fmt_num(tol)}% "
            f"en {max_iter} iteración(es)."
        ),
        "consejo": "Aumenta el número de iteraciones, usa otra semilla o prueba un método directo.",
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
    parece_latex = "\\" in limpio or "{" in limpio or "}" in limpio
    if parece_latex:
        try:
            expr = parse_latex(limpio)
        except Exception:
            expr = parse_expr(
                limpio.replace("^", "**"),
                local_dict=local,
                transformations=_PARSER_TRANSFORMATIONS,
                evaluate=True,
            )
    else:
        try:
            expr = parse_expr(
                limpio,
                local_dict=local,
                transformations=_PARSER_TRANSFORMATIONS,
                evaluate=True,
            )
        except Exception:
            expr = parse_latex(limpio)

    expr = _normalizar_constantes_expr(expr)
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
#  BISECCIÓN
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
#  REGLA FALSA
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
# NEWTON-RAPHSON
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
# SERIE DE TAYLOR
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
        if coef == 0 or coef.is_zero is True:
            continue

        coef_real = _safe_float(sp.N(coef, 50))
        if coef_real is None:
            return {
                "error": True,
                "titulo": "🛑 Coeficiente complejo",
                "mensaje": f"El coeficiente de orden {potencia} no es real en x0 = {x0}.",
                "consejo": "Elige un centro x0 donde la serie sea real.",
            }
        if not math.isfinite(coef_real):
            return {
                "error": True,
                "titulo": "🛑 Coeficiente inválido",
                "mensaje": f"El coeficiente de orden {potencia} no es finito en x0 = {x0}.",
                "consejo": "Elige un centro x0 dentro del dominio de la función.",
            }

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
            "derivada_latex":    _math_inline(
                f"f^{{({n})}}\\left({sp.latex(x0_sym)}\\right) = {sp.latex(sp.N(val_deriv, _ROUND_DIGITS))}"
            ),
            "termino_latex":     _math_inline(_latex_expr(termino_simb)),
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
        "funcion_latex":   _math_inline(f"f\\left({sp.latex(var)}\\right) = {_latex_expr(f_sim, expand=False)}"),
        "polinomio_latex": _math_inline(f"P\\left({sp.latex(var)}\\right) = {_latex_expr(polinomio_taylor)}"),
        "grafica":         _grafica_b64(fig),
    }


# =============================================================================
# INTEGRACIÓN Y DIFERENCIACIÓN
# =============================================================================
def _validar_intervalo_integracion(f_sim, var, a, b):
    """Valida intervalo no degenerado y sin singularidades detectables."""
    if abs(float(b) - float(a)) < _DIV_ZERO_THRESH:
        return _error_parametro(
            "El intervalo no puede tener a y b iguales.",
            "Usa dos límites distintos para aproximar el área.",
        )

    izq, der = sorted((float(a), float(b)))
    try:
        singulares = sp.calculus.util.singularities(f_sim, var)
        dentro = singulares.intersect(sp.Interval(izq, der))
    except Exception:
        return None

    if dentro != sp.EmptySet:
        return {
            "error": True,
            "titulo": "⚠️ Intervalo no válido",
            "mensaje": (
                f"La función tiene una discontinuidad o singularidad en "
                f"[{_fmt_num(izq)}, {_fmt_num(der)}]: {str(dentro)[:120]}."
            ),
            "consejo": "Divide el intervalo o evita puntos donde la función no esté definida.",
        }
    return None


def _valor_exactitud_integral(f_sim, var, a, b):
    """Intenta calcular la integral exacta o numérica de referencia con SymPy."""
    try:
        integral = sp.integrate(f_sim, (var, sp.Float(a), sp.Float(b)))
        valor = _safe_float(sp.N(integral, 30))
        if valor is None or not math.isfinite(valor):
            return None, None
        return valor, _math_inline(
            f"\\int_{{{_latex_num(a)}}}^{{{_latex_num(b)}}}"
            f"{_latex_expr(f_sim, expand=False)}\\,d{sp.latex(var)}"
            f"={_latex_num(valor)}"
        )
    except Exception:
        return None, None


def _error_contra_referencia(aprox, exacto):
    if exacto is None:
        return None, None
    error_abs = abs(float(aprox) - float(exacto))
    error_pct = None
    if abs(float(exacto)) > _DIV_ZERO_THRESH:
        error_pct = error_abs / abs(float(exacto)) * 100.0
    return error_abs, error_pct


def _graficar_integracion(f_lam, var, a, b, nodos, valores, color, metodo):
    izq, der = sorted((float(a), float(b)))
    margen = max((der - izq) * 0.14, 1.0)
    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_lam, izq - margen, der + margen, color, f"f({var})")

    xs = np.array(nodos, dtype=float)
    ys = np.array(valores, dtype=float)
    orden = np.argsort(xs)
    xs_ord = xs[orden]
    ys_ord = ys[orden]

    if metodo in ("trapecio", "romberg"):
        for xi, xj, yi, yj in zip(xs_ord[:-1], xs_ord[1:], ys_ord[:-1], ys_ord[1:]):
            ax.fill([xi, xi, xj, xj], [0, yi, yj, 0],
                    color=color, alpha=0.13, edgecolor=color, linewidth=1.0)
            ax.plot([xi, xj], [yi, yj], color=_COLOR["limite"], lw=1.2, alpha=0.75)
    else:
        for i in range(0, len(xs_ord) - 2, 2):
            x_seg = np.linspace(xs_ord[i], xs_ord[i + 2], 120)
            with np.errstate(all="ignore"):
                y_seg = f_lam(x_seg)
            y_seg = np.array([_safe_float(v) for v in y_seg], dtype=float)
            ax.fill_between(x_seg, 0, y_seg, color=color, alpha=0.12)
            ax.axvline(xs_ord[i + 1], color=_COLOR["limite"], ls=":", lw=1.0, alpha=0.6)

    ax.scatter(xs_ord, ys_ord, s=36, color=_COLOR["raiz"], edgecolor=_BG,
               linewidth=0.7, zorder=6, label="Nodos")
    for i, (xi, yi) in enumerate(zip(xs_ord, ys_ord)):
        if i in {0, len(xs_ord) - 1} or len(xs_ord) <= 12:
            ax.annotate(f"x{i}", (xi, yi), xytext=(4, 8), textcoords="offset points",
                        color=_TEXT, fontsize=7)
    ax.set_title("Aproximación del área bajo la curva")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()
    return _grafica_b64(fig)


def _preparar_funcion_calculo(latex_str):
    f_sim, var, err = parsear_funcion(latex_str)
    if err:
        return None, None, None, err
    try:
        f_lam = sp.lambdify(var, f_sim, "numpy")
    except Exception as exc:
        return None, None, None, {
            "error": True,
            "titulo": "🛑 Función no evaluable",
            "mensaje": str(exc)[:200],
            "consejo": "Revisa la sintaxis de la función.",
        }
    return f_sim, var, f_lam, None


def metodo_trapecio(latex_str, a, b, n):
    """Regla del trapecio compuesta."""
    err = _validar_reales(a=a, b=b)
    if err:
        return err
    err = _validar_entero_rango(n, "Número de subintervalos", 1, 200)
    if err:
        return err
    a, b, n = float(a), float(b), int(float(n))

    f_sim, var, f_lam, err = _preparar_funcion_calculo(latex_str)
    if err:
        return err
    err = _validar_intervalo_integracion(f_sim, var, a, b)
    if err:
        return err

    h = (b - a) / n
    nodos = [a + i * h for i in range(n + 1)]
    valores = []
    for i, xi in enumerate(nodos):
        fxi, e = _eval_seguro(f_lam, xi, f"x_{i}")
        if e:
            return e
        valores.append(fxi)

    suma_ponderada = 0.0
    resultados = []
    for i, (xi, fxi) in enumerate(zip(nodos, valores)):
        peso = 1 if i in (0, n) else 2
        aporte = peso * fxi
        suma_ponderada += aporte
        resultados.append({
            "i": i,
            "x_i": _fmt_num(xi),
            "fx_i": _fmt_num(fxi),
            "peso": peso,
            "aporte": _fmt_num(aporte),
        })

    aproximacion = (h / 2.0) * suma_ponderada
    exacto, exacto_latex = _valor_exactitud_integral(f_sim, var, a, b)
    error_abs, error_pct = _error_contra_referencia(aproximacion, exacto)
    grafica = _graficar_integracion(f_lam, var, a, b, nodos, valores,
                                    _COLOR["trapecio"], "trapecio")

    estadisticas = [
        {"label": "h", "value": _fmt_num(h)},
        {"label": "Subintervalos", "value": n},
        {"label": "Suma ponderada", "value": _fmt_num(suma_ponderada)},
    ]
    if exacto is not None:
        estadisticas.extend([
            {"label": "Valor de referencia", "value": _fmt_num(exacto)},
            {"label": "Error absoluto", "value": _fmt_num(error_abs)},
            {"label": "Error %", "value": "---" if error_pct is None else _fmt_num(error_pct)},
        ])

    return {
        "error": False,
        "tipo": "trapecio",
        "titulo": "Regla del Trapecio",
        "resultado": _fmt_num(aproximacion),
        "resultado_label": "Integral aproximada",
        "convergencia": "Compuesta: aproxima el área con trapecios en cada subintervalo.",
        "funcion_latex": _math_inline(f"f\\left({sp.latex(var)}\\right)={_latex_expr(f_sim, expand=False)}"),
        "formula_latex": _math_inline(
            r"I \approx \frac{h}{2}\left[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\right]"
        ),
        "sustitucion_latex": _math_inline(
            f"I \\approx \\frac{{{_latex_num(h)}}}{{2}}\\left({ _latex_num(suma_ponderada) }\\right)"
            f"={_latex_num(aproximacion)}"
        ),
        "exacto_latex": exacto_latex,
        "error_abs": None if error_abs is None else _fmt_num(error_abs),
        "error_pct": None if error_pct is None else _fmt_num(error_pct),
        "estadisticas": estadisticas,
        "resultados": resultados,
        "columnas": [
            {"key": "i", "label": "i", "clase": "cell-iter"},
            {"key": "x_i", "label": "xᵢ", "clase": "cell-blue"},
            {"key": "fx_i", "label": "f(xᵢ)", "clase": "cell-result"},
            {"key": "peso", "label": "peso", "clase": "cell-purple"},
            {"key": "aporte", "label": "peso·f(xᵢ)", "clase": "cell-yellow"},
        ],
        "grafica": grafica,
    }


def metodo_romberg(latex_str, a, b, niveles):
    """Integración de Romberg basada en trapecio compuesto y Richardson."""
    err = _validar_reales(a=a, b=b)
    if err:
        return err
    err = _validar_entero_rango(niveles, "Número de niveles", 2, 8)
    if err:
        return err
    a, b, niveles = float(a), float(b), int(float(niveles))

    f_sim, var, f_lam, err = _preparar_funcion_calculo(latex_str)
    if err:
        return err
    err = _validar_intervalo_integracion(f_sim, var, a, b)
    if err:
        return err

    tabla = []
    resultados = []
    nodos_finales = []
    valores_finales = []

    for k in range(niveles):
        n_sub = 2 ** k
        h = (b - a) / n_sub
        nodos = [a + i * h for i in range(n_sub + 1)]
        valores = []
        for i, xi in enumerate(nodos):
            fxi, e = _eval_seguro(f_lam, xi, f"x_{i}")
            if e:
                return e
            valores.append(fxi)

        trapecio_k = h * (0.5 * valores[0] + sum(valores[1:-1]) + 0.5 * valores[-1])
        fila_romberg = [trapecio_k]
        for j in range(1, k + 1):
            refinado = fila_romberg[j - 1] + (
                fila_romberg[j - 1] - tabla[k - 1][j - 1]
            ) / ((4 ** j) - 1)
            fila_romberg.append(refinado)
        tabla.append(fila_romberg)

        fila = {
            "nivel": k,
            "subintervalos": n_sub,
            "h": _fmt_num(h),
        }
        for j in range(niveles):
            fila[f"r{j}"] = _fmt_num(fila_romberg[j]) if j <= k else "—"
        resultados.append(fila)

        if k == niveles - 1:
            nodos_finales = nodos
            valores_finales = valores

    aproximacion = tabla[-1][-1]
    exacto, exacto_latex = _valor_exactitud_integral(f_sim, var, a, b)
    error_abs, error_pct = _error_contra_referencia(aproximacion, exacto)
    grafica = _graficar_integracion(f_lam, var, a, b, nodos_finales, valores_finales,
                                    _COLOR["romberg"], "romberg")

    estadisticas = [
        {"label": "Niveles", "value": niveles},
        {"label": "Subintervalos finales", "value": 2 ** (niveles - 1)},
        {"label": "h final", "value": _fmt_num((b - a) / (2 ** (niveles - 1)))},
        {"label": "Mejor estimación", "value": _fmt_num(aproximacion)},
    ]
    if exacto is not None:
        estadisticas.extend([
            {"label": "Valor de referencia", "value": _fmt_num(exacto)},
            {"label": "Error absoluto", "value": _fmt_num(error_abs)},
            {"label": "Error %", "value": "---" if error_pct is None else _fmt_num(error_pct)},
        ])

    return {
        "error": False,
        "tipo": "romberg",
        "titulo": "Integración de Romberg",
        "resultado": _fmt_num(aproximacion),
        "resultado_label": "Integral aproximada",
        "convergencia": "Refina Trapecio con n = 1, 2, 4, ... y extrapolación de Richardson.",
        "funcion_latex": _math_inline(f"f\\left({sp.latex(var)}\\right)={_latex_expr(f_sim, expand=False)}"),
        "formula_latex": _math_inline(
            r"R_{k,j}=R_{k,j-1}+\frac{R_{k,j-1}-R_{k-1,j-1}}{4^j-1},\qquad R_{k,0}=T_{2^k}"
        ),
        "sustitucion_latex": _math_inline(
            f"I \\approx R_{{{niveles - 1},{niveles - 1}}}={_latex_num(aproximacion)}"
        ),
        "exacto_latex": exacto_latex,
        "error_abs": None if error_abs is None else _fmt_num(error_abs),
        "error_pct": None if error_pct is None else _fmt_num(error_pct),
        "estadisticas": estadisticas,
        "resultados": resultados,
        "columnas": (
            [
                {"key": "nivel", "label": "k", "clase": "cell-iter"},
                {"key": "subintervalos", "label": "n = 2ᵏ", "clase": "cell-blue"},
                {"key": "h", "label": "h", "clase": "cell-purple"},
            ] + [
                {"key": f"r{j}", "label": f"R(k,{j})", "clase": "cell-result" if j == niveles - 1 else "cell-yellow"}
                for j in range(niveles)
            ]
        ),
        "grafica": grafica,
    }


def _graficar_diferenciacion(f_lam, var, x0, h, puntos, valores, aproximacion, orden, color):
    margen = max(4.0 * abs(h), 1.0)
    fig, ax = _hacer_figura()
    _trazar_funcion(ax, f_lam, x0 - margen, x0 + margen, color, f"f({var})")

    xs = np.array(puntos, dtype=float)
    ys = np.array(valores, dtype=float)
    ax.scatter(xs, ys, s=55, color=_COLOR["raiz"], edgecolor=_BG,
               linewidth=0.8, zorder=6, label="Puntos usados")

    f0, _ = _eval_seguro(f_lam, x0, "x0")
    if f0 is not None:
        ax.plot(x0, f0, "D", color=_COLOR["limite"], ms=7, zorder=7, label="x₀")
        if orden == 1:
            linea_x = np.linspace(x0 - margen * 0.55, x0 + margen * 0.55, 80)
            linea_y = f0 + aproximacion * (linea_x - x0)
            ax.plot(linea_x, linea_y, color=_COLOR["limite"], lw=1.8, ls="--",
                    label="Recta con pendiente aproximada")

    for i, (xi, yi) in enumerate(zip(xs, ys)):
        ax.annotate(f"p{i}", (xi, yi), xytext=(4, 8), textcoords="offset points",
                    color=_TEXT, fontsize=7)
    ax.set_title("Aproximación local de la derivada")
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()
    return _grafica_b64(fig)


def metodo_diferenciacion_numerica(latex_str, x0, h, esquema="centrada"):
    """Diferenciación numérica por diferencias finitas básicas."""
    esquemas = {
        "adelante": {
            "nombre": "Diferencia hacia adelante",
            "orden": 1,
            "offsets": [0, 1],
            "coef": [-1, 1],
            "divisor": lambda h_: h_,
            "formula": r"f'(x_0)\approx\frac{f(x_0+h)-f(x_0)}{h}",
        },
        "atras": {
            "nombre": "Diferencia hacia atrás",
            "orden": 1,
            "offsets": [-1, 0],
            "coef": [-1, 1],
            "divisor": lambda h_: h_,
            "formula": r"f'(x_0)\approx\frac{f(x_0)-f(x_0-h)}{h}",
        },
        "centrada": {
            "nombre": "Diferencia centrada",
            "orden": 1,
            "offsets": [-1, 1],
            "coef": [-1, 1],
            "divisor": lambda h_: 2 * h_,
            "formula": r"f'(x_0)\approx\frac{f(x_0+h)-f(x_0-h)}{2h}",
        },
        "segunda_centrada": {
            "nombre": "Segunda derivada centrada",
            "orden": 2,
            "offsets": [-1, 0, 1],
            "coef": [1, -2, 1],
            "divisor": lambda h_: h_ ** 2,
            "formula": r"f''(x_0)\approx\frac{f(x_0+h)-2f(x_0)+f(x_0-h)}{h^2}",
        },
    }
    if esquema not in esquemas:
        return _error_parametro(
            "El esquema de diferenciación seleccionado no existe.",
            "Elige adelante, atrás, centrada o segunda derivada centrada.",
        )

    err = _validar_reales(x0=x0, h=h)
    if err:
        return err
    x0, h = float(x0), float(h)
    if h <= 0:
        return _error_parametro("h debe ser mayor que 0.", "Usa un paso positivo, por ejemplo 0.01.")
    if h < 1e-10:
        return _error_parametro(
            "h es demasiado pequeño para trabajar con seguridad numérica.",
            "Un h extremadamente pequeño puede causar cancelación por redondeo.",
        )

    f_sim, var, f_lam, err = _preparar_funcion_calculo(latex_str)
    if err:
        return err

    cfg = esquemas[esquema]
    puntos = [x0 + offset * h for offset in cfg["offsets"]]
    valores = []
    for i, xi in enumerate(puntos):
        fxi, e = _eval_seguro(f_lam, xi, f"p_{i}")
        if e:
            return e
        valores.append(fxi)

    divisor = cfg["divisor"](h)
    numerador = sum(coef * valor for coef, valor in zip(cfg["coef"], valores))
    if abs(divisor) < _DIV_ZERO_THRESH:
        return _error_parametro("El divisor del esquema quedó demasiado cerca de cero.")
    aproximacion = numerador / divisor

    exacto = None
    exacto_latex = None
    try:
        derivada = sp.diff(f_sim, var, cfg["orden"])
        exacto = _safe_float(sp.N(derivada.subs(var, sp.Float(x0)), 30))
        if exacto is not None and math.isfinite(exacto):
            exacto_latex = _math_inline(
                f"f^{{({cfg['orden']})}}\\left({_latex_num(x0)}\\right)="
                f"{_latex_num(exacto)}"
            )
        else:
            exacto = None
    except Exception:
        exacto = None

    error_abs, error_pct = _error_contra_referencia(aproximacion, exacto)
    resultados = []
    for i, (offset, coef, xi, fxi) in enumerate(zip(cfg["offsets"], cfg["coef"], puntos, valores)):
        etiqueta = "x₀" if offset == 0 else ("x₀+h" if offset > 0 else "x₀-h")
        resultados.append({
            "i": i,
            "punto": etiqueta,
            "x_i": _fmt_num(xi),
            "fx_i": _fmt_num(fxi),
            "coef": coef,
            "aporte": _fmt_num(coef * fxi),
        })

    grafica = _graficar_diferenciacion(
        f_lam, var, x0, h, puntos, valores, aproximacion,
        cfg["orden"], _COLOR["diferenciacion"],
    )
    estadisticas = [
        {"label": "x₀", "value": _fmt_num(x0)},
        {"label": "h", "value": _fmt_num(h)},
        {"label": "Numerador", "value": _fmt_num(numerador)},
        {"label": "Divisor", "value": _fmt_num(divisor)},
    ]
    if exacto is not None:
        estadisticas.extend([
            {"label": "Valor exacto", "value": _fmt_num(exacto)},
            {"label": "Error absoluto", "value": _fmt_num(error_abs)},
            {"label": "Error %", "value": "---" if error_pct is None else _fmt_num(error_pct)},
        ])

    return {
        "error": False,
        "tipo": "diferenciacion_numerica",
        "titulo": cfg["nombre"],
        "resultado": _fmt_num(aproximacion),
        "resultado_label": "Derivada aproximada" if cfg["orden"] == 1 else "Segunda derivada aproximada",
        "convergencia": (
            "La diferencia centrada suele ser más precisa que adelante/atrás con el mismo h."
            if esquema == "centrada" else
            "El resultado depende mucho del tamaño de h: si h es muy grande hay error de truncamiento; si es muy pequeño hay redondeo."
        ),
        "funcion_latex": _math_inline(f"f\\left({sp.latex(var)}\\right)={_latex_expr(f_sim, expand=False)}"),
        "formula_latex": _math_inline(cfg["formula"]),
        "sustitucion_latex": _math_inline(
            f"\\text{{aprox.}}=\\frac{{{_latex_num(numerador)}}}{{{_latex_num(divisor)}}}"
            f"={_latex_num(aproximacion)}"
        ),
        "exacto_latex": exacto_latex,
        "error_abs": None if error_abs is None else _fmt_num(error_abs),
        "error_pct": None if error_pct is None else _fmt_num(error_pct),
        "estadisticas": estadisticas,
        "resultados": resultados,
        "columnas": [
            {"key": "i", "label": "#", "clase": "cell-iter"},
            {"key": "punto", "label": "punto", "clase": "cell-blue"},
            {"key": "x_i", "label": "x", "clase": "cell-blue"},
            {"key": "fx_i", "label": "f(x)", "clase": "cell-result"},
            {"key": "coef", "label": "coef.", "clase": "cell-purple"},
            {"key": "aporte", "label": "coef·f(x)", "clase": "cell-yellow"},
        ],
        "grafica": grafica,
    }


# =============================================================================
#  PUNTO FIJO 
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
#  HORNER
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
# HORNER-NEWTON 
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
# MÜLLER
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
# MBAIRSTOW 
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
#  ELIMINACIÓN DE GAUSS
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
#  GAUSS-JORDAN 
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
#FACTORIZACIÓN LU (PA = LU)
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
#  JACOBI el goat
# =============================================================================
def metodo_jacobi(matriz_texto, inicial_texto, tol, max_iter):
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    tol, max_iter = float(tol), int(float(max_iter))

    A_original, b_original, err = _parse_matriz_aumentada(matriz_texto)
    if err:
        return err
    inicial, err = _parse_vector_inicial(inicial_texto, len(b_original))
    if err:
        return err

    A, b, B, c, rho, analisis, err = _preparar_iterativo(A_original, b_original, "jacobi")
    if err:
        return err

    return _resultado_sistema_iterativo(
        "Jacobi", A_original, b_original, A, b, B, c, rho,
        inicial, tol, max_iter, analisis, _COLOR["jacobi"],
    )


# =============================================================================
# GAUSS-SEIDEL 
# =============================================================================
def metodo_gauss_seidel(matriz_texto, inicial_texto, tol, max_iter):
    err = _validar_tol_iter(tol, max_iter)
    if err:
        return err
    tol, max_iter = float(tol), int(float(max_iter))

    A_original, b_original, err = _parse_matriz_aumentada(matriz_texto)
    if err:
        return err
    inicial, err = _parse_vector_inicial(inicial_texto, len(b_original))
    if err:
        return err

    A, b, B, c, rho, analisis, err = _preparar_iterativo(A_original, b_original, "gauss_seidel")
    if err:
        return err

    return _resultado_sistema_iterativo(
        "Gauss-Seidel", A_original, b_original, A, b, B, c, rho,
        inicial, tol, max_iter, analisis, _COLOR["gauss_seidel"],
    )


# =============================================================================
# NEWTON-RAPHSON PARA SISTEMAS NO LINEALES
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
    y_sim_latex = _latex_vector_simbolico(len(variables), "y")

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
            y_cero = np.zeros_like(x_actual)
            resultados.append({
                "iteracion": i,
                "x": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_actual)),
                "x_vector": _fmt_vec(x_actual),
                "F": _fmt_vec(F_val),
                "J": _fmt_matrix(J_val),
                "y": _fmt_vec(y_cero),
                "delta": _fmt_vec(y_cero),
                "x_siguiente": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_actual)),
                "x_siguiente_vector": _fmt_vec(x_actual),
                "norma": _fmt_num(norma_f),
                "ea": 0,
                "x_latex": _latex_vector_numeric(x_actual),
                "F_latex": _latex_vector_numeric(F_val),
                "J_latex": _latex_matrix_numeric(J_val),
                "sistema_latex": (
                    f"{_latex_matrix_numeric(J_val)}{y_sim_latex}="
                    f"{_latex_vector_numeric(F_val)}"
                ),
                "y_latex": _latex_vector_numeric(y_cero),
                "actualizacion_latex": (
                    f"x^{{({i})}}={_latex_vector_numeric(x_actual)}-"
                    f"{_latex_vector_numeric(y_cero)}={_latex_vector_numeric(x_actual)}"
                ),
            })
            break

        try:
            y_correccion = np.linalg.solve(J_val, F_val)
        except np.linalg.LinAlgError:
            return {
                "error": True,
                "titulo": "⚠️ Jacobiano singular",
                "mensaje": f"det(J) ≈ 0 en la iteración {i}; no se puede resolver J·y = F.",
                "consejo": "Prueba otra semilla inicial.",
            }

        x_siguiente = x_actual - y_correccion
        if (not np.all(np.isfinite(x_siguiente))) or np.linalg.norm(x_siguiente, ord=np.inf) > _DIVERGE_THRESH:
            return {
                "error": True,
                "titulo": "🚀 Divergencia",
                "mensaje": f"La iteración {i} salió del rango seguro.",
                "consejo": "Usa una semilla más cercana a la solución.",
            }

        ea = float(np.linalg.norm(y_correccion, ord=np.inf) /
                   max(np.linalg.norm(x_siguiente, ord=np.inf), _DIV_ZERO_THRESH) * 100)

        resultados.append({
            "iteracion": i,
            "x": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_actual)),
            "x_vector": _fmt_vec(x_actual),
            "F": _fmt_vec(F_val),
            "J": _fmt_matrix(J_val),
            "y": _fmt_vec(y_correccion),
            "delta": _fmt_vec(-y_correccion),
            "x_siguiente": ", ".join(f"{v}={_fmt_num(val)}" for v, val in zip(variables, x_siguiente)),
            "x_siguiente_vector": _fmt_vec(x_siguiente),
            "norma": _fmt_num(norma_f),
            "ea": _fmt_num(ea),
            "x_latex": _latex_vector_numeric(x_actual),
            "F_latex": _latex_vector_numeric(F_val),
            "J_latex": _latex_matrix_numeric(J_val),
            "sistema_latex": (
                f"{_latex_matrix_numeric(J_val)}{y_sim_latex}="
                f"{_latex_vector_numeric(F_val)}"
            ),
            "y_latex": _latex_vector_numeric(y_correccion),
            "actualizacion_latex": (
                f"x^{{({i})}}={_latex_vector_numeric(x_actual)}-"
                f"{_latex_vector_numeric(y_correccion)}={_latex_vector_numeric(x_siguiente)}"
            ),
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
        "funciones_latex": [sp.latex(e) for e in exprs],
        "jacobiano": [[str(J_sim[i, j]).replace("**", "^").replace("*", "·")
                       for j in range(J_sim.shape[1])]
                      for i in range(J_sim.shape[0])],
        "jacobiano_latex": sp.latex(J_sim),
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
#INTERPOLACIÓN DE NEWTON diferencias divididas
# =============================================================================
def metodo_newton_diferencias(puntos_texto, x_eval=None):
    puntos, err = _parse_puntos(puntos_texto, min_n=2, max_n=30, ordenar=True)
    if err:
        return err
    x_eval, err = _parse_x_eval(x_eval)
    if err:
        return err

    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    n = len(puntos)
    tabla_dd = [[None for _ in range(n)] for _ in range(n)]
    for i, y_val in enumerate(ys):
        tabla_dd[i][0] = y_val

    for j in range(1, n):
        for i in range(n - j):
            denom = xs[i + j] - xs[i]
            if abs(denom) <= _DIV_ZERO_THRESH:
                return _error_parametro(
                    "No se pueden calcular diferencias divididas con valores x repetidos."
                )
            tabla_dd[i][j] = (tabla_dd[i + 1][j - 1] - tabla_dd[i][j - 1]) / denom

    x = sp.Symbol("x")
    polinomio = sp.Integer(0)
    producto = sp.Integer(1)
    coeficientes = []

    for orden in range(n):
        if orden > 0:
            producto *= (x - xs[orden - 1])
        coef = tabla_dd[0][orden]
        termino = coef * producto
        polinomio += termino
        coeficientes.append({
            "orden": orden,
            "coeficiente": _fmt_num(coef),
            "termino": _fmt_expr(termino),
            "termino_latex": _math_inline(_latex_expr(termino)),
        })

    polinomio = sp.expand(polinomio)
    valor_eval = None
    if x_eval is not None:
        valor_eval, err = _evaluar_polinomio(polinomio, x, x_eval)
        if err:
            return err

    headers = ["f[xᵢ]"] + [f"Δ{j}" for j in range(1, n)]
    tabla = []
    for i in range(n):
        tabla.append({
            "i": i,
            "x": _fmt_num(xs[i]),
            "columnas": [
                _fmt_num(tabla_dd[i][j]) if j <= n - i - 1 else ""
                for j in range(n)
            ],
        })

    grafica = _grafica_interpolacion(
        puntos, polinomio, x, _COLOR["newton_diff"],
        "Polinomio de Newton", x_eval, valor_eval,
    )

    return {
        "error": False,
        "tipo": "newton_diferencias",
        "nombre": "Newton por diferencias divididas",
        "puntos": _puntos_para_tabla(puntos),
        "headers": headers,
        "tabla": tabla,
        "coeficientes": coeficientes,
        "polinomio_final": _fmt_expr(polinomio),
        "polinomio_latex": _math_inline(f"P\\left(x\\right) = {_latex_expr(polinomio)}"),
        "x_eval": _fmt_num(x_eval) if x_eval is not None else None,
        "valor_eval": _fmt_num(valor_eval) if valor_eval is not None else None,
        "raiz": _fmt_num(valor_eval) if valor_eval is not None else _fmt_expr(polinomio),
        "convergencia": (
            f"Se construyó un polinomio interpolante de grado máximo {n - 1} "
            f"con {n} puntos únicos."
        ),
        "grafica": grafica,
        "resultados": tabla,
    }


# =============================================================================
#  POLINOMIO DE LAGRANGE
# =============================================================================
def metodo_lagrange(puntos_texto, x_eval=None, metodo_resolucion="lagrange"):
    puntos, err = _parse_puntos(puntos_texto, min_n=2, max_n=30, ordenar=True)
    if err:
        return err
    x_eval, err = _parse_x_eval(x_eval)
    if err:
        return err

    metodo_resolucion = str(metodo_resolucion or "lagrange").strip().lower()
    alias_metodos = {
        "lagrange": "lagrange",
        "clasico": "lagrange",
        "clásico": "lagrange",
        "vandermonde": "vandermonde",
        "se": "vandermonde",
        "sistema": "vandermonde",
    }
    metodo_resolucion = alias_metodos.get(metodo_resolucion)
    if metodo_resolucion is None:
        return _error_parametro(
            "La forma de resolución de Lagrange no es válida.",
            "Elige Lagrange clásico o Vandermonde por sistema de ecuaciones.",
        )

    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    n = len(puntos)
    x = sp.Symbol("x")
    polinomio_lagrange = sp.Integer(0)
    bases = []

    for i in range(n):
        Li = sp.Integer(1)
        for j in range(n):
            if i != j:
                Li *= (x - xs[j]) / (xs[i] - xs[j])
        termino = ys[i] * Li
        polinomio_lagrange += termino
        bases.append({
            "i": i,
            "x": _fmt_num(xs[i]),
            "y": _fmt_num(ys[i]),
            "base": _fmt_expr(Li),
            "termino": _fmt_expr(termino),
            "base_latex": _math_inline(f"L_{{{i}}}\\left(x\\right) = {_latex_expr(Li)}"),
            "termino_latex": _math_inline(_latex_expr(termino)),
        })

    try:
        V = sp.Matrix([
            [sp.Float(xi, 15) ** potencia for potencia in range(n)]
            for xi in xs
        ])
        Y = sp.Matrix([sp.Float(yi, 15) for yi in ys])
        coefs = V.LUsolve(Y)
        polinomio_vandermonde = sp.expand(
            sum(coefs[potencia] * x ** potencia for potencia in range(n))
        )
        vandermonde = [[_fmt_num(V[i, j]) for j in range(n)] for i in range(n)]
        coeficientes = [
            {"potencia": i, "coeficiente": _fmt_num(coefs[i])}
            for i in range(n)
        ]
        coef_sim_latex = (
            r"\begin{bmatrix}"
            + r"\\ ".join(f"a_{i}" for i in range(n))
            + r"\end{bmatrix}"
        )
    except Exception as exc:
        return {
            "error": True,
            "titulo": "⚠️ Matriz de Vandermonde singular",
            "mensaje": str(exc)[:180],
            "consejo": "Verifica que todos los valores de x sean distintos.",
        }

    polinomio_lagrange = sp.expand(polinomio_lagrange)
    polinomio = polinomio_vandermonde if metodo_resolucion == "vandermonde" else polinomio_lagrange

    valor_eval = None
    if x_eval is not None:
        valor_eval, err = _evaluar_polinomio(polinomio, x, x_eval)
        if err:
            return err

    metodo_nombre = (
        "Vandermonde por sistema de ecuaciones"
        if metodo_resolucion == "vandermonde"
        else "Lagrange clásico"
    )
    grafica = _grafica_interpolacion(
        puntos, polinomio, x, _COLOR["lagrange"],
        metodo_nombre, x_eval, valor_eval,
    )

    return {
        "error": False,
        "tipo": "lagrange",
        "nombre": metodo_nombre,
        "metodo_resolucion": metodo_resolucion,
        "puntos": _puntos_para_tabla(puntos),
        "bases": bases,
        "vandermonde": vandermonde,
        "vandermonde_latex": _latex_matrix_numeric(V),
        "vector_y_latex": _latex_vector_numeric(Y),
        "coeficientes_latex": _latex_vector_numeric(coefs),
        "coeficientes_sim_latex": coef_sim_latex,
        "sistema_vandermonde_latex": (
            f"{_latex_matrix_numeric(V)}{coef_sim_latex}="
            f"{_latex_vector_numeric(Y)}"
        ),
        "coeficientes": coeficientes,
        "polinomio_final": _fmt_expr(polinomio),
        "polinomio_latex": _math_inline(f"P\\left(x\\right) = {_latex_expr(polinomio)}"),
        "polinomio_lagrange_latex": _math_inline(f"P_L\\left(x\\right) = {_latex_expr(polinomio_lagrange)}"),
        "polinomio_vandermonde_latex": _math_inline(f"P_V\\left(x\\right) = {_latex_expr(polinomio_vandermonde)}"),
        "x_eval": _fmt_num(x_eval) if x_eval is not None else None,
        "valor_eval": _fmt_num(valor_eval) if valor_eval is not None else None,
        "raiz": _fmt_num(valor_eval) if valor_eval is not None else _fmt_expr(polinomio),
        "convergencia": (
            f"{metodo_nombre} generó el polinomio único de grado máximo {n - 1} "
            f"con {n} puntos. Lagrange y Vandermonde son formas equivalentes."
        ),
        "grafica": grafica,
        "resultados": bases,
    }


# =============================================================================
#  INTERPOLACIÓN CON TRAZADORES CÚBICOS NATURALES
# =============================================================================
def metodo_trazadores_cubicos(puntos_texto, x_eval=None):
    puntos, err = _parse_puntos(puntos_texto, min_n=3, max_n=40, ordenar=True)
    if err:
        return err
    x_eval, err = _parse_x_eval(x_eval)
    if err:
        return err

    xs = np.array([p[0] for p in puntos], dtype=float)
    ys = np.array([p[1] for p in puntos], dtype=float)
    n = len(puntos)
    h = np.diff(xs)
    if np.any(h <= _DIV_ZERO_THRESH):
        return _error_parametro(
            "Los valores de x deben estar estrictamente ordenados y sin repetirse."
        )

    if x_eval is not None and (x_eval < xs[0] - _DIV_ZERO_THRESH or x_eval > xs[-1] + _DIV_ZERO_THRESH):
        return _error_parametro(
            "x a evaluar debe estar dentro del rango de los puntos.",
            f"Usa un valor entre {_fmt_num(xs[0])} y {_fmt_num(xs[-1])}.",
        )

    a = ys.copy()
    alpha = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        alpha[i] = (
            (3.0 / h[i]) * (a[i + 1] - a[i])
            - (3.0 / h[i - 1]) * (a[i] - a[i - 1])
        )

    l = np.ones(n, dtype=float)
    mu = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        if abs(l[i]) <= _DIV_ZERO_THRESH:
            return {
                "error": True,
                "titulo": "⚠️ Sistema singular",
                "mensaje": f"No se pudo resolver el trazador en el nodo {i}.",
                "consejo": "Revisa que los puntos estén bien espaciados.",
            }
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros(n, dtype=float)
    b = np.zeros(n - 1, dtype=float)
    d = np.zeros(n - 1, dtype=float)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = ((a[j + 1] - a[j]) / h[j]) - (h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0)
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    segmentos = []
    x_sym = sp.Symbol("x")
    for i in range(n - 1):
        dx_txt = f"(x - {_fmt_num(xs[i])})"
        dx_sym = x_sym - sp.Float(xs[i], 15)
        spline_expr = (
            sp.Float(a[i], 15)
            + sp.Float(b[i], 15) * dx_sym
            + sp.Float(c[i], 15) * dx_sym**2
            + sp.Float(d[i], 15) * dx_sym**3
        )
        segmentos.append({
            "i": i,
            "intervalo": f"[{_fmt_num(xs[i])}, {_fmt_num(xs[i + 1])}]",
            "a": _fmt_num(a[i]),
            "b": _fmt_num(b[i]),
            "c": _fmt_num(c[i]),
            "d": _fmt_num(d[i]),
            "polinomio": (
                f"{_fmt_num(a[i])} + {_fmt_num(b[i])}{dx_txt} + "
                f"{_fmt_num(c[i])}{dx_txt}^2 + {_fmt_num(d[i])}{dx_txt}^3"
            ),
            "polinomio_latex": _math_inline(
                f"S_{{{i}}}\\left(x\\right) = {_latex_expr(spline_expr, expand=False)}"
            ),
        })

    valor_eval = None
    intervalo_eval = None
    if x_eval is not None:
        idx = int(np.searchsorted(xs, x_eval, side="right") - 1)
        idx = max(0, min(idx, n - 2))
        dx = x_eval - xs[idx]
        valor_eval = a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3
        intervalo_eval = segmentos[idx]["intervalo"]

    fig, ax = _hacer_figura()
    for i in range(n - 1):
        x_seg = np.linspace(xs[i], xs[i + 1], 160)
        dx = x_seg - xs[i]
        y_seg = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        ax.plot(
            x_seg, y_seg, color=_COLOR["spline"], linewidth=2.2,
            label="Trazador cúbico" if i == 0 else None, zorder=3,
        )
    ax.scatter(xs, ys, s=55, color=_COLOR["raiz"], edgecolor=_BG,
               linewidth=1.0, zorder=5, label="Puntos")
    if x_eval is not None:
        ax.plot(x_eval, valor_eval, "D", color=_COLOR["limite"], ms=8,
                zorder=6, label=f"S({_fmt_num(x_eval)}) = {_fmt_num(valor_eval)}")
        ax.axvline(x_eval, color=_COLOR["limite"], ls=":", lw=1.1, alpha=0.75)
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    return {
        "error": False,
        "tipo": "trazadores_cubicos",
        "nombre": "Trazadores cúbicos naturales",
        "puntos": _puntos_para_tabla(puntos),
        "segmentos": segmentos,
        "x_eval": _fmt_num(x_eval) if x_eval is not None else None,
        "valor_eval": _fmt_num(valor_eval) if valor_eval is not None else None,
        "intervalo_eval": intervalo_eval,
        "raiz": _fmt_num(valor_eval) if valor_eval is not None else f"{n - 1} segmentos",
        "convergencia": (
            "Se resolvió el sistema tridiagonal del trazador natural "
            "con condición S''(x₀)=S''(xₙ)=0."
        ),
        "grafica": _grafica_b64(fig),
        "resultados": segmentos,
    }


# =============================================================================
#  REGRESIÓN LINEAL SIMPLE
# =============================================================================
def metodo_regresion_lineal(puntos_texto, x_eval=None):
    puntos, err = _parse_puntos(puntos_texto, min_n=2, max_n=80, ordenar=True)
    if err:
        return err
    x_eval, err = _parse_x_eval(x_eval, "x para predecir")
    if err:
        return err

    xs = np.array([p[0] for p in puntos], dtype=float)
    ys = np.array([p[1] for p in puntos], dtype=float)
    n = len(puntos)
    sx = float(np.sum(xs))
    sy = float(np.sum(ys))
    sxx = float(np.sum(xs * xs))
    syy = float(np.sum(ys * ys))
    sxy = float(np.sum(xs * ys))
    denom = n * sxx - sx**2

    if abs(denom) <= _DIV_ZERO_THRESH:
        return _error_parametro(
            "No se puede ajustar una recta si todos los valores de x son iguales.",
            "Cambia al menos una abscisa para que exista variación horizontal.",
        )

    pendiente = (n * sxy - sx * sy) / denom
    intercepto = (sy - pendiente * sx) / n
    estimados = intercepto + pendiente * xs
    residuos = ys - estimados
    ss_res = float(np.sum(residuos**2))
    ss_tot = float(np.sum((ys - np.mean(ys))**2))
    r2 = 1.0 if ss_tot <= _DIV_ZERO_THRESH and ss_res <= _DIV_ZERO_THRESH else (
        0.0 if ss_tot <= _DIV_ZERO_THRESH else 1.0 - ss_res / ss_tot
    )
    r2 = max(min(r2, 1.0), 0.0)
    denom_r = denom * (n * syy - sy**2)
    if denom_r > _DIV_ZERO_THRESH:
        r = (n * sxy - sx * sy) / math.sqrt(denom_r)
    else:
        r = 0.0
    error_estandar = math.sqrt(ss_res / (n - 2)) if n > 2 else 0.0

    valor_eval = None
    if x_eval is not None:
        valor_eval = intercepto + pendiente * x_eval

    tabla = []
    for i in range(n):
        tabla.append({
            "i": i,
            "x": _fmt_num(xs[i]),
            "y": _fmt_num(ys[i]),
            "y_estimado": _fmt_num(estimados[i]),
            "residuo": _fmt_num(residuos[i]),
        })

    x_min, x_max = _rango_desde_x(xs)
    x_line = np.linspace(x_min, x_max, 240)
    y_line = intercepto + pendiente * x_line
    fig, ax = _hacer_figura()
    ax.scatter(xs, ys, s=58, color=_COLOR["raiz"], edgecolor=_BG,
               linewidth=1.0, zorder=5, label="Datos")
    ax.plot(x_line, y_line, color=_COLOR["regresion"], linewidth=2.3,
            label="Recta ajustada", zorder=3)
    if x_eval is not None:
        ax.plot(x_eval, valor_eval, "D", color=_COLOR["limite"], ms=8,
                zorder=6, label=f"ŷ({_fmt_num(x_eval)}) = {_fmt_num(valor_eval)}")
        ax.axvline(x_eval, color=_COLOR["limite"], ls=":", lw=1.1, alpha=0.75)
    ax.legend(fontsize=8, facecolor=_LEGEND, edgecolor=_AXIS, labelcolor=_TEXT)
    fig.tight_layout()

    ecuacion = f"ŷ = {_fmt_num(intercepto)} + {_fmt_num(pendiente)}x"
    x_sym = sp.Symbol("x")
    recta_expr = sp.Float(intercepto, 15) + sp.Float(pendiente, 15) * x_sym
    return {
        "error": False,
        "tipo": "regresion_lineal",
        "nombre": "Regresión lineal simple",
        "puntos": _puntos_para_tabla(puntos),
        "tabla": tabla,
        "ecuacion": ecuacion,
        "ecuacion_latex": _math_inline(f"\\hat{{y}} = {_latex_expr(recta_expr)}"),
        "pendiente": _fmt_num(pendiente),
        "intercepto": _fmt_num(intercepto),
        "r": _fmt_num(r),
        "r2": _fmt_num(r2),
        "error_estandar": _fmt_num(error_estandar),
        "x_eval": _fmt_num(x_eval) if x_eval is not None else None,
        "valor_eval": _fmt_num(valor_eval) if valor_eval is not None else None,
        "raiz": _fmt_num(valor_eval) if valor_eval is not None else ecuacion,
        "convergencia": (
            f"Recta de mínimos cuadrados ajustada con {n} puntos. "
            f"R² = {_fmt_num(r2)}."
        ),
        "grafica": _grafica_b64(fig),
        "resultados": tabla,
        "estadisticas": [
            {"label": "Pendiente b₁", "value": _fmt_num(pendiente)},
            {"label": "Intercepto b₀", "value": _fmt_num(intercepto)},
            {"label": "r", "value": _fmt_num(r)},
            {"label": "R²", "value": _fmt_num(r2)},
            {"label": "Error estándar", "value": _fmt_num(error_estandar)},
        ],
    }


# =============================================================================
# aqui s eponen las ruta de flask que conectan a los metodos con las rutas web
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


def _deps_analisis_inteligente():
    return {
        "parsear_funcion": parsear_funcion,
        "parse_matriz": _parse_matriz_aumentada,
        "parse_puntos": _parse_puntos,
        "metodo_biseccion": metodo_biseccion,
        "metodo_falsa_posicion": metodo_falsa_posicion,
        "metodo_newton_raphson": metodo_newton_raphson,
        "metodo_secante": metodo_secante,
        "metodo_gauss": metodo_gauss,
        "metodo_gauss_jordan": metodo_gauss_jordan,
        "metodo_lu": metodo_lu,
        "metodo_jacobi": metodo_jacobi,
        "metodo_gauss_seidel": metodo_gauss_seidel,
        "metodo_newton_diferencias": metodo_newton_diferencias,
        "metodo_lagrange": metodo_lagrange,
        "metodo_trazadores_cubicos": metodo_trazadores_cubicos,
        "metodo_regresion_lineal": metodo_regresion_lineal,
    }


@app.route("/analisis_inteligente", methods=["GET", "POST"])
def analisis_inteligente_route():
    datos = None
    if request.method == "POST":
        datos = analisis_inteligente.analizar_solicitud(
            request.form,
            _deps_analisis_inteligente(),
        )
    return render_template("analisis_inteligente.html", datos=datos)


@app.route("/teoria_metodos")
def teoria_metodos_route():
    metodos_visibles = {"biseccion", "regla_falsa", "newton", "secante"}
    familias_visibles = {"todos", "cerrado", "abierto"}
    metodos = [m for m in obtener_teoria_metodos() if m["id"] in metodos_visibles]
    familias = [f for f in obtener_familias_teoria() if f["id"] in familias_visibles]
    return render_template(
        "teoria_metodos.html",
        metodos=metodos,
        familias=familias,
    )


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


@app.route("/trapecio", methods=["GET","POST"])
def trapecio():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_trapecio(
                request.form["ecuacion_latex"],
                float(request.form["a"]),
                float(request.form["b"]),
                int(request.form["n"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("trapecio.html", datos=datos)


@app.route("/romberg", methods=["GET","POST"])
def romberg():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_romberg(
                request.form["ecuacion_latex"],
                float(request.form["a"]),
                float(request.form["b"]),
                int(request.form["niveles"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("romberg.html", datos=datos)


@app.route("/diferenciacion_numerica", methods=["GET","POST"])
def diferenciacion_numerica():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_diferenciacion_numerica(
                request.form["ecuacion_latex"],
                float(request.form["x0"]),
                float(request.form["h"]),
                request.form.get("esquema", "centrada"),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("diferenciacion_numerica.html", datos=datos)


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


@app.route("/jacobi", methods=["GET","POST"])
def jacobi():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_jacobi(
                request.form["matriz"],
                request.form["inicial"],
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("jacobi.html", datos=datos)


@app.route("/gauss_seidel", methods=["GET","POST"])
def gauss_seidel():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_gauss_seidel(
                request.form["matriz"],
                request.form["inicial"],
                float(request.form["tol"]),
                int(request.form["max_iter"]),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("gauss_seidel.html", datos=datos)


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


@app.route("/newton_diferencias", methods=["GET","POST"])
def newton_diferencias():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_newton_diferencias(
                request.form["puntos"],
                request.form.get("x_eval", ""),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("newton_diferencias.html", datos=datos)


@app.route("/lagrange", methods=["GET","POST"])
def lagrange():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_lagrange(
                request.form["puntos"],
                request.form.get("x_eval", ""),
                request.form.get("metodo_resolucion", "lagrange"),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("lagrange.html", datos=datos)


@app.route("/trazadores_cubicos", methods=["GET","POST"])
def trazadores_cubicos():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_trazadores_cubicos(
                request.form["puntos"],
                request.form.get("x_eval", ""),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("trazadores_cubicos.html", datos=datos)


@app.route("/regresion_lineal", methods=["GET","POST"])
def regresion_lineal():
    datos = None
    if request.method == "POST":
        try:
            datos = metodo_regresion_lineal(
                request.form["puntos"],
                request.form.get("x_eval", ""),
            )
        except (KeyError, ValueError, OverflowError) as exc:
            datos = _error_formulario(exc)
        except Exception as exc:
            datos = {"error": True, "titulo": "🛑 Error inesperado",
                     "mensaje": str(exc)[:300]}
    return render_template("regresion_lineal.html", datos=datos)


if __name__ == "__main__":
    app.run(debug=True)
