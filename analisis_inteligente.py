"""Motor de reglas para el Asistente Inteligente de Métodos Numéricos.

El módulo no depende de Flask ni importa ``home.py``. La ruta Flask le inyecta
las funciones numéricas existentes para evitar acoplamiento circular y mantener
el sistema manual intacto.
"""

import itertools
import math

import numpy as np
import sympy as sp


DIV_ZERO = 1e-12


def analizar_solicitud(form, deps):
    """Analiza el problema seleccionado y devuelve un diccionario para la vista."""
    tipo = _texto(form.get("tipo_problema", "raices"))
    analizadores = {
        "raices": _analizar_raices,
        "sistemas_lineales": _analizar_sistemas_lineales,
        "interpolacion": _analizar_interpolacion,
        "regresion": _analizar_regresion,
    }
    if tipo not in analizadores:
        return _error(
            "Tipo de problema no reconocido",
            "Selecciona raíces, sistemas lineales, interpolación o regresión.",
        )

    try:
        datos = analizadores[tipo](form, deps)
    except Exception as exc:
        return _error(
            "No se pudo completar el análisis",
            str(exc)[:220],
            "Revisa los datos ingresados; el asistente controla errores, pero necesita entradas coherentes.",
        )

    datos.setdefault("error", False)
    datos.setdefault("tipo", "analisis_inteligente")
    datos.setdefault("problema", tipo)
    return datos


def _analizar_raices(form, deps):
    latex = _texto(form.get("ecuacion_latex"))
    if not latex:
        return _error("Función requerida", "Escribe una función f(x) para analizar métodos de raíces.")

    tol, err = _leer_float(form, "raiz_tol", "Tolerancia", default=0.001, positivo=True)
    if err:
        return err
    max_iter, err = _leer_int(form, "raiz_max_iter", "Máx. iteraciones", default=80, minimo=1, maximo=1000)
    if err:
        return err

    expr, var, parse_err = deps["parsear_funcion"](latex)
    if parse_err:
        return parse_err

    f = sp.lambdify(var, expr, "numpy")
    df_expr = sp.diff(expr, var)
    df = sp.lambdify(var, df_expr, "numpy")
    comparacion = []
    advertencias = []
    explicacion = []
    grafica = None

    a, err_a = _leer_float(form, "xl", "Límite inferior a", required=False)
    b, err_b = _leer_float(form, "xu", "Límite superior b", required=False)
    if err_a:
        return err_a
    if err_b:
        return err_b
    tiene_intervalo = err_a is None and err_b is None and a is not None and b is not None
    cambio_signo = False

    if tiene_intervalo:
        fa, fa_err = _eval_real(f, a)
        fb, fb_err = _eval_real(f, b)
        if fa_err or fb_err:
            razon = "f(a) o f(b) no existe como número real finito."
            comparacion.append(_fila("Bisección", False, "No aplicable", "-", "-", "-", razon, "Revisa dominio del intervalo.", "/biseccion"))
            comparacion.append(_fila("Regla Falsa", False, "No aplicable", "-", "-", "-", razon, "Revisa dominio del intervalo.", "/falsa_posicion"))
        elif fa * fb < 0:
            cambio_signo = True
            explicacion.append(
                "Bisección y Regla Falsa son aplicables porque f(a) y f(b) tienen signos opuestos. "
                "Por el Teorema del Valor Intermedio existe al menos una raíz dentro del intervalo."
            )
            for nombre, fn, url, estado in [
                ("Bisección", deps["metodo_biseccion"], "/biseccion", "Seguro pero lento"),
                ("Regla Falsa", deps["metodo_falsa_posicion"], "/falsa_posicion", "Seguro"),
            ]:
                datos = _ejecutar(fn, latex, a, b, tol, max_iter)
                comparacion.append(_fila_desde_resultado(nombre, datos, estado, url))
                grafica = grafica or datos.get("grafica")
        else:
            razon = f"f(a)={_fmt(fa)} y f(b)={_fmt(fb)} no tienen signos opuestos."
            comparacion.append(_fila("Bisección", False, "No aplicable", "-", "-", "-", razon, "Requiere f(a)·f(b)<0.", "/biseccion"))
            comparacion.append(_fila("Regla Falsa", False, "No aplicable", "-", "-", "-", razon, "Requiere cambio de signo.", "/falsa_posicion"))
            advertencias.append("Sin cambio de signo no hay garantía de raíz por métodos cerrados.")
    else:
        comparacion.append(_fila("Bisección", False, "No evaluada", "-", "-", "-", "Falta el intervalo [a,b].", "Completa a y b si quieres métodos cerrados.", "/biseccion"))
        comparacion.append(_fila("Regla Falsa", False, "No evaluada", "-", "-", "-", "Falta el intervalo [a,b].", "Completa a y b si quieres métodos cerrados.", "/falsa_posicion"))

    x0, err_x0 = _leer_float(form, "x0", "Punto inicial x0", required=False)
    if err_x0:
        return err_x0
    if err_x0 is None and x0 is not None:
        fx0, fx0_err = _eval_real(f, x0)
        dfx0, dfx0_err = _eval_real(df, x0)
        if fx0_err or dfx0_err:
            comparacion.append(_fila("Newton-Raphson", False, "Peligroso", "-", "-", "-", "f(x0) o f'(x0) no es real finita.", "Cambia x0 o revisa el dominio.", "/newton"))
        elif abs(dfx0) <= 1e-8:
            comparacion.append(_fila("Newton-Raphson", False, "Peligroso", "-", "-", "-", f"f'(x0)≈0 en x0={_fmt(x0)}.", "La tangente puede quedar horizontal y fallar.", "/newton"))
            advertencias.append("Newton es sensible al punto inicial; con derivada cercana a cero puede divergir.")
        else:
            datos = _ejecutar(deps["metodo_newton_raphson"], latex, x0, tol, max_iter)
            fila = _fila_desde_resultado("Newton-Raphson", datos, "Rápido", "/newton")
            fila["criterio"] = f"f'(x0)={_fmt(dfx0)} no está cerca de cero."
            comparacion.append(fila)
            grafica = grafica or datos.get("grafica")
            explicacion.append("Newton-Raphson es recomendable si la derivada no se anula cerca de la raíz; suele converger cuadráticamente.")
    else:
        comparacion.append(_fila("Newton-Raphson", False, "No evaluado", "-", "-", "-", "Falta x0.", "Completa x0 para analizar Newton.", "/newton"))

    x1, err_x1 = _leer_float(form, "x1", "Segundo punto x1", required=False)
    if err_x1:
        return err_x1
    if err_x0 is None and err_x1 is None and x0 is not None and x1 is not None:
        fx0, fx0_err = _eval_real(f, x0)
        fx1, fx1_err = _eval_real(f, x1)
        if abs(x1 - x0) <= DIV_ZERO:
            comparacion.append(_fila("Secante", False, "No aplicable", "-", "-", "-", "x0 y x1 son iguales.", "Usa dos semillas distintas.", "/secante"))
        elif fx0_err or fx1_err:
            comparacion.append(_fila("Secante", False, "Peligroso", "-", "-", "-", "f(x0) o f(x1) no existe como real finito.", "Cambia las semillas.", "/secante"))
        elif abs(fx1 - fx0) <= 1e-10:
            comparacion.append(_fila("Secante", False, "Peligroso", "-", "-", "-", "f(x1)-f(x0)≈0.", "La recta secante queda casi horizontal.", "/secante"))
        else:
            datos = _ejecutar(deps["metodo_secante"], latex, x0, x1, tol, max_iter)
            fila = _fila_desde_resultado("Secante", datos, "Rápido sin derivada", "/secante")
            fila["criterio"] = "Usa dos puntos iniciales y no necesita calcular f'(x)."
            comparacion.append(fila)
            grafica = grafica or datos.get("grafica")
    else:
        comparacion.append(_fila("Secante", False, "No evaluado", "-", "-", "-", "Faltan x0 y/o x1.", "Completa ambas semillas para analizar Secante.", "/secante"))

    recomendada = _mejor_fila(comparacion)
    if recomendada:
        texto = _recomendacion_raices(recomendada, cambio_signo)
    else:
        texto = "No hay un método recomendable con los datos actuales. Completa intervalo o semillas válidas."

    return {
        "titulo": "Análisis de raíces de ecuaciones",
        "resumen": [
            {"label": "Función", "value": f"f({var}) = {sp.sstr(expr)}"},
            {"label": "Derivada", "value": f"f'({var}) = {sp.sstr(df_expr)}"},
            {"label": "Tolerancia", "value": f"{_fmt(tol)}%"},
        ],
        "diagnostico": "Se revisaron métodos cerrados y abiertos con condiciones de aplicabilidad antes de ejecutarlos.",
        "comparacion": comparacion,
        "recomendacion": {"metodo": recomendada["metodo"] if recomendada else "Sin recomendación", "texto": texto},
        "advertencias": advertencias or ["Los métodos abiertos pueden converger rápido, pero dependen mucho de las semillas iniciales."],
        "explicacion": explicacion,
        "grafica": grafica,
    }


def _analizar_sistemas_lineales(form, deps):
    matriz = _texto(form.get("matriz"))
    if not matriz:
        return _error("Matriz requerida", "Ingresa una matriz aumentada para analizar el sistema lineal.")

    tol, err = _leer_float(form, "lineal_tol", "Tolerancia", default=0.0001, positivo=True)
    if err:
        return err
    max_iter, err = _leer_int(form, "lineal_max_iter", "Máx. iteraciones", default=100, minimo=1, maximo=1000)
    if err:
        return err

    A, b, parse_err = deps["parse_matriz"](matriz)
    if parse_err:
        return parse_err

    n = len(b)
    inicial = _texto(form.get("inicial")) or ",".join("0" for _ in range(n))
    det = float(np.linalg.det(A)) if n else 0.0
    singular = abs(det) <= 1e-10
    diagonal_original = _es_diagonal_dominante(A)
    reorden = _mejor_reorden_diagonal(A, b)
    diagonal_reordenada = reorden["dominante"]
    comparacion = []
    advertencias = []
    explicacion = []

    if singular:
        advertencias.append("El determinante está muy cerca de cero; el sistema puede no tener solución única.")
    else:
        for nombre, fn, url, estado in [
            ("Gauss", deps["metodo_gauss"], "/gauss", "Directo"),
            ("Gauss-Jordan", deps["metodo_gauss_jordan"], "/gauss_jordan", "Directo exacto"),
            ("LU", deps["metodo_lu"], "/lu", "Directo reutilizable"),
        ]:
            datos = _ejecutar(fn, matriz)
            comparacion.append(_fila_sistema_desde_resultado(nombre, datos, estado, url, "Método directo con pivoteo/triangularización."))

    for nombre, fn, metodo_iter, url in [
        ("Jacobi", deps["metodo_jacobi"], "jacobi", "/jacobi"),
        ("Gauss-Seidel", deps["metodo_gauss_seidel"], "gauss_seidel", "/gauss_seidel"),
    ]:
        iter_info = _analizar_iteracion(A, b, metodo_iter)
        datos = _ejecutar(fn, matriz, inicial, tol, max_iter)
        if not datos.get("error"):
            rho_real = _float_o_none((datos.get("extras") or {}).get("radio_espectral"))
            reordenado = bool((datos.get("extras") or {}).get("reordenado"))
            criterio = (
                f"ρ(B)={_fmt(rho_real)} después del análisis"
                + (" y con reordenamiento de filas." if reordenado else ".")
                + " Como ρ(B)<1, la convergencia está garantizada para esta matriz de iteración."
            )
            comparacion.append(_fila_sistema_desde_resultado(nombre, datos, "Iterativo convergente", url, criterio, rho=rho_real))
        else:
            criterio = (
                f"ρ(B) original={_fmt(iter_info['rho'])}. "
                + ("Como ρ(B)≥1, puede divergir." if iter_info["rho"] >= 1 else "El método no pudo ejecutarse con la semilla dada.")
            )
            comparacion.append(_fila(nombre, False, "No recomendado", "-", "-", "-", criterio, "Usa un método directo o reordena/escala el sistema.", url, rho=_fmt(iter_info["rho"])))

    if diagonal_original:
        explicacion.append("La matriz ya es diagonal dominante; eso favorece la convergencia de Jacobi y Gauss-Seidel.")
    elif diagonal_reordenada:
        explicacion.append("La matriz no era diagonal dominante en el orden original, pero se encontró un reordenamiento que la mejora.")
    else:
        advertencias.append("No se encontró diagonal dominante clara; los métodos iterativos deben justificarse con radio espectral.")

    recomendada = _mejor_sistema(comparacion, n)
    texto = _recomendacion_sistemas(recomendada, n) if recomendada else "No se pudo recomendar un método con seguridad."

    return {
        "titulo": "Análisis de sistemas lineales",
        "resumen": [
            {"label": "Dimensión", "value": f"{n} x {n}"},
            {"label": "det(A)", "value": _fmt(det)},
            {"label": "Diagonal dominante", "value": "Sí" if diagonal_original else ("Con reordenamiento" if diagonal_reordenada else "No")},
        ],
        "diagnostico": "Se revisó solución única, diagonal dominante, radio espectral y conveniencia entre métodos directos e iterativos.",
        "comparacion": comparacion,
        "recomendacion": {"metodo": recomendada["metodo"] if recomendada else "Sin recomendación", "texto": texto},
        "advertencias": advertencias,
        "explicacion": explicacion,
        "grafica": next((fila.get("grafica") for fila in comparacion if fila.get("grafica")), None),
    }


def _analizar_interpolacion(form, deps):
    puntos_texto = _texto(form.get("puntos"))
    if not puntos_texto:
        return _error("Puntos requeridos", "Ingresa al menos dos puntos para analizar interpolación.")

    puntos_crudos, crudo_err = _parse_puntos_liviano(puntos_texto)
    puntos, parse_err = deps["parse_puntos"](puntos_texto, 2, 40, True)
    if parse_err:
        return parse_err

    n = len(puntos)
    x_eval = _texto(form.get("x_eval_datos"))
    objetivo = _texto(form.get("objetivo_interpolacion", "auto"))
    ordenados = not crudo_err and all(puntos_crudos[i][0] <= puntos_crudos[i + 1][0] for i in range(len(puntos_crudos) - 1))
    comparacion = []
    advertencias = []
    explicacion = []
    grafica = None

    for nombre, fn, url, criterio, estado in [
        ("Newton diferencias divididas", deps["metodo_newton_diferencias"], "/newton_diferencias", "Construye una tabla incremental y permite extender con nuevos puntos con menos trabajo manual.", "Aplicable"),
        ("Lagrange", deps["metodo_lagrange"], "/lagrange", "Construye P(x)=Σ yᵢLᵢ(x), directo y didáctico para pocos puntos.", "Aplicable"),
    ]:
        datos = _ejecutar(fn, puntos_texto, x_eval)
        comparacion.append(_fila_datos_desde_resultado(nombre, datos, estado, url, criterio))
        grafica = grafica or datos.get("grafica")

    if n >= 3:
        datos = _ejecutar(deps["metodo_trazadores_cubicos"], puntos_texto, x_eval)
        comparacion.append(_fila_datos_desde_resultado("Trazadores cúbicos", datos, "Suave por tramos", "/trazadores_cubicos", "Evita usar un único polinomio de grado alto y conserva suavidad por intervalos."))
        grafica = grafica or datos.get("grafica")
    else:
        comparacion.append(_fila("Trazadores cúbicos", False, "No aplicable", "-", "-", "-", "Requiere al menos 3 puntos.", "Agrega más puntos para usar splines.", "/trazadores_cubicos"))

    if not ordenados:
        explicacion.append("Los puntos ingresados no estaban ordenados; el sistema los ordena por x antes de calcular.")
    if n > 10:
        advertencias.append("Con muchos puntos, un polinomio único de Lagrange/Newton puede oscilar; conviene considerar trazadores cúbicos.")
    if n <= 6:
        explicacion.append("Con pocos puntos, Lagrange y Newton por diferencias son apropiados y producen el mismo polinomio.")

    recomendada = _recomendar_interpolacion(comparacion, n, objetivo)
    texto = _texto_recomendacion_interpolacion(recomendada, n, objetivo)

    return {
        "titulo": "Análisis de interpolación",
        "resumen": [
            {"label": "Puntos", "value": str(n)},
            {"label": "Ordenados", "value": "Sí" if ordenados else "No, se ordenan por x"},
            {"label": "Objetivo", "value": _objetivo_interpolacion_label(objetivo)},
        ],
        "diagnostico": "Se revisó cantidad de puntos, repetición de abscisas, riesgo de oscilación y suavidad requerida.",
        "comparacion": comparacion,
        "recomendacion": {"metodo": recomendada["metodo"], "texto": texto},
        "advertencias": advertencias,
        "explicacion": explicacion,
        "grafica": grafica,
    }


def _analizar_regresion(form, deps):
    puntos_texto = _texto(form.get("puntos"))
    if not puntos_texto:
        return _error("Puntos requeridos", "Ingresa puntos para analizar regresión.")

    puntos, parse_err = deps["parse_puntos"](puntos_texto, 2, 80, True)
    if parse_err:
        return parse_err
    n = len(puntos)
    xs = np.array([p[0] for p in puntos], dtype=float)
    ys = np.array([p[1] for p in puntos], dtype=float)
    x_eval = _texto(form.get("x_eval_datos"))

    lineal = _ejecutar(deps["metodo_regresion_lineal"], puntos_texto, x_eval)
    r2_lineal = _float_o_none(lineal.get("r2"))
    mse_lineal = _mse_lineal(xs, ys)
    comparacion = [
        _fila_datos_desde_resultado(
            "Regresión lineal",
            lineal,
            "Modelo disponible",
            "/regresion_lineal",
            f"R²={_fmt(r2_lineal)}; mide qué tanto la recta explica la variación de los datos.",
            r2=_fmt(r2_lineal),
            mse=_fmt(mse_lineal),
        )
    ]
    grafica = lineal.get("grafica")
    advertencias = []
    explicacion = []

    r2_quad = None
    mse_quad = None
    if n >= 3:
        r2_quad, mse_quad = _ajuste_polinomico(xs, ys, 2)
        comparacion.append(_fila(
            "Regresión polinómica grado 2",
            True,
            "Sugerencia analítica",
            "-",
            "-",
            "-",
            f"R²={_fmt(r2_quad)}; se compara contra la recta para detectar curvatura.",
            "No reemplaza el método manual actual; es una recomendación del asistente.",
            "#",
            r2=_fmt(r2_quad),
            mse=_fmt(mse_quad),
        ))
    else:
        comparacion.append(_fila("Regresión polinómica", False, "No aplicable", "-", "-", "-", "Se necesitan al menos 3 puntos para ajustar una parábola.", "Agrega más datos.", "#"))

    if n < 4:
        advertencias.append("Hay pocos datos; cualquier ajuste puede verse artificialmente bueno.")

    if r2_lineal is not None and r2_lineal >= 0.95:
        recomendada = comparacion[0]
        texto = "Conviene regresión lineal: la recta explica muy bien los datos y mantiene el modelo simple."
        explicacion.append("La regresión lineal es preferible cuando R² es alto porque evita sobreajustar.")
    elif r2_quad is not None and r2_lineal is not None and r2_quad - r2_lineal > 0.08:
        recomendada = comparacion[1]
        texto = "Los datos muestran curvatura: una regresión polinómica de grado 2 explica mejor la tendencia que una recta."
        advertencias.append("El sistema manual actual tiene regresión lineal; el asistente sugiere polinómica como extensión matemática.")
    else:
        recomendada = comparacion[0]
        texto = "La regresión lineal es una primera aproximación razonable; no hay evidencia fuerte de curvatura con estos datos."

    return {
        "titulo": "Análisis de regresión",
        "resumen": [
            {"label": "Puntos", "value": str(n)},
            {"label": "R² lineal", "value": _fmt(r2_lineal)},
            {"label": "MSE lineal", "value": _fmt(mse_lineal)},
        ],
        "diagnostico": "Se comparó ajuste lineal contra una curvatura cuadrática sencilla mediante R² y error cuadrático medio.",
        "comparacion": comparacion,
        "recomendacion": {"metodo": recomendada["metodo"], "texto": texto},
        "advertencias": advertencias,
        "explicacion": explicacion,
        "grafica": grafica,
    }


def _fila(metodo, aplicable, estado, iteraciones, raiz, error_final, criterio, riesgo, url, **extras):
    fila = {
        "metodo": metodo,
        "aplicable": aplicable,
        "estado": estado,
        "estado_key": _estado_key(estado, aplicable),
        "iteraciones": iteraciones,
        "resultado": raiz,
        "error_final": error_final,
        "criterio": criterio,
        "riesgo": riesgo,
        "url": url,
    }
    fila.update(extras)
    return fila


def _fila_desde_resultado(nombre, datos, estado, url):
    if datos.get("error"):
        return _fila(nombre, False, "No aplicable", "-", "-", "-", datos.get("mensaje", "El método devolvió error."), datos.get("consejo", "Revisa los datos."), url)
    return _fila(
        nombre,
        True,
        estado,
        len(datos.get("resultados") or []),
        datos.get("raiz", "-"),
        _error_final(datos),
        datos.get("convergencia", "Método ejecutado correctamente."),
        "Depende de las condiciones iniciales." if nombre in {"Newton-Raphson", "Secante"} else "Puede requerir más iteraciones.",
        url,
        grafica=datos.get("grafica"),
    )


def _fila_sistema_desde_resultado(nombre, datos, estado, url, criterio, rho=None):
    if datos.get("error"):
        return _fila(nombre, False, "No aplicable", "-", "-", "-", datos.get("mensaje", criterio), datos.get("consejo", "Usa otro método."), url, rho=_fmt(rho) if rho is not None else "-")
    return _fila(
        nombre,
        True,
        estado,
        len(datos.get("resultados") or datos.get("pasos") or []),
        datos.get("raiz", "-"),
        _error_final(datos),
        criterio,
        "Puede crecer en costo para sistemas grandes." if "Directo" in estado else "Depende de ρ(B) y de la semilla.",
        url,
        rho=_fmt(rho) if rho is not None else "-",
        grafica=datos.get("grafica"),
    )


def _fila_datos_desde_resultado(nombre, datos, estado, url, criterio, **extras):
    if datos.get("error"):
        return _fila(nombre, False, "No aplicable", "-", "-", "-", datos.get("mensaje", criterio), datos.get("consejo", "Revisa los datos."), url, **extras)
    return _fila(
        nombre,
        True,
        estado,
        "-",
        datos.get("raiz", datos.get("valor_eval", "-")),
        "-",
        criterio,
        "La calidad depende de la distribución de los puntos.",
        url,
        grafica=datos.get("grafica"),
        **extras,
    )


def _error_final(datos):
    filas = datos.get("resultados") or []
    if not filas:
        return "-"
    for fila in reversed(filas):
        ea = fila.get("ea")
        if ea not in (None, "---", "-"):
            return ea
    return "-"


def _mejor_fila(comparacion):
    candidatos = [f for f in comparacion if f.get("aplicable")]
    if not candidatos:
        return None
    prioridad = {"Newton-Raphson": 0, "Secante": 1, "Regla Falsa": 2, "Bisección": 3}
    return min(candidatos, key=lambda f: (f.get("iteraciones") if isinstance(f.get("iteraciones"), int) else 9999, prioridad.get(f["metodo"], 9)))


def _mejor_sistema(comparacion, n):
    aplicables = [f for f in comparacion if f.get("aplicable")]
    if not aplicables:
        return None
    iterativos = [f for f in aplicables if f["metodo"] in {"Jacobi", "Gauss-Seidel"} and _float_o_none(f.get("rho")) is not None]
    convergentes = [f for f in iterativos if _float_o_none(f.get("rho")) < 1]
    if convergentes and n >= 5:
        return min(convergentes, key=lambda f: (_float_o_none(f.get("rho")), 0 if f["metodo"] == "Gauss-Seidel" else 1))
    for preferido in ["LU", "Gauss-Jordan", "Gauss"]:
        for fila in aplicables:
            if fila["metodo"] == preferido:
                return fila
    return aplicables[0]


def _recomendar_interpolacion(comparacion, n, objetivo):
    aplicables = [f for f in comparacion if f["aplicable"]]
    por_nombre = {f["metodo"]: f for f in aplicables}
    if objetivo == "suavidad" and "Trazadores cúbicos" in por_nombre:
        return por_nombre["Trazadores cúbicos"]
    if objetivo == "agregar" and "Newton diferencias divididas" in por_nombre:
        return por_nombre["Newton diferencias divididas"]
    if n > 10 and "Trazadores cúbicos" in por_nombre:
        return por_nombre["Trazadores cúbicos"]
    return por_nombre.get("Newton diferencias divididas") or aplicables[0]


def _texto_recomendacion_interpolacion(fila, n, objetivo):
    if fila["metodo"] == "Trazadores cúbicos":
        return "Se recomiendan trazadores cúbicos porque evitan un único polinomio de grado alto y dan una curva suave por tramos."
    if fila["metodo"] == "Newton diferencias divididas":
        return "Se recomienda Newton por diferencias divididas porque construye el polinomio por tabla y es más cómodo si luego agregas puntos."
    return "Lagrange es adecuado para pocos puntos y para mostrar claramente cómo cada punto aporta una base Lᵢ(x)."


def _recomendacion_raices(fila, cambio_signo):
    if fila["metodo"] == "Newton-Raphson":
        extra = " Bisección queda como respaldo seguro." if cambio_signo else ""
        return "El método recomendado es Newton-Raphson porque fue aplicable y normalmente requiere menos iteraciones al usar la derivada." + extra
    if fila["metodo"] == "Secante":
        return "El método recomendado es Secante: no necesita derivada y las semillas producen una recta secante válida."
    if fila["metodo"] == "Regla Falsa":
        return "El método recomendado es Regla Falsa: conserva la garantía de cambio de signo y suele avanzar más que Bisección."
    return "El método recomendado es Bisección porque el cambio de signo garantiza convergencia, aunque sea más lento."


def _recomendacion_sistemas(fila, n):
    if fila["metodo"] == "Gauss-Seidel":
        return "Se recomienda Gauss-Seidel porque el radio espectral indica convergencia y suele aprovechar mejor los valores recién actualizados que Jacobi."
    if fila["metodo"] == "Jacobi":
        return "Se recomienda Jacobi porque el radio espectral es menor que 1; es simple y paralelizable, aunque suele ser más lento que Gauss-Seidel."
    if fila["metodo"] == "LU":
        return "Se recomienda LU porque resuelve de forma directa y permite reutilizar la factorización si cambia el vector b."
    if fila["metodo"] == "Gauss-Jordan":
        return "Se recomienda Gauss-Jordan para sistemas pequeños porque deja la solución visible en la matriz reducida."
    return "Se recomienda un método directo porque ofrece una solución más robusta cuando los iterativos no tienen convergencia clara."


def _analizar_iteracion(A, b, metodo):
    try:
        B = _matriz_iteracion(A, b, metodo)
        rho = float(np.max(np.abs(np.linalg.eigvals(B)))) if B.size else 0.0
    except Exception:
        rho = math.inf
    return {"rho": rho}


def _matriz_iteracion(A, b, metodo):
    A = np.asarray(A, dtype=float)
    D = np.diag(np.diag(A))
    if np.any(np.abs(np.diag(A)) <= DIV_ZERO):
        raise ValueError("diagonal cero")
    if metodo == "jacobi":
        return -np.linalg.solve(D, A - D)
    DL = np.tril(A)
    U = np.triu(A, 1)
    return -np.linalg.solve(DL, U)


def _es_diagonal_dominante(A):
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return False
    cumple = []
    estricta = False
    for i in range(A.shape[0]):
        diag = abs(A[i, i])
        resto = float(np.sum(np.abs(A[i, :])) - diag)
        cumple.append(diag + DIV_ZERO >= resto)
        estricta = estricta or diag > resto + DIV_ZERO
    return all(cumple) and estricta


def _mejor_reorden_diagonal(A, b):
    n = A.shape[0]
    mejor = (None, -1, -math.inf)
    perms = itertools.permutations(range(n)) if n <= 8 else [_greedy_perm(A)]
    for perm in perms:
        PA = A[list(perm), :]
        score = 0
        margen_min = math.inf
        for i in range(n):
            diag = abs(PA[i, i])
            resto = float(np.sum(np.abs(PA[i, :])) - diag)
            if diag + DIV_ZERO >= resto:
                score += 1
            margen_min = min(margen_min, diag - resto)
        if score > mejor[1] or (score == mejor[1] and margen_min > mejor[2]):
            mejor = (perm, score, margen_min)
    perm = mejor[0] or tuple(range(n))
    PA = A[list(perm), :]
    return {"perm": perm, "dominante": _es_diagonal_dominante(PA)}


def _greedy_perm(A):
    n = A.shape[0]
    disponibles = set(range(n))
    perm = []
    for col in range(n):
        mejor = max(
            disponibles,
            key=lambda row: abs(A[row, col]) / max(float(np.sum(np.abs(A[row, :]))) - abs(A[row, col]), DIV_ZERO),
        )
        perm.append(mejor)
        disponibles.remove(mejor)
    return tuple(perm)


def _parse_puntos_liviano(texto):
    puntos = []
    try:
        for raw in str(texto).replace(";", "\n").splitlines():
            linea = raw.strip()
            if not linea:
                continue
            partes = linea.strip("()[]{}").replace("|", " ").replace(",", " ").split()
            if len(partes) != 2:
                return [], True
            puntos.append((float(sp.N(sp.sympify(partes[0].replace("^", "**")))), float(sp.N(sp.sympify(partes[1].replace("^", "**"))))))
        return puntos, False
    except Exception:
        return [], True


def _mse_lineal(xs, ys):
    if len(xs) < 2:
        return None
    m, c = np.polyfit(xs, ys, 1)
    pred = m * xs + c
    return float(np.mean((ys - pred) ** 2))


def _ajuste_polinomico(xs, ys, grado):
    coefs = np.polyfit(xs, ys, grado)
    pred = np.polyval(coefs, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 if ss_tot <= DIV_ZERO and ss_res <= DIV_ZERO else (0.0 if ss_tot <= DIV_ZERO else 1.0 - ss_res / ss_tot)
    mse = float(np.mean((ys - pred) ** 2))
    return max(min(r2, 1.0), 0.0), mse


def _ejecutar(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        return {"error": True, "mensaje": str(exc)[:200], "consejo": "El método no pudo ejecutarse con estos datos."}


def _eval_real(fn, x):
    try:
        val = complex(fn(float(x)))
        if abs(val.imag) > 1e-8 or not math.isfinite(val.real):
            return None, True
        return float(val.real), False
    except Exception:
        return None, True


def _leer_float(form, nombre, etiqueta, default=None, required=True, positivo=False):
    raw = form.get(nombre)
    if raw is None or str(raw).strip() == "":
        if required and default is None:
            return None, _error(f"{etiqueta} requerido", f"Completa el campo {etiqueta}.")
        return default, None
    try:
        valor = float(str(raw).strip())
    except (TypeError, ValueError, OverflowError):
        return None, _error("Entrada inválida", f"{etiqueta} debe ser un número real.")
    if not math.isfinite(valor):
        return None, _error("Entrada inválida", f"{etiqueta} debe ser finito.")
    if positivo and valor <= 0:
        return None, _error("Entrada inválida", f"{etiqueta} debe ser mayor que cero.")
    return valor, None


def _leer_int(form, nombre, etiqueta, default=None, minimo=1, maximo=1000):
    raw = form.get(nombre)
    if raw is None or str(raw).strip() == "":
        if default is not None:
            return default, None
        return None, _error(f"{etiqueta} requerido", f"Completa el campo {etiqueta}.")
    try:
        valor = float(str(raw).strip())
    except (TypeError, ValueError, OverflowError):
        return None, _error("Entrada inválida", f"{etiqueta} debe ser un entero.")
    if not valor.is_integer():
        return None, _error("Entrada inválida", f"{etiqueta} debe ser un entero.")
    valor = int(valor)
    if valor < minimo or valor > maximo:
        return None, _error("Entrada inválida", f"{etiqueta} debe estar entre {minimo} y {maximo}.")
    return valor, None


def _texto(valor):
    return "" if valor is None else str(valor).strip()


def _fmt(valor):
    if valor is None:
        return "-"
    try:
        if not math.isfinite(float(valor)):
            return "∞"
        return round(float(valor), 8)
    except Exception:
        return str(valor)


def _float_o_none(valor):
    try:
        if valor in (None, "-", "---"):
            return None
        numero = float(valor)
        return numero if math.isfinite(numero) else None
    except Exception:
        return None


def _estado_key(estado, aplicable):
    if not aplicable:
        return "bad"
    limpio = estado.lower()
    if "rápido" in limpio or "convergente" in limpio:
        return "good"
    if "directo" in limpio or "seguro" in limpio or "modelo" in limpio:
        return "ok"
    if "peligroso" in limpio or "no recomendado" in limpio:
        return "warn"
    return "ok"


def _objetivo_interpolacion_label(valor):
    return {
        "auto": "Automático",
        "polinomio": "Polinomio único",
        "suavidad": "Suavidad por tramos",
        "agregar": "Agregar puntos luego",
    }.get(valor, "Automático")


def _error(titulo, mensaje, consejo=None):
    return {
        "error": True,
        "titulo": f"⚠️ {titulo}",
        "mensaje": mensaje,
        "consejo": consejo or "Corrige los datos y vuelve a intentar.",
    }
