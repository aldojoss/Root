from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg') # Esto evita que la gráfica intente abrirse en una ventana de Windows y bloquee el servidor
import matplotlib.pyplot as plt
import io
import base64
import math
from sympy import symbols, lambdify
from sympy.parsing.latex import parse_latex
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- FUNCIÓN AUXILIAR PARA TRADUCIR LATEX ---
def procesar_ecuacion(latex_str):
    """
    Toma una cadena LaTeX, la convierte en una expresión de SymPy 
    y luego en una función evaluable por Python/NumPy.
    """
    # Definimos 'x' como la variable matemática
    x = symbols('x')
    
    # 1. Convertimos el texto LaTeX a una expresión matemática de SymPy
    # Ejemplo: "\frac{x^2}{2}" se vuelve x**2 / 2
    expresion_sympy = parse_latex(latex_str)
    
    # 2. Convertimos esa expresión a una función real de Python que usa NumPy
    # Esto nos permite pasarle valores: f(5), f(2.3), etc.
    funcion_evaluable = lambdify(x, expresion_sympy, modules=['numpy', 'math'])
    
    # Retornamos ambas cosas: la función para evaluar y la expresión por si 
    # ocupamos derivarla (como en el método de Newton)
    return funcion_evaluable, expresion_sympy, x

def metodo_biseccion(latex_str, xl, xu, tol, max_iter):
    x = sp.Symbol('x')
    
    # --- BLOQUEO DE SEGURIDAD ---
    # Si llega vacío, cortamos de una vez
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe la función en la pizarra virtual antes de calcular."
        }
    
    try:
        # === LIMPIEZA EXTREMA DEL LATEX ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Esto convierte TODA la ecuación a minúsculas automáticamente
        
        # Traducimos
        funcion_simbolica = parse_latex(latex_limpio)
        funcion_simbolica = funcion_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- NUEVA MAGIA DINÁMICA ---
        # Detectamos qué letra usó el usuario
        simbolos_usados = list(funcion_simbolica.free_symbols)
        
        # Quitamos constantes como 'e' o 'pi' por si SymPy las confunde con variables
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}. Los métodos de este proyecto solo soportan 1 sola incógnita.",
                "consejo": "Usa solo una letra (ej: solo x, o solo t)."
            }
        
        if len(simbolos_usados) == 0:
            return {
                "error": True,
                "titulo": "🛑 Falta la Variable",
                "mensaje": "La ecuación es solo un número constante.",
                "consejo": "Asegúrate de incluir una variable (como x) en tu función."
            }

        # Asignamos la letra que el usuario escogió
        variable_dinamica = simbolos_usados[0]

        # Convertimos la función para que NumPy la pueda evaluar usando ESA letra
        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy')

        xl_original = xl
        xu_original = xu
        
        fxl = float(f(xl))
        fxu = float(f(xu))

    except Exception as err:
        # ¡ESTO ES VITAL! Imprimirá en tu terminal negra de VS Code el texto exacto que causó el error
        print(f"==============\n🛑 FALLA AL PARSEAR:\nTexto original: '{latex_str}'\nTexto limpio: '{latex_limpio}'\n==============") 
        
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"SymPy no pudo entender la función. Detalle: {str(err)}",
            "consejo": "Revisa la consola (terminal) de VS Code para ver qué texto exacto envió MathLive."
        }

    # === Si llegamos aquí, los números son perfectos ===
    if fxl * fxu >= 0:
        return {
            "error": True,
            "titulo": "⚠️ Intervalo sin cambio de signo",
            "mensaje": f"Evaluamos tus límites y obtuvimos f({xl}) = {round(fxl, 8)} y f({xu}) = {round(fxu, 8)}. Mismo signo.",
            "consejo": "Para la bisección, un resultado debe ser positivo y el otro negativo."
        }

    resultados = []
    xr_anterior = 0

    # 3. Ciclo iterativo
    for i in range(1, max_iter + 1):
        xr = (xl + xu) / 2
        fxl = f(xl)
        fxr = f(xr)
        
        # Calcular el error aproximado (salvo en la primera iteración)
        ea = abs((xr - xr_anterior) / xr) * 100 if i > 1 else 100
        
        # Guardamos los datos de esta iteración para la tabla
        resultados.append({
            "iteracion": i,
            "xl": round(xl, 8),
            "xu": round(xu, 8),
            "xr": round(xr, 8),
            "fxl": round(fxl, 8),
            "fxr": round(fxr, 8),
            "ea": round(ea, 8) if i > 1 else "---"
        })

        # Criterio de parada por tolerancia
        if i > 1 and ea < tol:
            break

        # Reasignar límites
        if fxl * fxr < 0:
            xu = xr
        elif fxl * fxr > 0:
            xl = xr
        else:
            break # Encontramos la raíz exacta
            
        xr_anterior = xr

    # === GENERAR LA GRÁFICA ===
    # Usamos los valores originales para que la gráfica no se corte
    margen = (xu_original - xl_original) * 0.5 
    x_vals = np.linspace(xl_original - margen, xu_original + margen, 200)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='f(x)', color='#0d6efd', linewidth=2)
    plt.axhline(0, color='black', linewidth=1) # Eje X
    
    # Dibujar los límites originales y la raíz encontrada
    plt.axvline(xl_original, color='orange', linestyle='--', label='xl inicial')
    plt.axvline(xu_original, color='purple', linestyle='--', label='xu inicial')
    plt.plot(xr, 0, 'ro', markersize=8, label=f'Raíz ({round(xr, 8)})')
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Convertir la gráfica a una imagen base64 para enviarla al HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "resultados": resultados, 
        "raiz": round(xr, 8),
        "convergencia": "Lineal O(n) - El error se reduce a la mitad por iteración.",
        "grafica": grafica_url
    }

# ==========================================
# MÉTODO 2: REGLA FALSA (FALSA POSICIÓN)
# ==========================================
def metodo_falsa_posicion(latex_str, xl, xu, tol, max_iter):
    # Ya no quemamos x = sp.Symbol('x') aquí, dejamos que se detecte sola
    
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Escribe la función en la pizarra virtual antes de calcular."
        }
    
    try:
        # === 1. LIMPIEZA EXTREMA Y TRADUCCIÓN ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Parche anti-X mayúscula y otras letras
        
        # Traducimos de LaTeX a SymPy
        funcion_simbolica = parse_latex(latex_limpio)
        funcion_simbolica = funcion_simbolica.subs(sp.Symbol('e'), sp.E) # Forzamos Euler
        
        # --- DETECCIÓN DE VARIABLE DINÁMICA ---
        simbolos_usados = list(funcion_simbolica.free_symbols)
        # Filtramos constantes por si acaso
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "Los métodos de una sola raíz solo aceptan 1 variable (ej: solo x o solo t)."
            }
        
        if len(simbolos_usados) == 0:
            # Si es una constante (ej: f(x) = 5), no hay raíz que buscar
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]
        
        # Convertimos a función evaluable por NumPy usando la variable detectada
        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy') 

        # PRUEBA DE FUEGO: Evaluamos los límites
        xl_original = xl
        xu_original = xu
        fxl = float(f(xl))
        fxu = float(f(xu))

    except Exception as err:
        print(f"🛑 Error parseando: {latex_str}")
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo calcular la función. Detalle: {str(err)}",
            "consejo": "Verifica que la ecuación esté completa (que no falten números en fracciones o potencias)."
        }

    # === VALIDACIÓN DE BOLZANO ===
    if fxl * fxu >= 0:
        return {
            "error": True,
            "titulo": "⚠️ Intervalo sin cambio de signo",
            "mensaje": f"f({xl}) = {round(fxl, 6)} y f({xu}) = {round(fxu, 6)}. Ambos tienen el mismo signo.",
            "consejo": "Para Regla Falsa, la función debe cruzar el eje X entre los dos límites."
        }

    resultados = []
    xr_anterior = 0

    # === CICLO DE REGLA FALSA ===
    for i in range(1, max_iter + 1):
        fxl = f(xl)
        fxu = f(xu)
        
        # Evitar división por cero (línea horizontal)
        if fxl - fxu == 0:
            break

        # LA FÓRMULA DE REGLA FALSA (Intersección de la secante)
        xr = xu - (fxu * (xl - xu)) / (fxl - fxu)
        fxr = f(xr)

        ea = abs((xr - xr_anterior) / xr) * 100 if i > 1 else 100

        resultados.append({
            "iteracion": i,
            "xl": round(float(xl), 8),
            "xu": round(float(xu), 8),
            "xr": round(float(xr), 8),
            "fxl": round(float(fxl), 8),
            "fxr": round(float(fxr), 8),
            "ea": round(float(ea), 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            break

        # Reemplazo de límites
        if fxl * fxr < 0:
            xu = xr
        else:
            xl = xr

        xr_anterior = xr

    # === GENERAR LA GRÁFICA ===
    margen = (xu_original - xl_original) * 0.5
    x_vals = np.linspace(xl_original - margen, xu_original + margen, 200)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f({variable_dinamica})', color='#198754', linewidth=2) 
    plt.axhline(0, color='black', linewidth=1) 
    
    plt.axvline(xl_original, color='orange', linestyle='--', label='xl inicial')
    plt.axvline(xu_original, color='purple', linestyle='--', label='xu inicial')
    plt.plot(xr, 0, 'ro', markersize=8, label=f'Raíz ({round(float(xr), 6)})')
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "resultados": resultados, 
        "raiz": round(float(xr), 8),
        "convergencia": f"Lineal - Se utilizó la variable '{variable_dinamica}'.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 3: NEWTON-RAPHSON
# ==========================================
def metodo_newton_raphson(latex_str, x0, tol, max_iter):
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe la función en la pizarra virtual antes de calcular."
        }

    try:
        # === LIMPIEZA EXTREMA Y TRADUCCIÓN ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*')
        latex_limpio = latex_limpio.lower() # Anti-mayúsculas

        funcion_simbolica = parse_latex(latex_limpio)
        funcion_simbolica = funcion_simbolica.subs(sp.Symbol('e'), sp.E)

        # --- VARIABLE DINÁMICA ---
        simbolos_usados = list(funcion_simbolica.free_symbols)
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "Newton-Raphson solo soporta 1 variable (ej: solo x)."
            }

        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]

        # MAGIA PURA: Derivada analítica con la variable dinámica
        derivada_simbolica = sp.diff(funcion_simbolica, variable_dinamica)

        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy')
        df = sp.lambdify(variable_dinamica, derivada_simbolica, 'numpy')

        # Prueba de fuego
        float(f(x0))
        float(df(x0))

    except Exception as err:
        print(f"🛑 Error parseando: {latex_str}")
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar la función o su derivada. Detalle: {str(err)}",
            "consejo": "Verifica que la ecuación esté bien escrita."
        }

    resultados = []
    xi = float(x0)

    for i in range(1, max_iter + 1):
        try:
            fxi = float(f(xi))
            dfxi = float(df(xi))
        except (OverflowError, TypeError):
            return {
                "error": True,
                "titulo": "🚀 ¡El método explotó (Divergencia)!",
                "mensaje": f"En la iteración {i}, los números se volvieron demasiado grandes o la raíz se volvió imaginaria.",
                "consejo": "Newton-Raphson es rápido pero sensible. Prueba con un x0 diferente, más cercano a la raíz real."
            }

        # Validación crítica: Evitar división por cero
        if dfxi == 0:
            return {
                "error": True,
                "titulo": "⚠️ Derivada Cero (Línea Horizontal)",
                "mensaje": f"En la iteración {i}, la derivada se volvió 0.",
                "consejo": "El método falla cuando toca un valle o cresta. Intenta con otro x0."
            }

        # Fórmula de Newton-Raphson
        x_siguiente = xi - (fxi / dfxi)

        # Calcular error
        ea = abs((x_siguiente - xi) / x_siguiente) * 100 if x_siguiente != 0 else 100

        resultados.append({
            "iteracion": i,
            "xi": round(xi, 8),
            "fxi": round(fxi, 8),
            "dfxi": round(dfxi, 8),
            "x_siguiente": round(x_siguiente, 8),
            "ea": round(ea, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            xi = x_siguiente
            break

        xi = x_siguiente

    # === GENERAR LA GRÁFICA ===
    margen = abs(xi - float(x0)) * 0.5 if abs(xi - float(x0)) > 0 else 2
    x_min = min(float(x0), xi) - margen
    x_max = max(float(x0), xi) + margen

    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals)
    y_vals = f(x_vals)

    altura_maxima = max(abs(f(x_min)), abs(f(x_max))) * 3
    y_vals = np.clip(y_vals, -altura_maxima, altura_maxima)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f({variable_dinamica})', color='#dc3545', linewidth=2) # Rojo para Newton
    plt.axhline(0, color='black', linewidth=1)

    plt.axvline(float(x0), color='orange', linestyle='--', label='x0 inicial')
    plt.plot(xi, 0, 'go', markersize=8, label=f'Raíz ({round(xi, 8)})')

    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "abierto", # Mantiene la tabla formateada para métodos abiertos
        "resultados": resultados,
        "raiz": round(xi, 8),
        "convergencia": f"Cuadrática O(n²) - Variable usada: '{variable_dinamica}'.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 4: SECANTE
# ==========================================
def metodo_secante(latex_str, x0, x1, tol, max_iter):
    x = sp.Symbol('x')
    
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe la función en la pizarra virtual antes de calcular."
        }
    
    try:
        # === LIMPIEZA EXTREMA DEL LATEX ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Esto convierte TODA la ecuación a minúsculas automáticamente
        
        # Traducimos
        funcion_simbolica = parse_latex(latex_limpio)
        funcion_simbolica = funcion_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- NUEVA MAGIA DINÁMICA ---
        # Detectamos qué letra usó el usuario
        simbolos_usados = list(funcion_simbolica.free_symbols)
        
        # Quitamos constantes como 'e' o 'pi' por si SymPy las confunde con variables
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}. Los métodos de este proyecto solo soportan 1 sola incógnita.",
                "consejo": "Usa solo una letra (ej: solo x, o solo t)."
            }
        
        if len(simbolos_usados) == 0:
            return {
                "error": True,
                "titulo": "🛑 Falta la Variable",
                "mensaje": "La ecuación es solo un número constante.",
                "consejo": "Asegúrate de incluir una variable (como x) en tu función."
            }

        # Asignamos la letra que el usuario escogió
        variable_dinamica = simbolos_usados[0]

        # Convertimos la función para que NumPy la pueda evaluar usando ESA letra
        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy')

        # Prueba de fuego numérica con los valores iniciales
        float(f(x0))
        float(f(x1))

    except Exception as err:
        # Imprimirá en tu terminal negra de VS Code el texto exacto que causó el error
        print(f"==============\n🛑 FALLA AL PARSEAR:\nTexto original: '{latex_str}'\nTexto limpio: '{latex_limpio}'\n==============") 
        
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"SymPy no pudo entender la función. Detalle: {str(err)}",
            "consejo": "Revisa la consola (terminal) de VS Code para ver qué texto exacto envió MathLive."
        }

    resultados = []
    xi_menos_1 = x0
    xi = x1

    for i in range(1, max_iter + 1):
        fxi_menos_1 = f(xi_menos_1)
        fxi = f(xi)
        
        # Evitar división por cero
        if (fxi_menos_1 - fxi) == 0:
            return {
                "error": True,
                "titulo": "⚠️ División por Cero",
                "mensaje": f"En la iteración {i}, f(xi-1) y f(xi) son iguales.",
                "consejo": "Esto genera una línea horizontal sin intersección. Intenta con otros valores iniciales."
            }

        # Fórmula de la Secante
        x_siguiente = xi - (fxi * (xi_menos_1 - xi)) / (fxi_menos_1 - fxi)
        
        ea = abs((x_siguiente - xi) / x_siguiente) * 100 if x_siguiente != 0 else 100

        resultados.append({
            "iteracion": i,
            "xi_menos_1": round(xi_menos_1, 8),
            "xi": round(xi, 8),
            "fxi_menos_1": round(fxi_menos_1, 8),
            "fxi": round(fxi, 8),
            "x_siguiente": round(x_siguiente, 8),
            "ea": round(ea, 8) if i > 1 else "---" 
        })

        if ea < tol:
            break
            
        # Preparar la siguiente iteración
        xi_menos_1 = xi
        xi = x_siguiente

    # === GENERAR LA GRÁFICA ===
    margen = abs(xi - x0) if abs(xi - x0) > 0 else 2
    x_min = min(x0, x1, xi) - margen
    x_max = max(x0, x1, xi) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals) 
    y_vals = f(x_vals)

    altura_maxima = max(abs(f(x_min)), abs(f(x_max))) * 3
    y_vals = np.clip(y_vals, -altura_maxima, altura_maxima)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='f(x)', color='#0dcaf0', linewidth=2) # Cyan para la Secante
    plt.axhline(0, color='black', linewidth=1) 
    
    # Puntos iniciales
    plt.axvline(x0, color='orange', linestyle=':', label='x0 inicial')
    plt.axvline(x1, color='purple', linestyle=':', label='x1 inicial')
    plt.plot(xi, 0, 'ro', markersize=8, label=f'Raíz ({round(xi, 8)})') 
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "secante", # Activa la nueva tabla en el HTML
        "resultados": resultados, 
        "raiz": round(xi, 8),
        "convergencia": "Superlineal (Entre Bisección y Newton-Raphson).",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 5: SERIES DE TAYLOR
# ==========================================
def serie_taylor(latex_str, x0, x_eval, orden):
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe la función en la pizarra virtual antes de calcular."
        }
    
    try:
        # === LIMPIEZA EXTREMA DEL LATEX ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Parche anti-mayúsculas
        
        # Traducción matemática
        f_simbolica = parse_latex(latex_limpio)
        f_simbolica = f_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- VARIABLE DINÁMICA ---
        simbolos_usados = list(f_simbolica.free_symbols)
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "El método de Taylor en este proyecto solo soporta funciones de 1 variable."
            }
        
        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]
        
        # Convertimos la función usando nuestra letra dinámica
        f_numpy = sp.lambdify(variable_dinamica, f_simbolica, 'numpy') 
        
        # Evaluamos el valor verdadero desde el inicio
        valor_verdadero = float(f_numpy(x_eval))
        
    except Exception as err:
        print(f"🛑 Error parseando: {latex_str}") 
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"SymPy no pudo entender la función. Detalle: {str(err)}",
            "consejo": "Verifica que la ecuación esté bien escrita."
        }

    resultados = []
    aproximacion = 0
    derivada_actual = f_simbolica
    polinomio_taylor = 0

    for i in range(orden + 1):
        try:
            # 1. Evaluar la derivada en el punto x0 usando la letra correcta
            df_x0 = derivada_actual.subs(variable_dinamica, x0).evalf()
            
            # 2. Armar el término algebraico: ( f^(n)(x0) / n! ) * (variable - x0)^n
            termino_algebraico = (df_x0 / math.factorial(i)) * (variable_dinamica - x0)**i
            polinomio_taylor += termino_algebraico
            
            # 3. Evaluar el término en el punto solicitado (x_eval)
            valor_termino = termino_algebraico.subs(variable_dinamica, x_eval).evalf()
            aproximacion += valor_termino
            
        except (OverflowError, TypeError):
            # Seguro por si la serie estalla (Radio de Convergencia excedido)
            return {
                "error": True,
                "titulo": "🚀 ¡La Serie Explotó!",
                "mensaje": f"En el orden {i}, los números se volvieron demasiado grandes o imaginarios.",
                "consejo": "El punto a evaluar está demasiado lejos del centro. ¡Has excedido el radio de convergencia de esta función!"
            }

        # 4. Calcular Error Verdadero (Et)
        if valor_verdadero != 0:
            et = abs((valor_verdadero - aproximacion) / valor_verdadero) * 100
        else:
            et = abs(valor_verdadero - aproximacion) * 100

        resultados.append({
            "orden": i,
            "derivada": str(derivada_actual).replace('**', '^'),
            "df_x0": round(float(df_x0), 8),
            "termino": round(float(valor_termino), 8),
            "aproximacion": round(float(aproximacion), 8),
            "et": round(float(et), 8)
        })

        # Preparar la siguiente derivada usando nuestra letra
        derivada_actual = sp.diff(derivada_actual, variable_dinamica)

    # === GENERAR LA GRÁFICA (Comparativa) ===
    margen = abs(x_eval - x0) + 1
    x_min = min(x0, x_eval) - margen
    x_max = max(x0, x_eval) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals) 
    y_vals_original = f_numpy(x_vals)
    
    # Evaluar el polinomio de Taylor completo para la gráfica
    p_numpy = sp.lambdify(variable_dinamica, polinomio_taylor, 'numpy')
    y_vals_taylor = p_numpy(x_vals)
    
    # Si el polinomio resulta ser una constante, numpy necesita un array del mismo tamaño
    if isinstance(y_vals_taylor, (int, float)):
        y_vals_taylor = np.full_like(x_vals, y_vals_taylor)

    plt.figure(figsize=(8, 5))
    # Función real
    plt.plot(x_vals, y_vals_original, label=f'Original: f({variable_dinamica})', color='black', linewidth=3) 
    # Aproximación de Taylor
    plt.plot(x_vals, y_vals_taylor, label=f'Polinomio Taylor (Orden {orden})', color='#ffc107', linestyle='--', linewidth=2) 
    
    plt.axhline(0, color='gray', linewidth=1) 
    plt.axvline(x0, color='orange', linestyle=':', label=f'{variable_dinamica}0 (Centro)')
    plt.axvline(x_eval, color='purple', linestyle=':', label=f'{variable_dinamica} a evaluar')
    
    plt.plot(x_eval, float(valor_verdadero), 'ko', label=f'Valor Real ({round(float(valor_verdadero), 8)})')
    plt.plot(x_eval, float(aproximacion), 'yo', label=f'Aprox ({round(float(aproximacion), 8)})')
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    
    # Filtrar valores matemáticamente inválidos (NaN o Infinitos) antes de fijar la altura
    y_validos = y_vals_original[np.isfinite(y_vals_original)]
    
    if len(y_validos) > 0:
        plt.ylim(np.min(y_validos) - 2, np.max(y_validos) + 2)
        
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "taylor", 
        "resultados": resultados, 
        "raiz": round(float(aproximacion), 8), 
        "convergencia": f"Aproximación Polinomial. Variable usada: '{variable_dinamica}'.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 6: PUNTO FIJO (CON PREDICTOR)
# ==========================================
def metodo_punto_fijo(latex_gx_str, x0, tol, max_iter):
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_gx_str or latex_gx_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Ecuación g(x) vacía",
            "mensaje": "No se recibió ninguna ecuación despejada.",
            "consejo": "Por favor, escribe tu función g(x) en la pizarra virtual antes de calcular."
        }
    
    try:
        # === LIMPIEZA EXTREMA DEL LATEX ===
        latex_limpio = latex_gx_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Anti-mayúsculas
        
        # Traducción matemática
        g_simbolica = parse_latex(latex_limpio)
        g_simbolica = g_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- VARIABLE DINÁMICA ---
        simbolos_usados = list(g_simbolica.free_symbols)
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "Punto Fijo solo soporta 1 variable en el despeje."
            }
        
        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]
        
        # Convertimos la función a NumPy usando la variable detectada
        g = sp.lambdify(variable_dinamica, g_simbolica, 'numpy') 
        
        # EL PREDICTOR DE CONVERGENCIA (La magia)
        # Derivamos g usando nuestra variable dinámica (CRÍTICO)
        derivada_g = sp.diff(g_simbolica, variable_dinamica) 
        dg = sp.lambdify(variable_dinamica, derivada_g, 'numpy')
        
        # Evaluamos el valor absoluto de g'(x0)
        criterio_convergencia = abs(float(dg(x0)))
        
        # Generamos el diagnóstico
        if criterio_convergencia < 1:
            diagnostico = f"¡Excelente despeje! |g'({variable_dinamica}0)| = {round(criterio_convergencia, 8)} < 1. Convergencia garantizada."
        else:
            diagnostico = f"¡Alerta de Divergencia! |g'({variable_dinamica}0)| = {round(criterio_convergencia, 8)} > 1. La función explotará al infinito."

    except Exception as err:
        print(f"🛑 Error parseando: {latex_gx_str}") 
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar la función o su derivada. Detalle: {str(err)}",
            "consejo": "Revisa que la ecuación esté completa en la pizarra virtual."
        }

    resultados = []
    xi = float(x0)
    diverge = False

    for i in range(1, max_iter + 1):
        try:
            # Intentamos evaluar y forzar a que sea un número decimal normal
            gxi = float(g(xi))
        except TypeError:
            # Si da TypeError, la raíz dio un número imaginario
            return {
                "error": True,
                "titulo": "🛑 Raíz Compleja o Variable Inválida",
                "mensaje": f"En la iteración {i}, el cálculo generó un número imaginario (raíz negativa).",
                "consejo": "Para este valor inicial, la función no tiene solución real. ¡Intenta con otro valor o revisa tu despeje!"
            }
        except OverflowError:
            diverge = True
            break
            
        if abs(gxi) > 1e6:
            diverge = True
            break

        ea = abs((gxi - xi) / gxi) * 100 if gxi != 0 else 100

        resultados.append({
            "iteracion": i,
            "xi": round(xi, 8),
            "gxi": round(gxi, 8),
            "ea": round(ea, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            xi = gxi
            break
            
        xi = gxi

    if diverge:
        return {
            "error": True,
            "titulo": "🚀 ¡El método explotó (Divergencia)!",
            "mensaje": diagnostico, 
            "consejo": "Esta calculadora hace milagros, pero no sabe despejar por ti. Si no sabes hacer un despeje algebraico válido, te toca repasar las reglas del álgebra. 📚"
        }

    # === GENERAR LA GRÁFICA ===
    margen = abs(xi - float(x0)) + 2
    x_min = min(float(x0), xi) - margen
    x_max = max(float(x0), xi) - margen if max(float(x0), xi) == min(float(x0), xi) else max(float(x0), xi) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals) 
    y_vals_g = g(x_vals)
    y_vals_identidad = x_vals 

    altura_maxima = max(abs(g(x_min)), abs(g(x_max))) * 2
    y_vals_g = np.clip(y_vals_g, -altura_maxima, altura_maxima)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals_g, label=f'g({variable_dinamica})', color='#6f42c1', linewidth=2) 
    plt.plot(x_vals, y_vals_identidad, label=f'y = {variable_dinamica}', color='gray', linestyle='--', linewidth=1.5) 
    plt.axhline(0, color='black', linewidth=1) 
    plt.axvline(float(x0), color='orange', linestyle=':', label=f'{variable_dinamica}0 inicial')
    plt.plot(xi, xi, 'ro', markersize=8, label=f'Raíz ({round(xi, 8)})') 
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "punto_fijo", 
        "resultados": resultados, 
        "raiz": round(xi, 8),
        "convergencia": diagnostico, 
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 7: MÉTODO DE HORNER
# ==========================================
def metodo_horner(latex_str, x0):
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Polinomio vacío",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe el polinomio en la pizarra virtual antes de calcular."
        }
    
    try:
        # === 1. LIMPIEZA Y TRADUCCIÓN ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Parche anti-X mayúscula
        
        f_simbolica = parse_latex(latex_limpio)
        f_simbolica = f_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- DETECCIÓN DE VARIABLE DINÁMICA ---
        simbolos_usados = list(f_simbolica.free_symbols)
        # Filtramos constantes
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "El método de Horner solo funciona con polinomios de 1 sola variable."
            }
        
        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita (ej: x)."}

        variable_dinamica = simbolos_usados[0]

        # Validación de seguridad: ¿Es realmente un polinomio en la letra detectada?
        if not f_simbolica.is_polynomial(variable_dinamica):
            return {
                "error": True,
                "titulo": "🛑 Función No Polinomial",
                "mensaje": f"Horner SOLO funciona con polinomios en '{variable_dinamica}'.",
                "consejo": f"Asegúrate de que '{variable_dinamica}' no esté en denominadores, raíces o dentro de funciones como seno/logaritmo."
            }
            
        polinomio = sp.Poly(f_simbolica, variable_dinamica)
        coeffs = polinomio.all_coeffs() # Extrae todos los coeficientes
        n = polinomio.degree()
        
    except Exception as err:
        print(f"🛑 Error parseando Horner: {latex_str}") 
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar el polinomio. Detalle: {str(err)}",
            "consejo": "Verifica que el polinomio esté bien escrito en la pizarra."
        }

    resultados = []
    
    # La primera iteración (b_n = a_n)
    b_actual = float(coeffs[0])
    resultados.append({
        "grado": n,
        "a": round(float(coeffs[0]), 8),
        "operacion": "---",
        "b": round(b_actual, 8)
    })
    
    # El ciclo de Horner (División sintética)
    for i in range(1, len(coeffs)):
        grado_actual = n - i
        a_i = float(coeffs[i])
        operacion = b_actual * x0
        b_nuevo = a_i + operacion
        
        resultados.append({
            "grado": grado_actual,
            "a": round(a_i, 8),
            "operacion": round(operacion, 8),
            "b": round(b_nuevo, 8)
        })
        b_actual = b_nuevo

    # === GENERAR LA GRÁFICA ===
    f_numpy = sp.lambdify(variable_dinamica, f_simbolica, 'numpy') 
    x0_float = float(x0)
    
    margen = abs(x0_float) * 0.5 if x0_float != 0 else 5
    x_min = x0_float - margen - 2
    x_max = x0_float + margen + 2
    
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = f_numpy(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'P({variable_dinamica})', color='#fd7e14', linewidth=2) 
    plt.axhline(0, color='black', linewidth=1) 
    
    # Dibujar el punto evaluado
    plt.axvline(x0_float, color='gray', linestyle=':', label=f'{variable_dinamica}0 = {x0_float}')
    plt.plot(x0_float, b_actual, 'bo', markersize=8, label=f'P({x0_float}) = {round(b_actual, 4)}') 
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "horner", 
        "resultados": resultados, 
        "raiz": round(b_actual, 8), 
        "convergencia": f"Evaluación exitosa usando la variable '{variable_dinamica}'.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 8: HORNER-NEWTON (BIRGE-VIETA)
# ==========================================
def metodo_horner_newton(latex_str, x0, tol, max_iter):
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Polinomio vacío",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe la función en la pizarra virtual antes de calcular."
        }
    
    try:
        # === LIMPIEZA EXTREMA DEL LATEX ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Parche anti-mayúsculas
        
        # Traducción matemática
        f_simbolica = parse_latex(latex_limpio)
        f_simbolica = f_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- DETECCIÓN DE VARIABLE DINÁMICA ---
        simbolos_usados = list(f_simbolica.free_symbols)
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "Horner-Newton solo funciona con polinomios de 1 sola variable."
            }
        
        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]

        # Validación de seguridad: ¡Solo polinomios! (Usando la variable detectada)
        if not f_simbolica.is_polynomial(variable_dinamica):
            return {
                "error": True,
                "titulo": "🛑 Función No Polinomial",
                "mensaje": f"El Método de Horner-Newton utiliza doble división sintética y SOLO funciona con polinomios en '{variable_dinamica}'.",
                "consejo": f"Ingresa una función polinomial válida (Ej: {variable_dinamica}^3 - 2{variable_dinamica}^2 - 5). No uses fracciones con incógnitas abajo, senos o logaritmos."
            }
            
        polinomio = sp.Poly(f_simbolica, variable_dinamica)
        coeffs = polinomio.all_coeffs() 
        f_numpy = sp.lambdify(variable_dinamica, f_simbolica, 'numpy') 
        
    except Exception as err:
        print(f"==============\n🛑 FALLA AL PARSEAR:\nTexto original: '{latex_str}'\nTexto limpio: '{latex_limpio}'\n==============") 
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar el polinomio. Detalle: {str(err)}",
            "consejo": "Revisa la consola (terminal) de VS Code para ver qué texto exacto envió MathLive."
        }

    # Función auxiliar para hacer la división sintética rápida
    def division_sintetica(coeficientes, valor_x):
        b = [float(coeficientes[0])]
        for j in range(1, len(coeficientes)):
            b.append(float(coeficientes[j]) + b[-1] * valor_x)
        return b[-1], b[:-1] # Retorna (Residuo, Coeficientes del Cociente)

    resultados = []
    xi = float(x0)

    for i in range(1, max_iter + 1):
        # Horner 1: Sacamos P(xi) y el polinomio cociente Q(x)
        pxi, q_coeffs = division_sintetica(coeffs, xi)
        
        # Horner 2: Evaluamos el cociente Q(x) para sacar la derivada P'(xi)
        dpxi, _ = division_sintetica(q_coeffs, xi)
        
        if dpxi == 0:
            return {
                "error": True,
                "titulo": "⚠️ Derivada Cero (Línea Horizontal)",
                "mensaje": f"En la iteración {i}, la segunda división sintética (derivada) dio 0.",
                "consejo": "El método falla porque genera división por cero. Intenta con un valor inicial diferente."
            }

        # Fórmula de Newton usando los residuos de Horner
        x_siguiente = xi - (pxi / dpxi)
        
        ea = abs((x_siguiente - xi) / x_siguiente) * 100 if x_siguiente != 0 else 100

        resultados.append({
            "iteracion": i,
            "xi": round(xi, 8),
            "pxi": round(pxi, 8),
            "dpxi": round(dpxi, 8),
            "x_siguiente": round(x_siguiente, 8),
            "ea": round(ea, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            xi = x_siguiente
            break
            
        xi = x_siguiente

    # === GENERAR LA GRÁFICA ===
    margen = abs(xi - float(x0)) * 0.5 if abs(xi - float(x0)) > 0 else 2
    x_min = min(float(x0), xi) - margen
    x_max = max(float(x0), xi) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = f_numpy(x_vals)

    altura_maxima = max(abs(f_numpy(x_min)), abs(f_numpy(x_max))) * 3
    y_vals = np.clip(y_vals, -altura_maxima, altura_maxima)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'P({variable_dinamica})', color='#20c997', linewidth=2) # Color Teal
    plt.axhline(0, color='black', linewidth=1) 
    
    plt.axvline(float(x0), color='orange', linestyle='--', label=f'{variable_dinamica}0 inicial')
    plt.plot(xi, 0, 'go', markersize=8, label=f'Raíz ({round(xi, 8)})') 
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "horner_newton", 
        "resultados": resultados, 
        "raiz": round(xi, 8),
        "convergencia": f"Cuadrática usando '{variable_dinamica}' y Doble División Sintética.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 9: MÉTODO DE MÜLLER
# ==========================================
def metodo_muller(latex_str, x0, x1, x2, tol, max_iter):
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe la función en la pizarra virtual antes de calcular."
        }
    
    try:
        # === LIMPIEZA EXTREMA DEL LATEX ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Parche anti-mayúsculas
        
        # Traducción matemática
        f_simbolica = parse_latex(latex_limpio)
        f_simbolica = f_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- DETECCIÓN DE VARIABLE DINÁMICA ---
        simbolos_usados = list(f_simbolica.free_symbols)
        simbolos_usados = [s for s in simbolos_usados if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "El Método de Müller solo soporta funciones de 1 variable."
            }
        
        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]

        # Pasamos a NumPy usando la variable detectada
        f = sp.lambdify(variable_dinamica, f_simbolica, 'numpy') 

        # Prueba de fuego con los 3 puntos iniciales
        complex(f(x0))
        complex(f(x1))
        complex(f(x2))

    except Exception as err:
        print(f"🛑 Error parseando Müller: {latex_str}") 
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar la función. Detalle: {str(err)}",
            "consejo": "Revisa que la ecuación esté completa en la pizarra virtual."
        }

    resultados = []
    
    # Usamos números complejos internamente por si la raíz lo requiere
    h0 = x1 - x0
    h1 = x2 - x1
    
    # Pequeño seguro contra división por cero en el arranque
    if h0 == 0 or h1 == 0:
        return {
            "error": True,
            "titulo": "🛑 Puntos Iniciales Inválidos",
            "mensaje": "Los puntos x0, x1 y x2 no pueden ser iguales.",
            "consejo": "Elige tres puntos iniciales distintos para trazar la parábola."
        }

    d0 = (f(x1) - f(x0)) / h0
    d1 = (f(x2) - f(x1)) / h1
    a = (d1 - d0) / (h1 + h0)

    for i in range(1, max_iter + 1):
        b = d1 + h1 * a
        c = f(x2)
        
        # Discriminante
        discriminante = np.lib.scimath.sqrt(b**2 - 4*a*c)
        
        # Elegimos el signo que maximice el denominador
        if abs(b + discriminante) > abs(b - discriminante):
            denominador = b + discriminante
        else:
            denominador = b - discriminante
            
        if denominador == 0:
            break
            
        dx = -2 * c / denominador
        x3 = x2 + dx
        
        # Error aproximado
        ea = abs(dx / x3) * 100 if x3 != 0 else 100
        
        resultados.append({
            "iteracion": i,
            "x0": round(complex(x0).real, 8),
            "x1": round(complex(x1).real, 8),
            "x2": round(complex(x2).real, 8),
            "xr": round(complex(x3).real, 8),
            "fxr": round(complex(f(x3)).real, 8),
            "ea": round(complex(ea).real, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            break
            
        # Actualizamos puntos para la siguiente iteración
        x0, x1, x2 = x1, x2, x3
        h0 = x1 - x0
        h1 = x2 - x1
        
        if h0 == 0 or h1 == 0 or (h1 + h0) == 0:
            break

        d0 = (f(x1) - f(x0)) / h0
        d1 = (f(x2) - f(x1)) / h1
        a = (d1 - d0) / (h1 + h0)

    # === GENERAR LA GRÁFICA ===
    margen = 2
    x_min = min(x0.real, x1.real, x2.real) - margen
    x_max = max(x0.real, x1.real, x2.real) + margen
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f({variable_dinamica})', color='#e83e8c', linewidth=2) # Rosado para Müller
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(x3.real, 0, 'ro', markersize=8, label=f'Raíz ({round(x3.real, 4)})')
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "muller",
        "resultados": resultados,
        "raiz": round(x3.real, 8),
        "convergencia": f"Superlineal. Puede hallar raíces complejas. Variable usada: '{variable_dinamica}'.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 10: MÉTODO DE BAIRSTOW (CORREGIDO)
# ==========================================
def metodo_bairstow(latex_str, r, s, tol, max_iter):
    # --- BLOQUEO DE SEGURIDAD ---
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True,
            "titulo": "🛑 Polinomio vacío",
            "mensaje": "No se recibió ninguna ecuación.",
            "consejo": "Por favor, escribe el polinomio en la pizarra virtual antes de calcular."
        }
    
    try:
        # === LIMPIEZA EXTREMA DEL LATEX ===
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e')
        latex_limpio = latex_limpio.replace(r'\exponentialE', 'e')
        latex_limpio = latex_limpio.replace(r'\cdot', '*') 
        latex_limpio = latex_limpio.lower() # Parche anti-mayúsculas
        
        # Traducción matemática
        f_simbolica = parse_latex(latex_limpio)
        f_simbolica = f_simbolica.subs(sp.Symbol('e'), sp.E)
        
        # --- DETECCIÓN DE VARIABLE DINÁMICA ---
        simbolos_usados = list(f_simbolica.free_symbols)
        simbolos_usados = [sym for sym in simbolos_usados if str(sym) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {
                "error": True,
                "titulo": "🛑 Demasiadas Variables",
                "mensaje": f"Detectamos estas variables: {simbolos_usados}.",
                "consejo": "Bairstow solo funciona con polinomios de 1 sola variable."
            }
        
        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]

        # Validación de seguridad: ¡Solo polinomios usando la variable dinámica!
        if not f_simbolica.is_polynomial(variable_dinamica):
            return {
                "error": True,
                "titulo": "🛑 No es un Polinomio",
                "mensaje": f"El método de Bairstow solo funciona con funciones polinomiales puras en '{variable_dinamica}'.",
                "consejo": f"Ingresa una función polinomial válida (Ej: {variable_dinamica}^4 - 3{variable_dinamica}^3 + 2{variable_dinamica} - 1)."
            }
            
        polinomio = sp.Poly(f_simbolica, variable_dinamica)
        a = [float(c) for c in polinomio.all_coeffs()]
        a.reverse() 
        n = len(a) - 1
        
    except Exception as err:
        print(f"🛑 Error parseando Bairstow: {latex_str}") 
        return {
            "error": True, 
            "titulo": "🛑 Error de Sintaxis Matemática", 
            "mensaje": f"No se pudo evaluar el polinomio. Detalle: {str(err)}",
            "consejo": "Revisa que la ecuación esté completa en la pizarra virtual."
        }

    resultados = []
    current_r, current_s = float(r), float(s)

    for i in range(1, max_iter + 1):
        b = [0.0] * (n + 1)
        c = [0.0] * (n + 1)

        b[n] = a[n]
        b[n-1] = a[n-1] + current_r * b[n]
        for j in range(n-2, -1, -1):
            b[j] = a[j] + current_r * b[j+1] + current_s * b[j+2]

        c[n] = b[n]
        c[n-1] = b[n-1] + current_r * c[n]
        
        # 🐛 EL BUG ESTABA AQUÍ: Cambiamos el 1 por un 0. ¡Ahora c[1] sí nace!
        for j in range(n-2, 0, -1): 
            c[j] = b[j] + current_r * c[j+1] + current_s * c[j+2]

        det = c[2]*c[2] - c[3]*c[1]
        
        # Seguro anti-estancamiento
        if det == 0:
            current_r += 0.01
            current_s += 0.01
            continue 
        
        dr = (-b[1]*c[2] - (-b[0]*c[3])) / det
        ds = (c[2]*(-b[0]) - c[1]*(-b[1])) / det

        current_r += dr
        current_s += ds

        ea_r = abs(dr / current_r) * 100 if current_r != 0 else 100
        ea_s = abs(ds / current_s) * 100 if current_s != 0 else 100
        ea_max = max(ea_r, ea_s)

        resultados.append({
            "iteracion": i,
            "r": round(current_r, 8),
            "s": round(current_s, 8),
            "ea": round(ea_max, 8) if i > 1 else "---"
        })

        if ea_max < tol:
            break

    disc = current_r**2 + 4 * current_s
    
    # Armamos la respuesta de la raíz usando la variable que el usuario ingresó
    var = str(variable_dinamica)
    
    if disc >= 0:
        x1 = (current_r + math.sqrt(disc)) / 2
        x2 = (current_r - math.sqrt(disc)) / 2
        raiz_str = f"{var}1: {round(x1, 8)}, {var}2: {round(x2, 8)}"
    else:
        real = current_r / 2
        imag = math.sqrt(-disc) / 2
        raiz_str = f"{var}1,2: {round(real, 8)} ± {round(imag, 8)}i"

    return {
        "tipo": "bairstow",
        "resultados": resultados,
        "raiz": raiz_str,
        "convergencia": f"Factor hallado: {var}² - ({round(current_r, 4)}){var} - ({round(current_s, 4)})",
        "grafica": None
    }

# Rutas denavegación
@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/biseccion', methods=['GET', 'POST'])
def biseccion():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        xl = float(request.form['xl'])
        xu = float(request.form['xu'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_biseccion(funcion, xl, xu, tol, max_iter)
        
    return render_template('biseccion.html', datos=datos)

@app.route('/falsa_posicion', methods=['GET', 'POST'])
def falsa_posicion():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        xl = float(request.form['xl'])
        xu = float(request.form['xu'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_falsa_posicion(funcion, xl, xu, tol, max_iter)
        
    return render_template('falsa_posicion.html', datos=datos)

@app.route('/newton', methods=['GET', 'POST'])
def newton():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_newton_raphson(funcion, x0, tol, max_iter)
        
    return render_template('newton.html', datos=datos)

@app.route('/secante', methods=['GET', 'POST'])
def secante():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        x0 = float(request.form['x0'])
        x1 = float(request.form['x1'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_secante(funcion, x0, x1, tol, max_iter)
        
    return render_template('secante.html', datos=datos)

@app.route('/taylor', methods=['GET', 'POST'])
def taylor():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        x0 = float(request.form['x0'])
        x_eval = float(request.form['x_eval'])
        orden = int(request.form['orden'])
        
        datos = serie_taylor(funcion, x0, x_eval, orden)
        
    return render_template('taylor.html', datos=datos)

@app.route('/punto_fijo', methods=['GET', 'POST'])
def punto_fijo():
    datos = None
    if request.method == 'POST':
        latex_gx_str = request.form['ecuacion_latex'] 
        
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_punto_fijo(latex_gx_str, x0, tol, max_iter)
        
    return render_template('punto_fijo.html', datos=datos)

@app.route('/horner', methods=['GET', 'POST'])
def horner():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        x0 = float(request.form['x0'])
        
        datos = metodo_horner(funcion, x0)
        
    return render_template('horner.html', datos=datos)

@app.route('/horner_newton', methods=['GET', 'POST'])
def horner_newton():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_horner_newton(funcion, x0, tol, max_iter)
        
    return render_template('horner_newton.html', datos=datos)

@app.route('/muller', methods=['GET', 'POST'])
def muller():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        x0 = float(request.form['x0'])
        x1 = float(request.form['x1'])
        x2 = float(request.form['x2'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        datos = metodo_muller(funcion, x0, x1, x2, tol, max_iter)
    return render_template('muller.html', datos=datos)

@app.route('/bairstow', methods=['GET', 'POST'])
def bairstow():
    datos = None
    if request.method == 'POST':
        funcion = request.form['ecuacion_latex']
        r = float(request.form['r'])
        s = float(request.form['s'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        datos = metodo_bairstow(funcion, r, s, tol, max_iter)
    return render_template('bairstow.html', datos=datos)

if __name__ == '__main__':
    app.run(debug=True)