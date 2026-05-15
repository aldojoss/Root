from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg') # Esto evita que la gráfica intente abrirse en una ventana de Windows y bloquee el servidor
import matplotlib.pyplot as plt
import io
import base64
import math
import re
from sympy.parsing.latex import parse_latex

app = Flask(__name__)

# ==========================================
# MÉTODO 1: BISECCIÓN (BLINDADO)
# ==========================================
def metodo_biseccion(latex_str, xl, xu, tol, max_iter):
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True, "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.", "consejo": "Usa la pizarra virtual para escribir tu f(x)."
        }
    
    try:
        # Limpieza y parseo de LaTeX
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e').replace(r'\exponentialE', 'e').replace(r'\cdot', '*').lower()
        funcion_simbolica = parse_latex(latex_limpio).subs(sp.Symbol('e'), sp.E)
        
        simbolos_usados = [s for s in funcion_simbolica.free_symbols if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1: return {"error": True, "titulo": "🛑 Demasiadas Variables", "mensaje": f"Detectamos: {simbolos_usados}.", "consejo": "Bisección solo soporta 1 variable."}
        if len(simbolos_usados) == 0: return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]
        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy')

        
        #aqui validamos si se puede evaluar los limites en la funcion,por culquier error
        try:
            fxl = float(f(xl))
            fxu = float(f(xu))
        except (ValueError, TypeError, ZeroDivisionError):
            return {
                "error": True, "titulo": "🛑 Error de Dominio Matemático",
                "mensaje": f"No se puede evaluar la función en el intervalo [{xl}, {xu}].", 
                "consejo": "Asegúrate de no estar dividiendo por cero o sacando raíces cuadradas/logaritmos de números negativos en esos límites."
            }

    except Exception as err:
        return {"error": True, "titulo": "🛑 Error de Sintaxis", "mensaje": str(err), "consejo": "Revisa la escritura de tu fórmula."}

    # aqui aplicamos el teorema de bolzano, las condiciones,practicamente
    # se revisa si la funcion en ese intervalo tiene raizo cambia de signo
    if fxl * fxu > 0:
        return {
            "error": True, "titulo": "⚠️ Intervalo Inválido (Sin cambio de signo)",
            "mensaje": f"Evaluamos tus límites y obtuvimos f({xl}) = {round(fxl, 5)} y f({xu}) = {round(fxu, 5)}. Ambos tienen el mismo signo.",
            "consejo": "Para que la bisección funcione, la curva debe cruzar el eje X. Un límite debe ser positivo y el otro negativo."
        }
    #pero validaos esto por si encontramos una raiz
    elif fxl * fxu == 0:
        raiz_exacta = xl if fxl == 0 else xu
        return {
            "error": True, "titulo": "🎯 ¡Raíz encontrada al instante!",
            "mensaje": f"Uno de tus límites ya es la raíz exacta: {raiz_exacta}", 
            "consejo": "Intenta con otro intervalo si estás buscando una raíz diferente en la curva."
        }

    resultados = []
    xr_anterior = 0
    xl_original, xu_original = xl, xu

    # === CICLO ITERATIVO ===
    for i in range(1, max_iter + 1):
        xr = (xl + xu) / 2
        
        try:
            fxr = float(f(xr))
        except:
            return {"error": True, "titulo": "🛑 Discontinuidad detectada", "mensaje": f"La función falló al evaluar en el punto medio xr = {xr}.", "consejo": "Revisa que tu función sea continua en este intervalo."}
        
        ea = abs((xr - xr_anterior) / xr) * 100 if (xr != 0 and i > 1) else 100
        
        resultados.append({
            "iteracion": i, "xl": round(xl, 8), "xu": round(xu, 8), "xr": round(xr, 8),
            "fxl": round(fxl, 8), "fxr": round(fxr, 8), "ea": round(ea, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            break

        # Reemplazo de límites
        if fxl * fxr < 0:
            xu = xr
        elif fxl * fxr > 0:
            xl = xr
            fxl = fxr # Actualizamos fxl para la siguiente iteración
        else:
            break # fxr es exactamente 0
            
        xr_anterior = xr

    # === GENERAR LA GRÁFICA ===
    margen = (xu_original - xl_original) * 0.5
    if margen == 0: margen = 2
    x_vals = np.linspace(xl_original - margen, xu_original + margen, 200)
    
    try:
        y_vals = f(x_vals)
        if isinstance(y_vals, (int, float)): y_vals = np.full_like(x_vals, y_vals)
    except:
        y_vals = np.zeros_like(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f({variable_dinamica})', color='#0d6efd', linewidth=2)
    plt.axhline(0, color='black', linewidth=1) 
    
    plt.axvline(xl_original, color='orange', linestyle='--', label='xl inicial')
    plt.axvline(xu_original, color='purple', linestyle='--', label='xu inicial')
    plt.plot(xr, 0, 'ro', markersize=8, label=f'Raíz ({round(xr, 8)})')
    
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
        "raiz": round(xr, 8),
        "convergencia": "Lineal O(n) - El error se reduce a la mitad en cada paso.",
        "grafica": grafica_url
    }

# ==========================================
# MÉTODO 2: REGLA FALSA (FALSA POSICIÓN)
# ==========================================
def metodo_falsa_posicion(latex_str, xl, xu, tol, max_iter):
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True, "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.", "consejo": "Escribe tu f(x) en la pizarra virtual."
        }
    
    try:
        # 1. Limpieza y Traducción LaTeX a SymPy
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e').replace(r'\exponentialE', 'e').replace(r'\cdot', '*').lower()
        funcion_simbolica = parse_latex(latex_limpio).subs(sp.Symbol('e'), sp.E)
        
        # 2. Detección de Variable Dinámica
        simbolos_usados = [s for s in funcion_simbolica.free_symbols if str(s) not in ['e', 'pi']]
        if len(simbolos_usados) > 1: return {"error": True, "titulo": "🛑 Demasiadas Variables", "mensaje": f"Detectamos: {simbolos_usados}."}
        if len(simbolos_usados) == 0: return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene incógnita."}

        variable_dinamica = simbolos_usados[0]
        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy')

        # 3. Prueba de fuego en los límites
        xl_original, xu_original = xl, xu
        fxl = float(f(xl))
        fxu = float(f(xu))

    except Exception as err:
        return {"error": True, "titulo": "🛑 Error Matemático", "mensaje": str(err)}

    # === Validación de Bolzano ===
    if fxl * fxu >= 0:
        return {
            "error": True, "titulo": "⚠️ Intervalo Inválido",
            "mensaje": f"f({xl}) = {round(fxl, 5)} y f({xu}) = {round(fxu, 5)}. Mismo signo.",
            "consejo": "Regla Falsa requiere que la función cruce el eje X entre xl y xu."
        }

    resultados = []
    xr_anterior = 0

    # === Ciclo de Regla Falsa ===
    for i in range(1, max_iter + 1):
        fxl = f(xl)
        fxu = f(xu)
        
        if fxl - fxu == 0: break

        # Fórmula de Regla Falsa (Intersección de la secante)
        xr = xu - (fxu * (xl - xu)) / (fxl - fxu)
        fxr = f(xr)

        ea = abs((xr - xr_anterior) / xr) * 100 if i > 1 else 100

        resultados.append({
            "iteracion": i, "xl": round(xl, 8), "xu": round(xu, 8), "xr": round(xr, 8),
            "fxl": round(fxl, 8), "fxr": round(fxr, 8), "ea": round(ea, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol: break

        # Reemplazo de límites
        if fxl * fxr < 0:
            xu = xr
        else:
            xl = xr
        
        xr_anterior = xr

    # === Gráfica ===
    margen = (xu_original - xl_original) * 0.5
    x_vals = np.linspace(xl_original - margen - 1, xu_original + margen + 1, 200)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f({variable_dinamica})', color='#198754', linewidth=2) 
    plt.axhline(0, color='black', linewidth=1) 
    plt.axvline(xl_original, color='orange', linestyle='--', label='xl inicial')
    plt.axvline(xu_original, color='purple', linestyle='--', label='xu inicial')
    plt.plot(xr, 0, 'ro', markersize=8, label=f'Raíz ({round(xr, 8)})')
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend(); plt.tight_layout()

    img = io.BytesIO(); plt.savefig(img, format='png', transparent=True); img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8'); plt.close()

    return {
        "resultados": resultados, "raiz": round(xr, 8),
        "convergencia": f"Lineal - Variable: '{variable_dinamica}'.", "grafica": grafica_url
    }
def metodo_newton_raphson(latex_str, x0, tol, max_iter):
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True, "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.", "consejo": "Escribe tu f(x) en la pizarra virtual."
        }

    try:
        # 1. Limpieza y traducción a SymPy
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e').replace(r'\exponentialE', 'e').replace(r'\cdot', '*').lower()
        funcion_simbolica = parse_latex(latex_limpio).subs(sp.Symbol('e'), sp.E)

        # 2. Detección de variable dinámica
        simbolos_usados = [s for s in funcion_simbolica.free_symbols if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1:
            return {"error": True, "titulo": "🛑 Demasiadas Variables", "mensaje": f"Detectamos: {simbolos_usados}."}
        if len(simbolos_usados) == 0:
            return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene incógnita."}

        variable_dinamica = simbolos_usados[0]

        # 3. Derivada analítica automática
        derivada_simbolica = sp.diff(funcion_simbolica, variable_dinamica)

        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy')
        df = sp.lambdify(variable_dinamica, derivada_simbolica, 'numpy')

        # Prueba de fuego numérica
        float(f(x0))
        float(df(x0))

    except Exception as err:
        return {"error": True, "titulo": "🛑 Error Matemático", "mensaje": str(err)}

    resultados = []
    xi = float(x0)

    # 4. Ciclo de Newton-Raphson
    for i in range(1, max_iter + 1):
        try:
            fxi = float(f(xi))
            dfxi = float(df(xi))
        except:
            return {"error": True, "titulo": "🚀 Divergencia", "mensaje": "Los números se volvieron demasiado grandes o complejos."}

        if dfxi == 0:
            return {"error": True, "titulo": "⚠️ Derivada Cero", "mensaje": "La pendiente es horizontal. El método no puede avanzar.", "consejo": "Prueba con otro x0."}

        # Fórmula de Newton-Raphson: xi+1 = xi - f(xi)/f'(xi)
        x_siguiente = xi - (fxi / dfxi)
        ea = abs((x_siguiente - xi) / x_siguiente) * 100 if x_siguiente != 0 else 100

        resultados.append({
            "iteracion": i, "xi": round(xi, 8), "fxi": round(fxi, 8),
            "dfxi": round(dfxi, 8), "x_siguiente": round(x_siguiente, 8),
            "ea": round(ea, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            xi = x_siguiente
            break
        xi = x_siguiente

    # 5. Gráfica
    margen = abs(xi - float(x0)) + 2
    x_vals = np.linspace(min(float(x0), xi) - margen, max(float(x0), xi) + margen, 200)
    try:
        y_vals = f(x_vals)
        if isinstance(y_vals, (int, float)): y_vals = np.full_like(x_vals, y_vals)
    except: y_vals = np.zeros_like(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f({variable_dinamica})', color='#dc3545', linewidth=2)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(float(x0), color='orange', linestyle='--', label='x0 inicial')
    plt.plot(xi, 0, 'go', markersize=8, label=f'Raíz ({round(xi, 8)})')
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend(); plt.tight_layout()

    img = io.BytesIO(); plt.savefig(img, format='png', transparent=True); img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8'); plt.close()

    return {
        "tipo": "abierto", "resultados": resultados, "raiz": round(xi, 8),
        "convergencia": f"Cuadrática O(n²) - Variable: '{variable_dinamica}'.", "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 4: SECANTE (ACTUALIZADO)
# ==========================================
def metodo_secante(latex_str, x0, x1, tol, max_iter):
    if not latex_str or latex_str.strip() == "":
        return {
            "error": True, "titulo": "🛑 Ecuación vacía",
            "mensaje": "No se recibió ninguna ecuación.", "consejo": "Usa la pizarra virtual."
        }

    try:
        # 1. Limpieza y traducción a SymPy
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e').replace(r'\exponentialE', 'e').replace(r'\cdot', '*').lower()
        funcion_simbolica = parse_latex(latex_limpio).subs(sp.Symbol('e'), sp.E)

        # 2. Detección de variable dinámica
        simbolos_usados = [s for s in funcion_simbolica.free_symbols if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1: return {"error": True, "titulo": "🛑 Demasiadas Variables", "mensaje": f"Detectamos: {simbolos_usados}."}
        if len(simbolos_usados) == 0: return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene incógnita."}

        variable_dinamica = simbolos_usados[0]
        f = sp.lambdify(variable_dinamica, funcion_simbolica, 'numpy')

        # Evaluación inicial de los dos puntos
        fx0 = float(f(x0))
        fx1 = float(f(x1))

    except Exception as err:
        return {"error": True, "titulo": "🛑 Error Matemático", "mensaje": str(err)}

    resultados = []
    x_previo = float(x0)
    x_actual = float(x1)

    # 3. Ciclo de la Secante
    for i in range(1, max_iter + 1):
        try:
            fx_previo = float(f(x_previo))
            fx_actual = float(f(x_actual))
        except:
            return {"error": True, "titulo": "🚀 Divergencia", "mensaje": "La función generó valores demasiado grandes o complejos."}

        # Evitar división por cero si f(x_previo) y f(x_actual) son iguales (Línea horizontal)
        if fx_previo - fx_actual == 0:
            return {"error": True, "titulo": "⚠️ División por Cero", "mensaje": "La recta secante se volvió horizontal y no cruzará el eje X.", "consejo": "Intenta con otros valores iniciales."}

        # Fórmula de la Secante
        x_siguiente = x_actual - (fx_actual * (x_previo - x_actual)) / (fx_previo - fx_actual)
        
        ea = abs((x_siguiente - x_actual) / x_siguiente) * 100 if x_siguiente != 0 else 100

        resultados.append({
            "iteracion": i, "x_previo": round(x_previo, 8), "x_actual": round(x_actual, 8),
            "fx_actual": round(fx_actual, 8), "x_siguiente": round(x_siguiente, 8),
            "ea": round(ea, 8) if i > 1 else "---"
        })

        if ea < tol:
            x_actual = x_siguiente
            break
        
        # Desplazamiento para la siguiente iteración
        x_previo = x_actual
        x_actual = x_siguiente

    # 4. Gráfica
    margen = abs(x_actual - float(x0)) + 2
    x_vals = np.linspace(min(float(x0), float(x1), x_actual) - margen, max(float(x0), float(x1), x_actual) + margen, 200)
    
    try:
        y_vals = f(x_vals)
        if isinstance(y_vals, (int, float)): y_vals = np.full_like(x_vals, y_vals)
        y_vals = np.clip(y_vals, -100, 100) # Evitar que la gráfica se deforme
    except: y_vals = np.zeros_like(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f({variable_dinamica})', color='#0dcaf0', linewidth=2)
    plt.axhline(0, color='black', linewidth=1)
    
    # Dibujamos los dos puntos iniciales
    plt.axvline(float(x0), color='orange', linestyle=':', label='x0 inicial')
    plt.axvline(float(x1), color='purple', linestyle=':', label='x1 inicial')
    plt.plot(x_actual, 0, 'go', markersize=8, label=f'Raíz ({round(x_actual, 8)})')
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend(); plt.tight_layout()

    img = io.BytesIO(); plt.savefig(img, format='png', transparent=True); img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8'); plt.close()

    return {
        "tipo": "secante", "resultados": resultados, "raiz": round(x_actual, 8),
        "convergencia": f"Superlineal (Aprox. 1.618) - Variable: '{variable_dinamica}'.", "grafica": grafica_url
    }
    

# ==========================================
# MÉTODO 5: SERIE DE TAYLOR 
# ==========================================
def metodo_taylor(latex_str, x0, x_eval, n_terminos):
    if not latex_str or latex_str.strip() == "":
        return {"error": True, "titulo": "🛑 Ecuación vacía", "mensaje": "Escribe tu f(x) en la pizarra."}

    try:
        # Limpieza y Traducción a SymPy
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e').replace(r'\exponentialE', 'e').replace(r'\cdot', '*').lower()
        f_simbolica = parse_latex(latex_limpio).subs(sp.Symbol('e'), sp.E)

        # Detectar variable
        simbolos_usados = [s for s in f_simbolica.free_symbols if str(s) not in ['e', 'pi']]
        if len(simbolos_usados) > 1: return {"error": True, "titulo": "🛑 Demasiadas Variables", "mensaje": f"Detectamos: {simbolos_usados}."}
        x = simbolos_usados[0] if len(simbolos_usados) == 1 else sp.Symbol('x')
        
        # Valor verdadero
        f_lambdify = sp.lambdify(x, f_simbolica, 'numpy')
        valor_verdadero = float(f_lambdify(x_eval))

    except Exception as err:
        return {"error": True, "titulo": "🛑 Error Matemático", "mensaje": str(err)}

    resultados = []
    aprox_actual = 0
    polinomio_taylor = 0

    # Ciclo de Taylor
    for i in range(n_terminos):
        if i == 0:
            derivada = f_simbolica
        else:
            derivada = sp.diff(f_simbolica, x, i)
        
        try:
            derivada_evaluada = float(derivada.subs(x, x0))
        except:
            return {"error": True, "titulo": "⚠️ Discontinuidad", "mensaje": f"No se puede evaluar la derivada de orden {i} en el centro."}
        
        # Fórmula de Taylor
        termino_valor = (derivada_evaluada / math.factorial(i)) * ((x_eval - x0) ** i)
        termino_simbolico = (derivada.subs(x, x0) / math.factorial(i)) * ((x - x0) ** i)
        
        aprox_actual += termino_valor
        polinomio_taylor += termino_simbolico
        
        et = abs((valor_verdadero - aprox_actual) / valor_verdadero) * 100 if valor_verdadero != 0 else abs(valor_verdadero - aprox_actual)
        
        # Estética visual de la derivada para la tabla
        derivada_bonita = str(derivada).replace('**', '^').replace('*', '·').replace('sqrt', '√')

        resultados.append({
            "orden": i,
            "derivada": derivada_bonita,
            "derivada_evaluada": round(derivada_evaluada, 8),
            "termino_calculado": round(termino_valor, 8),
            "aproximacion": round(aprox_actual, 8),
            "et": round(et, 8) if i > 0 else "---"
        })

    # Gráfica
    margen = abs(x_eval - x0) + 2
    x_vals = np.linspace(min(x0, x_eval) - margen, max(x0, x_eval) + margen, 200)
    
    try:
        y_vals_f = f_lambdify(x_vals)
        if isinstance(y_vals_f, (int, float)): y_vals_f = np.full_like(x_vals, y_vals_f)
        
        p_lambdify = sp.lambdify(x, polinomio_taylor, 'numpy')
        y_vals_p = p_lambdify(x_vals)
        if isinstance(y_vals_p, (int, float)): y_vals_p = np.full_like(x_vals, y_vals_p)
        
        y_min, y_max = np.min(y_vals_f) - 5, np.max(y_vals_f) + 5
        y_vals_p = np.clip(y_vals_p, y_min - 10, y_max + 10)
    except: 
        y_vals_f = np.zeros_like(x_vals)
        y_vals_p = np.zeros_like(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals_f, label=f'Original f({x})', color='#0d6efd', linewidth=2)
    plt.plot(x_vals, y_vals_p, label=f'Taylor (Orden {n_terminos-1})', color='#ffc107', linestyle='--', linewidth=2)
    plt.axvline(x0, color='gray', linestyle=':', label=f'Centro x0={x0}')
    plt.plot(x_eval, valor_verdadero, 'bo', markersize=6, label='Valor Real')
    plt.plot(x_eval, aprox_actual, 'yo', markersize=6, label='Aproximación')
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend(); plt.tight_layout()

    img = io.BytesIO(); plt.savefig(img, format='png', transparent=True); img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8'); plt.close()

    polinomio_str = str(polinomio_taylor).replace('**', '^').replace('*', '·').replace('sqrt', '√')

    return {
        "tipo": "taylor", 
        "resultados": resultados, 
        "valor_verdadero": round(valor_verdadero, 8),
        "aprox_final": round(aprox_actual, 8),
        "polinomio_final": polinomio_str,
        "grafica": grafica_url
    }


    
# ==========================================
# MÉTODO 6: PUNTO FIJO (CON DESPEJE AUTOMÁTICO)
# ==========================================
def metodo_punto_fijo(latex_fx_str, x0, tol, max_iter):
    if not latex_fx_str or latex_fx_str.strip() == "":
        return {
            "error": True, "titulo": "🛑 Ecuación f(x) vacía",
            "mensaje": "No se recibió ninguna ecuación.", "consejo": "Escribe tu función f(x) original."
        }
    
    try:
        latex_limpio = latex_fx_str.replace(r'\mathrm{e}', 'e').replace(r'\exponentialE', 'e').replace(r'\cdot', '*').lower()
        f_simbolica = parse_latex(latex_limpio).subs(sp.Symbol('e'), sp.E)
        
        simbolos_usados = [s for s in f_simbolica.free_symbols if str(s) not in ['e', 'pi']]

        if len(simbolos_usados) > 1: return {"error": True, "titulo": "🛑 Demasiadas Variables", "mensaje": f"Detectamos: {simbolos_usados}.", "consejo": "Punto Fijo solo soporta 1 variable."}
        if len(simbolos_usados) == 0: return {"error": True, "titulo": "🛑 Sin Variable", "mensaje": "La ecuación no tiene ninguna incógnita."}

        variable_dinamica = simbolos_usados[0]
        x = variable_dinamica
        
        # === GENERADOR AUTOMÁTICO DE DESPEJES g(x) ===
        posibles_g = [x + f_simbolica, x - f_simbolica]
        
        terminos = sp.Add.make_args(f_simbolica)
        for t in terminos:
            if t.has(x):
                resto = f_simbolica - t
                y = sp.Symbol('y_temp')
                try:
                    sub_despejes = sp.solve(t - y, x)
                    for sol in sub_despejes:
                        posibles_g.append(sol.subs(y, -resto))
                except: pass

        # === EVALUADOR DE CONVERGENCIA ===
        g_ganadora = None
        menor_derivada = float('inf')
        
        for g_expr in posibles_g:
            try:
                dg_expr = sp.diff(g_expr, x)
                dg_func = sp.lambdify(x, dg_expr, 'numpy')
                valor_derivada = abs(float(dg_func(x0)))
                
                # Elegimos la g(x) cuya derivada sea menor a 1
                if valor_derivada < 1 and valor_derivada < menor_derivada:
                    menor_derivada = valor_derivada
                    g_ganadora = g_expr
            except: continue

        if g_ganadora is None:
            return {
                "error": True, "titulo": "🛑 Divergencia Inevitable",
                "mensaje": f"Se generaron {len(posibles_g)} despejes posibles, pero NINGUNO converge en el punto x0 = {x0}.",
                "consejo": "El álgebra tiene un límite. Intenta con un valor inicial (x0) más cercano a la raíz real."
            }

        g = sp.lambdify(x, g_ganadora, 'numpy') 
        g_str_legible = str(g_ganadora).replace('**', '^')
        diagnostico = f"¡Despeje Automático Exitoso! g({x}) = {g_str_legible}. Convergencia garantizada: |g'({x0})| = {round(menor_derivada, 5)} < 1."

    except Exception as err:
        return {"error": True, "titulo": "🛑 Error Matemático", "mensaje": str(err), "consejo": "Revisa tu sintaxis."}

    # === CICLO ITERATIVO ===
    resultados = []
    xi = float(x0)
    diverge = False

    for i in range(1, max_iter + 1):
        try:
            gxi = float(g(xi))
        except: return {"error": True, "titulo": "🛑 Raíz Compleja", "mensaje": "El despeje topó con números imaginarios.", "consejo": "Cambia el x0."}
        
        if abs(gxi) > 1e6:
            diverge = True
            break

        ea = abs((gxi - xi) / gxi) * 100 if gxi != 0 else 100
        resultados.append({"iteracion": i, "xi": round(xi, 8), "gxi": round(gxi, 8), "ea": round(ea, 8) if i > 1 else "---"})

        if i > 1 and ea < tol:
            xi = gxi
            break
        xi = gxi

    if diverge: return {"error": True, "titulo": "🚀 Divergencia", "mensaje": "Divergencia numérica.", "consejo": "Prueba otra semilla."}

    # === GENERAR LA GRÁFICA ===
    margen = abs(xi - float(x0)) + 2
    x_min = min(float(x0), xi) - margen
    x_max = max(float(x0), xi) - margen if max(float(x0), xi) == min(float(x0), xi) else max(float(x0), xi) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals) 
    
    try:
        y_vals_g = g(x_vals)
        if isinstance(y_vals_g, (int, float)): y_vals_g = np.full_like(x_vals, y_vals_g)
    except: y_vals_g = np.zeros_like(x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals_g, label=f'g({variable_dinamica}) ganadora', color='#6f42c1', linewidth=2) 
    plt.plot(x_vals, x_vals, label=f'y = {variable_dinamica}', color='gray', linestyle='--', linewidth=1.5) 
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
        "tipo": "punto_fijo", "resultados": resultados, "raiz": round(xi, 8),
        "convergencia": diagnostico, "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 6: HORNER (EVALUACIÓN POLINOMIAL)
# ==========================================
def metodo_horner(latex_str, x0):
    if not latex_str or latex_str.strip() == "":
        return {"error": True, "titulo": "🛑 Polinomio vacío", "mensaje": "Escribe tu polinomio en la pizarra."}

    try:
        # 1. Limpieza y Traducción a SymPy
        latex_limpio = latex_str.replace(r'\mathrm{e}', 'e').replace(r'\exponentialE', 'e').replace(r'\cdot', '*').lower()
        f_simbolica = parse_latex(latex_limpio).subs(sp.Symbol('e'), sp.E)

        # 2. Detectar variable
        simbolos_usados = [s for s in f_simbolica.free_symbols if str(s) not in ['e', 'pi']]
        if len(simbolos_usados) > 1: return {"error": True, "titulo": "🛑 Demasiadas Variables", "mensaje": f"Detectamos: {simbolos_usados}."}
        x = simbolos_usados[0] if len(simbolos_usados) == 1 else sp.Symbol('x')

        # === 3. EXTRACCIÓN MÁGICA DE COEFICIENTES (CORREGIDA) ===
        # Primero expandimos (por si el usuario escribió algo como (x+1)^2)
        f_expandida = sp.expand(f_simbolica)
        
        # Usamos as_poly(x), que es a prueba de balas. Si no es polinomio, devuelve None.
        polinomio = f_expandida.as_poly(x)
        
        if polinomio is None:
            return {"error": True, "titulo": "⚠️ No es un Polinomio", "mensaje": "Horner solo acepta polinomios enteros (ej. x^3 - 2x + 1). No uses fracciones, raíces, ni senos."}

        # Extraemos los coeficientes reales y los convertimos a decimales
        coeficientes = [float(c) for c in polinomio.all_coeffs()]
        grado_maximo = len(coeficientes) - 1

    except Exception as err:
        return {"error": True, "titulo": "🛑 Error Matemático", "mensaje": str(err)}

    resultados = []
    
    # El primer coeficiente 'b' baja directo y es igual al primer 'a'
    b_actual = coeficientes[0] 

    resultados.append({
        "grado": grado_maximo,
        "a": round(b_actual, 8),
        "operacion": "---",
        "b": round(b_actual, 8)
    })

    # 4. Ciclo de Horner (La División Sintética)
    for i in range(1, len(coeficientes)):
        a_actual = coeficientes[i]
        
        # Multiplicamos el centro x0 por el b anterior
        operacion_val = b_actual * float(x0)
        
        # Sumamos hacia abajo
        b_nuevo = a_actual + operacion_val

        resultados.append({
            "grado": grado_maximo - i,
            "a": round(a_actual, 8),
            "operacion": round(operacion_val, 8),
            "b": round(b_nuevo, 8)
        })
        b_actual = b_nuevo

    # El último 'b' es el residuo, que equivale a evaluar P(x0)
    residuo = b_actual

    # 5. Generar la Gráfica Visual
    margen = 3
    x_vals = np.linspace(float(x0) - margen, float(x0) + margen, 200)
    try:
        f_lambdify = sp.lambdify(x, f_simbolica, 'numpy')
        y_vals = f_lambdify(x_vals)
        if isinstance(y_vals, (int, float)): y_vals = np.full_like(x_vals, y_vals)
    except:
        y_vals = np.zeros_like(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'P({x})', color='#fd7e14', linewidth=2)
    plt.axhline(0, color='black', linewidth=1)
    
    # Dibujamos el punto exacto evaluado
    plt.plot(float(x0), residuo, 'ro', markersize=8, label=f'P({x0}) = {round(residuo, 5)}')
    plt.axvline(float(x0), color='gray', linestyle=':')
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend(); plt.tight_layout()

    img = io.BytesIO(); plt.savefig(img, format='png', transparent=True); img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8'); plt.close()

    return {
        "tipo": "horner",
        "resultados": resultados,
        "raiz": round(residuo, 8), # Guardamos el residuo aquí para que el HTML lo muestre
        "convergencia": "Evaluación Polinomial mediante División Sintética.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 8: HORNER-NEWTON (BIRGE-VIETA)
# ==========================================
def metodo_horner_newton(funcion_str, x0, tol, max_iter):
    funcion_str = funcion_str.replace('^', '**').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        f_simbolica = sp.sympify(funcion_str)
        
        # Validación de seguridad: ¡Solo polinomios!
        if not f_simbolica.is_polynomial(x):
            return {
                "error": True,
                "titulo": "🛑 Función No Polinomial",
                "mensaje": "El Método de Horner-Newton utiliza doble división sintética y SOLO funciona con polinomios.",
                "consejo": "Ingresa una función polinomial válida (Ej: x**3 - 2*x**2 - 5). No uses fracciones, senos o logaritmos."
            }
            
        polinomio = sp.Poly(f_simbolica, x)
        coeffs = polinomio.all_coeffs() 
        f_numpy = sp.lambdify(x, f_simbolica, 'numpy') 
        
    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar el polinomio. Detalle: {str(err)}",
            "consejo": "Asegúrate de escribir el polinomio correctamente."
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
                "consejo": "El método falla porque genera división por cero. Intenta con un x0 diferente."
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
    margen = abs(xi - x0) * 0.5 if abs(xi - x0) > 0 else 2
    x_min = min(x0, xi) - margen
    x_max = max(x0, xi) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = f_numpy(x_vals)

    altura_maxima = max(abs(f_numpy(x_min)), abs(f_numpy(x_max))) * 3
    y_vals = np.clip(y_vals, -altura_maxima, altura_maxima)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'P(x)', color='#20c997', linewidth=2) # Color Teal
    plt.axhline(0, color='black', linewidth=1) 
    
    plt.axvline(x0, color='orange', linestyle='--', label='x0 inicial')
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
        "convergencia": "Cuadrática O(n²) usando Doble División Sintética.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 9: MÉTODO DE MÜLLER
# ==========================================
def metodo_muller(funcion_str, x0, x1, x2, tol, max_iter):
    funcion_str = funcion_str.replace('^', '**').replace('ln', 'log').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        diccionario_matematico = {'e': sp.E, 'pi': sp.pi}
        f_simbolica = sp.sympify(funcion_str, locals=diccionario_matematico)
        f = sp.lambdify(x, f_simbolica, 'numpy') 
    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis",
            "mensaje": f"No se pudo evaluar la función. Detalle: {str(err)}",
            "consejo": "Usa '*' para multiplicar y 'exp(x)' para la base e."
        }

    resultados = []
    # Usamos números complejos internamente por si la raíz lo requiere
    h0 = x1 - x0
    h1 = x2 - x1
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
        d0 = (f(x1) - f(x0)) / h0
        d1 = (f(x2) - f(x1)) / h1
        a = (d1 - d0) / (h1 + h0)

    # Gráfica
    margen = 2
    x_min = min(x0.real, x1.real, x2.real) - margen
    x_max = max(x0.real, x1.real, x2.real) + margen
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='f(x)', color='#e83e8c', linewidth=2) # Rosado para Müller
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
        "convergencia": "Superlineal (Casi cuadrática). Puede encontrar raíces complejas.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 10: MÉTODO DE BAIRSTOW (CORREGIDO)
# ==========================================
def metodo_bairstow(funcion_str, r, s, tol, max_iter):
    funcion_str = funcion_str.replace('^', '**').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        f_simbolica = sp.sympify(funcion_str)
        if not f_simbolica.is_polynomial(x):
            return {
                "error": True,
                "titulo": "🛑 No es un Polinomio",
                "mensaje": "Bairstow solo funciona con funciones polinomiales."
            }
        polinomio = sp.Poly(f_simbolica, x)
        a = [float(c) for c in polinomio.all_coeffs()]
        a.reverse() 
        n = len(a) - 1
    except Exception as err:
        return {"error": True, "titulo": "🛑 Error", "mensaje": str(err)}

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
    if disc >= 0:
        x1 = (current_r + math.sqrt(disc)) / 2
        x2 = (current_r - math.sqrt(disc)) / 2
        raiz_str = f"x1: {round(x1, 8)}, x2: {round(x2, 8)}"
    else:
        real = current_r / 2
        imag = math.sqrt(-disc) / 2
        raiz_str = f"x1,2: {round(real, 8)} ± {round(imag, 8)}i"

    return {
        "tipo": "bairstow",
        "resultados": resultados,
        "raiz": raiz_str,
        "convergencia": f"Factor cuadrático hallado: x² - ({round(current_r, 4)})x - ({round(current_s, 4)})",
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
        # Leemos la pizarra virtual
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
        # Leemos la pizarra virtual
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
        n_terminos = int(request.form['n_terminos'])
        
        datos = metodo_taylor(funcion, x0, x_eval, n_terminos)
        
    return render_template('taylor.html', datos=datos)

@app.route('/punto_fijo', methods=['GET', 'POST'])
def punto_fijo():
    datos = None
    if request.method == 'POST':
       
        funcion = request.form['ecuacion_latex'] 
        
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
  
        datos = metodo_punto_fijo(funcion, x0, tol, max_iter)
        
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
        funcion = request.form['funcion']
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_horner_newton(funcion, x0, tol, max_iter)
        
    return render_template('horner_newton.html', datos=datos)

@app.route('/muller', methods=['GET', 'POST'])
def muller():
    datos = None
    if request.method == 'POST':
        funcion = request.form['funcion']
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
        funcion = request.form['funcion']
        r = float(request.form['r'])
        s = float(request.form['s'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        datos = metodo_bairstow(funcion, r, s, tol, max_iter)
    return render_template('bairstow.html', datos=datos)

if __name__ == '__main__':
    app.run(debug=True)