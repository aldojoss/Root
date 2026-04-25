from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg') # Esto evita que la gráfica intente abrirse en una ventana de Windows y bloquee el servidor
import matplotlib.pyplot as plt
import io
import base64
import math

app = Flask(__name__)

def metodo_biseccion(funcion_str, xl, xu, tol, max_iter):
    # 1. Limpieza de la función
    funcion_str = funcion_str.replace('^', '**').replace('ln', 'log').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        # 2. Traducción matemática (Forzando la 'e')
        diccionario_matematico = {'e': sp.E, 'pi': sp.pi}
        funcion_simbolica = sp.sympify(funcion_str, locals=diccionario_matematico)
        
        # Un seguro extra por si la 'e' se escapa
        funcion_simbolica = funcion_simbolica.subs(sp.Symbol('e'), sp.E)
        
        f = sp.lambdify(x, funcion_simbolica, 'numpy') 

        # 3. LA PRUEBA DE FUEGO: Evaluamos los límites AQUÍ ADENTRO
        xl_original = xl
        xu_original = xu
        
        # Obligamos a que el resultado sea un número decimal (float)
        # Si la función tiene errores o letras sueltas, esto fallará y activará la alerta roja
        fxl = float(f(xl))
        fxu = float(f(xu))

    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo calcular la función. Detalle técnico: {str(err)}",
            "consejo": "Recuerda usar '*' para multiplicar (ej: 2*x) y usar 'exp(x)' en lugar de la letra 'e'.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html"
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
        
        # ... (Aquí termina el ciclo for de la bisección)

    # === GENERAR LA GRÁFICA ===
    # Creamos un arreglo de puntos X para dibujar la curva
    margen = (xu - xl) * 0.5 if 'xu_original' in locals() else 2
    x_vals = np.linspace(xl - margen, xu + margen, 200)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f(x)', color='#0d6efd', linewidth=2)
    plt.axhline(0, color='black', linewidth=1) # Eje X
    
    # Dibujar los límites y la raíz
    plt.axvline(xl, color='orange', linestyle='--', label='xl final')
    plt.axvline(xu, color='purple', linestyle='--', label='xu final')
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
def metodo_falsa_posicion(funcion_str, xl, xu, tol, max_iter):
    funcion_str = funcion_str.replace('^', '**').replace('ln', 'log').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        # 2. Traducción matemática (Forzando la 'e')
        diccionario_matematico = {'e': sp.E, 'pi': sp.pi}
        funcion_simbolica = sp.sympify(funcion_str, locals=diccionario_matematico)
        
        # Un seguro extra por si la 'e' se escapa
        funcion_simbolica = funcion_simbolica.subs(sp.Symbol('e'), sp.E)
        
        f = sp.lambdify(x, funcion_simbolica, 'numpy') 

        # 3. LA PRUEBA DE FUEGO: Evaluamos los límites AQUÍ ADENTRO
        xl_original = xl
        xu_original = xu
        
        # Obligamos a que el resultado sea un número decimal (float)
        # Si la función tiene errores o letras sueltas, esto fallará y activará la alerta roja
        fxl = float(f(xl))
        fxu = float(f(xu))

    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo calcular la función. Detalle técnico: {str(err)}",
            "consejo": "Recuerda usar '*' para multiplicar (ej: 2*x) y usar 'exp(x)' en lugar de la letra 'e'.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html"
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

    for i in range(1, max_iter + 1):
        fxl = f(xl)
        fxu = f(xu)
        
        # Evitar división por cero por si la línea se vuelve horizontal
        if fxl - fxu == 0:
            break

        # LA NUEVA FÓRMULA DE REGLA FALSA
        xr = xu - (fxu * (xl - xu)) / (fxl - fxu)
        fxr = f(xr)

        ea = abs((xr - xr_anterior) / xr) * 100 if i > 1 else 100

        resultados.append({
            "iteracion": i,
            "xl": round(xl, 8),
            "xu": round(xu, 8),
            "xr": round(xr, 8),
            "fxl": round(fxl, 8),
            "fxr": round(fxr, 8),
            "ea": round(ea, 8) if i > 1 else "---"
        })

        if i > 1 and ea < tol:
            break

        # Reemplazo de límites (igual que bisección)
        if fxl * fxr < 0:
            xu = xr
        elif fxl * fxr > 0:
            xl = xr
        else:
            break

        xr_anterior = xr

    # GENERAR LA GRÁFICA
    margen = (xu_original - xl_original) * 0.5
    x_vals = np.linspace(xl_original - margen, xu_original + margen, 200)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f(x)', color='#198754', linewidth=2) # En verde para distinguirlo
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
        "convergencia": "Lineal (Suele ser más rápida que Bisección).",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 3: NEWTON-RAPHSON
# ==========================================
def metodo_newton_raphson(funcion_str, x0, tol, max_iter):
    # 1. Limpieza de la función
    funcion_str = funcion_str.replace('^', '**').replace('ln', 'log').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        diccionario_matematico = {'e': sp.E, 'pi': sp.pi}
        funcion_simbolica = sp.sympify(funcion_str, locals=diccionario_matematico)
        
        # MAGIA PURA: SymPy calcula la derivada analítica por nosotros
        derivada_simbolica = sp.diff(funcion_simbolica, x)
        
        f = sp.lambdify(x, funcion_simbolica, 'numpy') 
        df = sp.lambdify(x, derivada_simbolica, 'numpy') 
    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar la función o su derivada. Detalle: {str(err)}",
            "consejo": "Recuerda usar '*' para multiplicaciones y 'exp(x)' en lugar de 'e'.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html"
        }

    resultados = []
    xi = x0

    for i in range(1, max_iter + 1):
        fxi = f(xi)
        dfxi = df(xi)
        
        # Validación crítica: Evitar división por cero si toca un mínimo/máximo
        if dfxi == 0:
            return {
                "error": True,
                "titulo": "⚠️ Derivada Cero (Línea Horizontal)",
                "mensaje": f"En la iteración {i}, la derivada se volvió 0.",
                "consejo": "El método de Newton falla cuando toca un valle o una cresta de la curva. Intenta con otro valor inicial (x0)."
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
            break
            
        xi = x_siguiente

    # === GENERAR LA GRÁFICA ===
    margen = abs(xi - x0) * 0.5 if abs(xi - x0) > 0 else 2
    x_min = min(x0, xi) - margen
    x_max = max(x0, xi) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals) 
    y_vals = f(x_vals)

    altura_maxima = max(abs(f(x_min)), abs(f(x_max))) * 3
    y_vals = np.clip(y_vals, -altura_maxima, altura_maxima)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'f(x)', color='#dc3545', linewidth=2) # Rojo para Newton
    plt.axhline(0, color='black', linewidth=1) 
    
    plt.axvline(x0, color='orange', linestyle='--', label='x0 inicial')
    plt.plot(xi, 0, 'go', markersize=8, label=f'Raíz ({round(xi, 8)})') # Punto verde para la raíz
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        "tipo": "abierto", # ¡Clave para que nuestra tabla cambie las columnas!
        "resultados": resultados, 
        "raiz": round(xi, 8),
        "convergencia": "Cuadrática O(n²) - ¡Es rapidísimo! Duplica los decimales correctos en cada paso.",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 4: SECANTE
# ==========================================
def metodo_secante(funcion_str, x0, x1, tol, max_iter):
    funcion_str = funcion_str.replace('^', '**').replace('ln', 'log').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        diccionario_matematico = {'e': sp.E, 'pi': sp.pi}
        funcion_simbolica = sp.sympify(funcion_str, locals=diccionario_matematico)
        f = sp.lambdify(x, funcion_simbolica, 'numpy') 
    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar la función. Detalle: {str(err)}",
            "consejo": "Recuerda usar '*' para multiplicaciones y 'exp(x)' en lugar de 'e'.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html"
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
            "ea": round(ea, 8) if i > 1 else "---" # El primer error a veces no se calcula, pero lo dejamos por consistencia
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
    plt.plot(x_vals, y_vals, label=f'f(x)', color='#0dcaf0', linewidth=2) # Cyan para la Secante
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
def serie_taylor(funcion_str, x0, x_eval, orden):
    funcion_str = funcion_str.replace('^', '**').replace('ln', 'log').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        diccionario_matematico = {'e': sp.E, 'pi': sp.pi}
        f_simbolica = sp.sympify(funcion_str, locals=diccionario_matematico)
        f_numpy = sp.lambdify(x, f_simbolica, 'numpy') 
        valor_verdadero = f_numpy(x_eval)
    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis",
            "mensaje": f"No se pudo evaluar la función. Detalle: {str(err)}",
            "consejo": "Asegúrate de escribir la función correctamente.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html"
        }

    resultados = []
    aproximacion = 0
    derivada_actual = f_simbolica
    polinomio_taylor = 0

    for i in range(orden + 1):
        # 1. Evaluar la derivada en el punto x0
        df_x0 = derivada_actual.subs(x, x0).evalf()
        
        # 2. Armar el término algebraico: ( f^(n)(x0) / n! ) * (x - x0)^n
        termino_algebraico = (df_x0 / math.factorial(i)) * (x - x0)**i
        polinomio_taylor += termino_algebraico
        
        # 3. Evaluar el término en el punto x solicitado (x_eval)
        valor_termino = termino_algebraico.subs(x, x_eval).evalf()
        aproximacion += valor_termino
        
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

        # Preparar la siguiente derivada
        derivada_actual = sp.diff(derivada_actual, x)

    # === GENERAR LA GRÁFICA (Comparativa) ===
    margen = abs(x_eval - x0) + 1
    x_min = min(x0, x_eval) - margen
    x_max = max(x0, x_eval) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals) 
    y_vals_original = f_numpy(x_vals)
    
    # Evaluar el polinomio de Taylor completo para la gráfica
    p_numpy = sp.lambdify(x, polinomio_taylor, 'numpy')
    y_vals_taylor = p_numpy(x_vals)
    # Si el polinomio resulta ser una constante, numpy necesita un array del mismo tamaño
    if isinstance(y_vals_taylor, (int, float)):
        y_vals_taylor = np.full_like(x_vals, y_vals_taylor)

    plt.figure(figsize=(8, 5))
    # Función real
    plt.plot(x_vals, y_vals_original, label=f'Original: f(x)', color='black', linewidth=3) 
    # Aproximación de Taylor
    plt.plot(x_vals, y_vals_taylor, label=f'Polinomio Taylor (Orden {orden})', color='#ffc107', linestyle='--', linewidth=2) 
    
    plt.axhline(0, color='gray', linewidth=1) 
    plt.axvline(x0, color='orange', linestyle=':', label='x0 (Centro)')
    plt.axvline(x_eval, color='purple', linestyle=':', label='x a evaluar')
    
    plt.plot(x_eval, valor_verdadero, 'ko', label=f'Valor Real ({round(float(valor_verdadero), 8)})')
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
        "raiz": round(float(aproximacion), 8), # Reusamos la tarjeta verde para mostrar la aproximación
        "convergencia": "Aproximación Polinomial (A mayor orden n, menor es el error).",
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 6: PUNTO FIJO (CON PREDICTOR)
# ==========================================
def metodo_punto_fijo(gx_str, x0, tol, max_iter):
    gx_str = gx_str.replace('^', '**').replace('ln', 'log').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        diccionario_matematico = {'e': sp.E, 'pi': sp.pi}
        g_simbolica = sp.sympify(gx_str, locals=diccionario_matematico)
        g_simbolica = g_simbolica.subs(sp.Symbol('e'), sp.E)
        
        g = sp.lambdify(x, g_simbolica, 'numpy') 
        
        # EL PREDICTOR DE CONVERGENCIA (La magia)
        derivada_g = sp.diff(g_simbolica, x) # Derivamos g(x)
        dg = sp.lambdify(x, derivada_g, 'numpy')
        
        # Evaluamos el valor absoluto de g'(x0)
        criterio_convergencia = abs(float(dg(x0)))
        
        # Generamos el diagnóstico
        if criterio_convergencia < 1:
            diagnostico = f"¡Excelente despeje! |g'(x0)| = {round(criterio_convergencia, 8)} < 1. Convergencia garantizada."
            color_diag = "success"
        else:
            diagnostico = f"¡Alerta de Divergencia! |g'(x0)| = {round(criterio_convergencia, 8)} > 1. Es muy probable que la función explote al infinito."
            color_diag = "danger"

    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar g(x) o su derivada. Detalle: {str(err)}",
            "consejo": "Recuerda usar '*' para multiplicaciones y 'exp(x)' en lugar de 'e'."
        }

    resultados = []
    xi = x0
    diverge = False

    for i in range(1, max_iter + 1):
        try:
            gxi = float(g(xi))
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
            "mensaje": diagnostico, # Mostramos por qué explotó basado en la derivada
            "consejo": "Esta calculadora hace milagros, pero no sabe despejar por ti. Si no sabes hacer un despeje algebraico válido para llegar a g(x), deberías considerar seriamente regresar al curso propedéutico. 📚"
        }

    # === GENERAR LA GRÁFICA ===
    margen = abs(xi - x0) + 2
    x_min = min(x0, xi) - margen
    x_max = max(x0, xi) + margen
    
    x_vals = np.linspace(x_min, x_max, 200)
    x_vals = np.where(x_vals == 0, 1e-10, x_vals) 
    y_vals_g = g(x_vals)
    y_vals_identidad = x_vals 

    altura_maxima = max(abs(g(x_min)), abs(g(x_max))) * 2
    y_vals_g = np.clip(y_vals_g, -altura_maxima, altura_maxima)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals_g, label=f'g(x)', color='#6f42c1', linewidth=2) 
    plt.plot(x_vals, y_vals_identidad, label=f'y = x', color='gray', linestyle='--', linewidth=1.5) 
    plt.axhline(0, color='black', linewidth=1) 
    plt.axvline(x0, color='orange', linestyle=':', label='x0 inicial')
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
        "convergencia": diagnostico, # Mandamos el diagnóstico a la tarjeta azul
        "grafica": grafica_url
    }
    
# ==========================================
# MÉTODO 7: MÉTODO DE HORNER
# ==========================================
def metodo_horner(funcion_str, x0):
    funcion_str = funcion_str.replace('^', '**').replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        f_simbolica = sp.sympify(funcion_str)
        
        # Validación de seguridad: ¿Es realmente un polinomio?
        if not f_simbolica.is_polynomial(x):
            return {
                "error": True,
                "titulo": "🛑 Función No Polinomial",
                "mensaje": "El Método de Horner es una técnica de división sintética que SOLO funciona con polinomios.",
                "consejo": "Ingresa una función polinomial válida (Ej: 2*x**3 - 4*x**2 + x - 5). No uses fracciones con x en el denominador, trigonométricas o logaritmos."
            }
            
        polinomio = sp.Poly(f_simbolica, x)
        coeffs = polinomio.all_coeffs() # Extrae todos los coeficientes, incluyendo ceros implícitos
        n = polinomio.degree()
        
    except Exception as err:
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar el polinomio. Detalle: {str(err)}",
            "consejo": "Asegúrate de escribir el polinomio correctamente."
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
    
    # El ciclo de Horner
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
    f_numpy = sp.lambdify(x, f_simbolica, 'numpy') 
    margen = abs(x0) * 0.5 if x0 != 0 else 5
    x_min = x0 - margen - 2
    x_max = x0 + margen + 2
    
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = f_numpy(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'P(x)', color='#fd7e14', linewidth=2) # Naranja para Horner
    plt.axhline(0, color='black', linewidth=1) 
    
    # Dibujar el punto evaluado
    plt.axvline(x0, color='gray', linestyle=':', label=f'x0 = {x0}')
    plt.plot(x0, b_actual, 'bo', markersize=8, label=f'P({x0}) = {round(b_actual, 4)}') 
    
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
        "raiz": round(b_actual, 8), # Usamos el espacio de "raíz" para mostrar el residuo/evaluación final
        "convergencia": "Evaluación Polinomial / División Sintética Exitosa.",
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
        funcion = request.form['funcion']
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
        funcion = request.form['funcion']
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
        funcion = request.form['funcion']
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_newton_raphson(funcion, x0, tol, max_iter)
        
    return render_template('newton.html', datos=datos)

@app.route('/secante', methods=['GET', 'POST'])
def secante():
    datos = None
    if request.method == 'POST':
        funcion = request.form['funcion']
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
        funcion = request.form['funcion']
        x0 = float(request.form['x0'])
        x_eval = float(request.form['x_eval'])
        orden = int(request.form['orden'])
        
        datos = serie_taylor(funcion, x0, x_eval, orden)
        
    return render_template('taylor.html', datos=datos)

@app.route('/punto_fijo', methods=['GET', 'POST'])
def punto_fijo():
    datos = None
    if request.method == 'POST':
        gx = request.form['gx']
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        datos = metodo_punto_fijo(gx, x0, tol, max_iter)
        
    return render_template('punto_fijo.html', datos=datos)

@app.route('/horner', methods=['GET', 'POST'])
def horner():
    datos = None
    if request.method == 'POST':
        funcion = request.form['funcion']
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