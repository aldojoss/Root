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
    # 1. Limpieza y traducción de la función
    funcion_str = funcion_str.replace('^', '**')
    funcion_str = funcion_str.replace('ln', 'log')
    funcion_str = funcion_str.replace('x(', 'x*(').replace('X(', 'x*(')
    
    # 2. Preparar la función matemática
    x = sp.Symbol('x')
    try:
        funcion_simbolica = sp.sympify(funcion_str)
        f = sp.lambdify(x, funcion_simbolica, 'numpy') 
    except Exception as err: # Atrapamos el error exacto en la variable 'err'
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar la función. Detalle técnico: {str(err)}",
            "consejo": "Recuerda usar '*' para multiplicaciones (ej: 2*x) y 'exp(x)' en lugar de 'e' para la exponencial.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html"
        }
    # ... (el resto del código sigue exactamente igual)

    # 2. Validar que exista un cambio de signo (condición de bisección)
    fxl = f(xl)
    fxu = f(xu)
    
    if fxl * fxu >= 0:
        return {
            "error": True,
            "titulo": "⚠️ Intervalo sin cambio de signo",
            "mensaje": f"Evaluamos tus límites y obtuvimos f({xl}) = {round(fxl, 4)} y f({xu}) = {round(fxu, 4)}. Como ambos resultados tienen el mismo signo, la curva no cruza el cero (eje X) en este tramo.",
            "consejo": "Para que la bisección funcione, un resultado debe ser positivo y el otro negativo. ¡Intenta con otros valores para xl y xu!"
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
            "xl": round(xl, 4),
            "xu": round(xu, 4),
            "xr": round(xr, 4),
            "fxl": round(fxl, 4),
            "fxr": round(fxr, 4),
            "ea": round(ea, 4) if i > 1 else "---"
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
    plt.plot(xr, 0, 'ro', markersize=8, label=f'Raíz ({round(xr, 4)})')
    
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
        "raiz": round(xr, 4),
        "convergencia": "Lineal O(n) - El error se reduce a la mitad por iteración.",
        "grafica": grafica_url
    }

    return {"resultados": resultados, "raiz": round(xr, 4)}

# ==========================================
# MÉTODO 2: REGLA FALSA (FALSA POSICIÓN)
# ==========================================
def metodo_falsa_posicion(funcion_str, xl, xu, tol, max_iter):
    funcion_str = funcion_str.replace('^', '**').replace('ln', 'log')
    funcion_str = funcion_str.replace('x(', 'x*(').replace('X(', 'x*(')
    x = sp.Symbol('x')
    
    try:
        funcion_simbolica = sp.sympify(funcion_str)
        f = sp.lambdify(x, funcion_simbolica, 'numpy') 
    except Exception as err: # Atrapamos el error exacto en la variable 'err'
        return {
            "error": True,
            "titulo": "🛑 Error de Sintaxis Matemática",
            "mensaje": f"No se pudo evaluar la función. Detalle técnico: {str(err)}",
            "consejo": "Recuerda usar '*' para multiplicaciones (ej: 2*x) y 'exp(x)' en lugar de 'e' para la exponencial.",
            "link_sympy": "https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html"
        }

    xl_original = xl
    xu_original = xu
    fxl = f(xl)
    fxu = f(xu)
    
    if fxl * fxu >= 0:
        return {
            "error": True,
            "titulo": "⚠️ Intervalo sin cambio de signo",
            "mensaje": f"Evaluamos tus límites y obtuvimos f({xl}) = {round(fxl, 4)} y f({xu}) = {round(fxu, 4)}. Mismo signo.",
            "consejo": "Para la Regla Falsa, un resultado debe ser positivo y el otro negativo."
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
            "xl": round(xl, 4),
            "xu": round(xu, 4),
            "xr": round(xr, 4),
            "fxl": round(fxl, 4),
            "fxr": round(fxr, 4),
            "ea": round(ea, 4) if i > 1 else "---"
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
    plt.plot(xr, 0, 'ro', markersize=8, label=f'Raíz ({round(xr, 4)})')
    
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
        "raiz": round(xr, 4),
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
            "xi": round(xi, 6),
            "fxi": round(fxi, 6),
            "dfxi": round(dfxi, 6),
            "x_siguiente": round(x_siguiente, 6),
            "ea": round(ea, 6) if i > 1 else "---"
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
    plt.plot(xi, 0, 'go', markersize=8, label=f'Raíz ({round(xi, 4)})') # Punto verde para la raíz
    
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
        "raiz": round(xi, 6),
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
            "xi_menos_1": round(xi_menos_1, 6),
            "xi": round(xi, 6),
            "fxi_menos_1": round(fxi_menos_1, 6),
            "fxi": round(fxi, 6),
            "x_siguiente": round(x_siguiente, 6),
            "ea": round(ea, 6) if i > 1 else "---" # El primer error a veces no se calcula, pero lo dejamos por consistencia
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
    plt.plot(xi, 0, 'ro', markersize=8, label=f'Raíz ({round(xi, 4)})') 
    
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
        "raiz": round(xi, 6),
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
            "df_x0": round(float(df_x0), 6),
            "termino": round(float(valor_termino), 6),
            "aproximacion": round(float(aproximacion), 6),
            "et": round(float(et), 6)
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
    
    plt.plot(x_eval, valor_verdadero, 'ko', label=f'Valor Real ({round(float(valor_verdadero), 2)})')
    plt.plot(x_eval, float(aproximacion), 'yo', label=f'Aprox ({round(float(aproximacion), 2)})')
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.ylim(min(y_vals_original)-2, max(y_vals_original)+2) # Limitar altura para que no se dispare
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
        "raiz": round(float(aproximacion), 6), # Reusamos la tarjeta verde para mostrar la aproximación
        "convergencia": "Aproximación Polinomial (A mayor orden n, menor es el error).",
        "grafica": grafica_url
    }

# Ruta 1: Pantalla de inicio
@app.route('/')
def inicio():
    return render_template('index.html')

# Ruta 2: Calculadora de Bisección
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

# RUTA PARA LA PÁGINA DE REGLA FALSA
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

if __name__ == '__main__':
    app.run(debug=True)