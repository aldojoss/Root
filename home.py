from flask import Flask, render_template, request
import sympy as sp
import numpy as np

app = Flask(__name__)

def metodo_biseccion(funcion_str, xl, xu, tol, max_iter):
    # 1. Limpieza y traducción de la función
    funcion_str = funcion_str.replace('^', '**')   # Arregla el problema del exponente
    funcion_str = funcion_str.replace('ln', 'log') # Arregla el problema del logaritmo natural
    
    # 2. Preparar la función matemática
    x = sp.Symbol('x')
    try:
        funcion_simbolica = sp.sympify(funcion_str)
        f = sp.lambdify(x, funcion_simbolica, 'numpy') 
    except Exception as e:
        return {"error": "Error al leer la función. Asegúrate de usar 'x' como variable."}

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

    return {"resultados": resultados, "raiz": round(xr, 4)}

# Ruta principal que maneja la vista y el formulario
@app.route('/', methods=['GET', 'POST'])
def inicio():
    datos = None
    if request.method == 'POST':
        # Capturamos lo que el usuario escribió en la web
        funcion = request.form['funcion']
        xl = float(request.form['xl'])
        xu = float(request.form['xu'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])
        
        # Ejecutamos el método
        datos = metodo_biseccion(funcion, xl, xu, tol, max_iter)
        
    # Enviamos los datos al HTML
    return render_template('index.html', datos=datos)

if __name__ == '__main__':
    app.run(debug=True)