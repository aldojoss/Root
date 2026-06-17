"""Contenido didáctico para el laboratorio interactivo de métodos numéricos.

La ruta de teoría no ejecuta los métodos: explica su funcionamiento. Por eso
este módulo guarda datos pedagógicos en una estructura uniforme que la plantilla
puede renderizar sin meter lógica grande dentro de Flask.
"""


def _demo(tipo, modelo, iteraciones, etapas):
    """Construye la configuración común de una demo visual."""
    return {
        "tipo": tipo,
        "modelo": modelo,
        "iteraciones": iteraciones,
        "etapas": etapas,
    }


def _metodo(
    id_,
    familia,
    nombre,
    subtitulo,
    icono,
    color,
    formula,
    idea,
    condicion,
    convergencia,
    fortalezas,
    riesgos,
    pasos,
    demo,
    ruta,
):
    """Normaliza un método para que la interfaz pueda consumirlo directo."""
    return {
        "id": id_,
        "familia": familia,
        "nombre": nombre,
        "subtitulo": subtitulo,
        "icono": icono,
        "color": color,
        "formula": formula,
        "idea": idea,
        "condicion": condicion,
        "convergencia": convergencia,
        "fortalezas": fortalezas,
        "riesgos": riesgos,
        "pasos": pasos,
        "demo": demo,
        "ruta": ruta,
    }


FAMILIAS_TEORIA = [
    {
        "id": "todos",
        "nombre": "Todos",
        "icono": "bi-collection",
        "resumen": "Vista completa del laboratorio teórico.",
    },
    {
        "id": "cerrado",
        "nombre": "Cerrados",
        "icono": "bi-shield-check",
        "resumen": "Usan intervalos y cambio de signo. Priorizan garantía.",
    },
    {
        "id": "abierto",
        "nombre": "Abiertos",
        "icono": "bi-lightning-charge",
        "resumen": "Usan semillas iniciales. Pueden ser rápidos, pero sensibles.",
    },
    {
        "id": "especializado",
        "nombre": "Especializados",
        "icono": "bi-cpu",
        "resumen": "Series, polinomios y métodos para raíces más particulares.",
    },
    {
        "id": "calculo",
        "nombre": "Cálculo",
        "icono": "bi-calculator",
        "resumen": "Aproximan integrales y derivadas desde valores de una función.",
    },
    {
        "id": "sistema",
        "nombre": "Sistemas",
        "icono": "bi-grid-3x3-gap",
        "resumen": "Resuelven sistemas lineales o no lineales con matrices.",
    },
    {
        "id": "datos",
        "nombre": "Datos",
        "icono": "bi-graph-up",
        "resumen": "Interpolan o ajustan puntos experimentales.",
    },
]


METODOS_TEORIA = [
    _metodo(
        "biseccion",
        "cerrado",
        "Bisección",
        "Garantía por cambio de signo",
        "bi-arrows-collapse-vertical",
        "#64a8ff",
        r"x_r=\frac{x_l+x_u}{2}",
        (
            "Divide el intervalo en dos mitades y conserva la mitad donde se mantiene "
            "el cambio de signo. Es lento, pero muy confiable cuando se cumple Bolzano."
        ),
        r"Requiere \( f(x_l)\cdot f(x_u)<0 \) y continuidad en \( [x_l,x_u] \).",
        "Lineal. El intervalo se reduce a la mitad en cada iteración.",
        [
            "No necesita derivadas.",
            "Tiene convergencia garantizada si hay cambio de signo.",
            "Es ideal como respaldo cuando otros métodos fallan.",
        ],
        [
            "Puede ser lento si la tolerancia es muy pequeña.",
            "No detecta raíces pares si no hay cambio de signo.",
        ],
        [
            r"Evaluar \( f(x_l) \) y \( f(x_u) \).",
            r"Calcular el punto medio \( x_r \).",
            r"Evaluar \( f(x_r) \).",
            "Conservar el subintervalo donde hay cambio de signo.",
            "Repetir hasta alcanzar la tolerancia.",
        ],
        {
            "tipo": "bracket",
            "modelo": "f(x) = x^3 - x - 2",
            "funcion": "x^3 - x - 2",
            "xl": 1,
            "xu": 2,
            "iteraciones": 8,
            "etapas": [
                "Se parte de un intervalo donde la función cambia de signo.",
                "El punto medio divide el intervalo en dos candidatos.",
                "Se conserva el lado que mantiene la raíz encerrada.",
            ],
        },
        "/biseccion",
    ),
    _metodo(
        "regla_falsa",
        "cerrado",
        "Regla Falsa",
        "Interpolación lineal con garantía",
        "bi-slash-lg",
        "#48ddb0",
        r"x_r=x_u-\frac{f(x_u)(x_l-x_u)}{f(x_l)-f(x_u)}",
        (
            "Une los extremos del intervalo con una recta y usa el corte de esa recta "
            "con el eje x como nueva aproximación. Mantiene la seguridad de los métodos cerrados."
        ),
        r"Igual que Bisección: \( f(x_l)\cdot f(x_u)<0 \) y continuidad.",
        "Lineal. Suele avanzar más que Bisección si la secante representa bien la curva.",
        [
            "No necesita derivadas.",
            "Aprovecha la forma de la función mejor que el punto medio.",
            "Conserva un intervalo con raíz garantizada.",
        ],
        [
            "Puede estancarse si uno de los extremos casi no cambia.",
            "Depende de que la recta secante no sea una mala aproximación local.",
        ],
        [
            "Verificar cambio de signo.",
            r"Trazar la recta entre \( (x_l,f(x_l)) \) y \( (x_u,f(x_u)) \).",
            r"Calcular el corte \( x_r \) con el eje x.",
            r"Actualizar el extremo que tenga el mismo signo que \( f(x_r) \).",
            "Repetir hasta cumplir la tolerancia.",
        ],
        {
            "tipo": "false-position",
            "modelo": "f(x) = x^3 - x - 2",
            "funcion": "x^3 - x - 2",
            "xl": 1,
            "xu": 2,
            "iteraciones": 8,
            "etapas": [
                "El intervalo encierra la raíz.",
                "La secante entre extremos estima el corte con el eje x.",
                "El intervalo se actualiza sin perder el cambio de signo.",
            ],
        },
        "/falsa_posicion",
    ),
    _metodo(
        "newton",
        "abierto",
        "Newton-Raphson",
        "Tangente hacia la raíz",
        "bi-lightning-charge",
        "#ff8658",
        r"x_{i+1}=x_i-\frac{f(x_i)}{f'(x_i)}",
        (
            "Aproxima la función por la recta tangente en el punto actual. El corte "
            "de esa tangente con el eje x se vuelve la siguiente aproximación."
        ),
        r"Necesita \( f'(x_i) \) definida y no cercana a cero.",
        "Cuadrática cerca de una raíz simple y con buen punto inicial.",
        [
            "Muy rápido cuando x₀ está cerca de la raíz.",
            "Usa la pendiente, por eso suele requerir pocas iteraciones.",
            "Excelente para funciones suaves.",
        ],
        [
            "Puede divergir con un mal x₀.",
            r"Falla si \( f'(x_i)=0 \) o casi cero.",
            "Puede saltar fuera del dominio de la función.",
        ],
        [
            "Escoger x₀.",
            r"Calcular \( f(x_i) \) y \( f'(x_i) \).",
            r"Construir la tangente en \( x_i \).",
            r"Tomar el corte con el eje x como \( x_{i+1} \).",
            "Repetir hasta que el error sea pequeño.",
        ],
        {
            "tipo": "newton",
            "modelo": "f(x) = x^2 - 2",
            "funcion": "x^2 - 2",
            "x0": 1.6,
            "iteraciones": 7,
            "etapas": [
                "La tangente toca la curva en la aproximación actual.",
                "El corte de esa tangente produce la siguiente x.",
                "Si la semilla es buena, los saltos se vuelven muy pequeños.",
            ],
        },
        "/newton",
    ),
    _metodo(
        "secante",
        "abierto",
        "Secante",
        "Newton sin derivada explícita",
        "bi-diagonal",
        "#d2a1ff",
        r"x_{i+1}=x_i-\frac{f(x_i)(x_i-x_{i-1})}{f(x_i)-f(x_{i-1})}",
        (
            "Sustituye la derivada de Newton por la pendiente de una recta secante "
            "construida usando dos aproximaciones previas."
        ),
        r"Necesita dos semillas distintas y \( f(x_i)-f(x_{i-1}) \) no cercano a cero.",
        r"Superlineal, de orden aproximado \( \varphi\approx1.618 \).",
        [
            "No necesita calcular derivadas.",
            "Suele ser más rápido que métodos cerrados.",
            "Útil cuando derivar es incómodo.",
        ],
        [
            "No garantiza convergencia.",
            "Puede fallar si la secante queda casi horizontal.",
            "Es sensible a las semillas iniciales.",
        ],
        [
            "Escoger x₀ y x₁.",
            "Trazar la recta entre ambos puntos de la curva.",
            "Usar el corte de esa recta con el eje x como x₂.",
            "Desplazar las semillas: x₀ ← x₁, x₁ ← x₂.",
            "Repetir hasta cumplir la tolerancia.",
        ],
        {
            "tipo": "secant",
            "modelo": "f(x) = x^2 - 2",
            "funcion": "x^2 - 2",
            "x0": 1,
            "x1": 2,
            "iteraciones": 7,
            "etapas": [
                "Dos puntos generan una recta secante.",
                "El corte con el eje x reemplaza una de las semillas.",
                "La secante se va acercando a la raíz.",
            ],
        },
        "/secante",
    ),
    _metodo(
        "punto_fijo",
        "abierto",
        "Punto Fijo",
        "Buscar x = g(x)",
        "bi-arrow-repeat",
        "#ffd98a",
        r"x_{i+1}=g(x_i)",
        (
            r"Reescribe el problema como \( x=g(x) \). La raíz aparece cuando la curva "
            r"\( y=g(x) \) interseca la recta identidad \( y=x \)."
        ),
        r"Conviene que \( |g'(x)|<1 \) cerca del punto fijo.",
        "Lineal cuando g es contractiva cerca de la solución.",
        [
            "Es conceptualmente simple.",
            "Permite visualizar convergencia con diagramas de telaraña.",
            "Sirve para estudiar estabilidad de iteraciones.",
        ],
        [
            "La forma g(x) elegida cambia totalmente el comportamiento.",
            r"Si \( |g'(x)|\ge 1 \), puede divergir u oscilar.",
            r"No cualquier despeje de \( f(x)=0 \) sirve.",
        ],
        [
            r"Transformar \( f(x)=0 \) en \( x=g(x) \).",
            "Escoger x₀.",
            r"Calcular \( x_{i+1}=g(x_i) \).",
            r"Comparar \( x_{i+1} \) con \( x_i \).",
            "Repetir hasta estabilizarse.",
        ],
        {
            "tipo": "fixed-point",
            "modelo": "g(x) = cos(x)",
            "funcion": "cos(x)",
            "x0": 0.5,
            "iteraciones": 10,
            "etapas": [
                "Se evalúa g en la semilla actual.",
                "Se proyecta hacia la recta y = x.",
                "La telaraña converge si g contrae las distancias.",
            ],
        },
        "/punto_fijo",
    ),
    _metodo(
        "taylor",
        "especializado",
        "Series de Taylor",
        "Aproximar funciones con derivadas",
        "bi-activity",
        "#f48fb1",
        r"P_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k",
        (
            "Construye un polinomio local alrededor de un centro a. Cada término agrega "
            "información de una derivada: valor, pendiente, curvatura y cambios sucesivos."
        ),
        r"Requiere derivadas sucesivas definidas cerca del centro \( a \).",
        "La precisión mejora cerca del centro; lejos de a puede necesitar muchos términos.",
        [
            "Explica cómo se comporta una función cerca de un punto.",
            "Permite aproximar funciones complicadas con polinomios.",
            "Muestra claramente el aporte de cada derivada.",
        ],
        [
            "Puede ser mala lejos del centro de expansión.",
            "Algunas funciones necesitan demasiados términos.",
            "Si hay problemas de dominio, la serie puede dejar de representar a la función.",
        ],
        [
            "Elegir el centro a.",
            r"Calcular \( f(a), f'(a), f''(a), \dots \).",
            r"Dividir cada derivada entre \( k! \).",
            r"Formar los términos \( \frac{f^{(k)}(a)}{k!}(x-a)^k \).",
            "Sumar términos hasta el orden pedido.",
        ],
        _demo(
            "taylor",
            "f(x)=e^x alrededor de a=0",
            5,
            [
                "Orden 0: solo conserva el valor de la función.",
                "Orden 1: agrega la pendiente local.",
                "Orden 2: agrega curvatura.",
                "Más términos hacen que el polinomio abrace mejor la función cerca del centro.",
                "El error crece al alejarse del centro si el orden es bajo.",
                "Con suficientes términos, la aproximación local se vuelve muy precisa.",
            ],
        ),
        "/taylor",
    ),
    _metodo(
        "horner",
        "especializado",
        "Horner",
        "Evaluación sintética de polinomios",
        "bi-table",
        "#4dd4ff",
        r"P(x_0)=((a_nx_0+a_{n-1})x_0+\cdots+a_1)x_0+a_0",
        (
            "Reorganiza el polinomio para evaluarlo con multiplicaciones encadenadas. "
            "Evita calcular muchas potencias y también sirve para división sintética."
        ),
        "Necesita los coeficientes ordenados por grado descendente.",
        r"Es exacto algebraicamente; numéricamente reduce operaciones y acumulación de error.",
        [
            "Rápido para evaluar polinomios.",
            r"Facilita división sintética entre \( x-x_0 \).",
            "Es la base de Horner-Newton y Bairstow.",
        ],
        [
            "Depende de ingresar bien el orden de coeficientes.",
            "No encuentra raíces por sí solo; evalúa o divide.",
            "Con coeficientes enormes puede aparecer sensibilidad numérica.",
        ],
        [
            "Bajar el primer coeficiente.",
            "Multiplicar por x₀.",
            "Sumar con el siguiente coeficiente.",
            "Repetir hasta llegar al término independiente.",
            r"El último valor es \( P(x_0) \).",
        ],
        _demo(
            "horner",
            "P(x)=x^3-6x^2+11x-6 en x=2",
            4,
            [
                "Coeficientes: 1, -6, 11, -6.",
                "Baja 1.",
                "Multiplica por 2 y suma con -6.",
                "Repite el patrón hasta el final.",
                "El residuo final es P(2).",
            ],
        ),
        "/horner",
    ),
    _metodo(
        "horner_newton",
        "especializado",
        "Horner-Newton",
        "Newton optimizado para polinomios",
        "bi-bezier2",
        "#00c2a8",
        r"x_{i+1}=x_i-\frac{P(x_i)}{P'(x_i)}",
        (
            r"Aplica Newton a polinomios, pero evalúa \( P(x_i) \) y \( P'(x_i) \) "
            "con esquemas tipo Horner para reducir trabajo."
        ),
        r"Necesita un polinomio y \( P'(x_i) \neq 0 \).",
        "Cuadrática cerca de una raíz simple; eficiente porque evalúa con Horner.",
        [
            "Especialmente bueno para polinomios.",
            "Evita derivar y evaluar de forma pesada.",
            "Puede combinarse con deflación para buscar varias raíces.",
        ],
        [
            "Sigue dependiendo del punto inicial.",
            "Raíces múltiples reducen la velocidad.",
            "Si se defla mal, los errores se propagan.",
        ],
        [
            "Elegir una semilla x₀.",
            "Evaluar P(xᵢ) con Horner.",
            "Evaluar P'(xᵢ) con Horner o derivada sintética.",
            "Aplicar la fórmula de Newton.",
            "Repetir hasta converger.",
        ],
        _demo(
            "horner-newton",
            "P(x)=x^3-6x^2+11x-6",
            5,
            [
                "Horner evalúa P(xᵢ).",
                "Otro esquema evalúa P'(xᵢ).",
                "Newton corrige la semilla.",
                "La nueva semilla se reevalúa.",
                "El proceso se detiene cuando P(xᵢ) es casi cero.",
                "La raíz encontrada puede usarse para deflactar el polinomio.",
            ],
        ),
        "/horner_newton",
    ),
    _metodo(
        "muller",
        "especializado",
        "Müller",
        "Parábola que busca raíces",
        "bi-node-plus",
        "#c084fc",
        r"x_{i+1}=x_i+\frac{-2c}{b\pm\sqrt{b^2-4ac}}",
        (
            "Usa tres puntos para construir una parábola interpolante. La raíz de esa "
            "parábola se toma como la siguiente aproximación."
        ),
        "Necesita tres semillas distintas y evitar denominadores pequeños.",
        "Suele ser superlineal y puede encontrar raíces complejas.",
        [
            "No necesita derivadas.",
            "Puede trabajar con raíces complejas.",
            "La parábola captura curvatura mejor que una secante.",
        ],
        [
            "La fórmula puede sufrir cancelación si se elige mal el signo.",
            "No garantiza convergencia global.",
            "Tres semillas malas pueden llevar a saltos grandes.",
        ],
        [
            "Escoger x₀, x₁ y x₂.",
            "Construir la parábola que pasa por los tres puntos.",
            "Resolver la raíz de esa parábola.",
            "Descartar el punto más antiguo.",
            "Repetir con el nuevo trío.",
        ],
        _demo(
            "muller",
            "Tres puntos → parábola → raíz aproximada",
            4,
            [
                "Se toman tres aproximaciones iniciales.",
                "La curva local se aproxima con una parábola.",
                "Se calcula el corte de la parábola.",
                "El nuevo punto reemplaza al más antiguo.",
                "La parábola se va ajustando alrededor de la raíz.",
            ],
        ),
        "/muller",
    ),
    _metodo(
        "bairstow",
        "especializado",
        "Bairstow",
        "Factores cuadráticos de polinomios",
        "bi-boxes",
        "#ffb86b",
        r"P(x)=(x^2-rx-s)Q(x)+R(x)",
        (
            "Busca factores cuadráticos de un polinomio. Ajusta r y s hasta que el "
            "residuo sea casi cero, permitiendo obtener raíces reales o complejas."
        ),
        "Necesita un polinomio real y valores iniciales para r y s.",
        "Iterativo; converge bien con buenas semillas y polinomios razonablemente condicionados.",
        [
            "Encuentra pares de raíces usando factores cuadráticos.",
            "Maneja raíces complejas conjugadas sin salir de coeficientes reales.",
            "Permite deflactar el polinomio por bloques.",
        ],
        [
            "Puede no converger con r y s mal elegidos.",
            "Polinomios mal condicionados amplifican errores.",
            "Requiere cuidado al deflactar varias veces.",
        ],
        [
            r"Suponer un factor \( x^2-rx-s \).",
            "Dividir sintéticamente el polinomio.",
            "Medir los residuos.",
            "Corregir r y s resolviendo un sistema pequeño.",
            "Repetir hasta que el residuo sea cercano a cero.",
        ],
        _demo(
            "bairstow",
            "P(x) → factor cuadrático → raíces",
            5,
            [
                "Se propone un factor cuadrático.",
                "La división sintética calcula residuos.",
                "Los residuos indican cuánto corregir r y s.",
                "Se ajusta el factor.",
                "Cuando el residuo desaparece, se extrae el factor.",
                "El proceso continúa con el polinomio reducido.",
            ],
        ),
        "/bairstow",
    ),
    _metodo(
        "gauss",
        "sistema",
        "Eliminación de Gauss",
        "Triangular y sustituir",
        "bi-grid-3x3",
        "#7aa2ff",
        r"[A|b]\longrightarrow[U|c]",
        (
            "Transforma la matriz aumentada en una forma triangular superior. Luego "
            "resuelve desde la última ecuación hacia la primera por sustitución regresiva."
        ),
        r"Requiere matriz cuadrada y pivotes no nulos; con pivoteo mejora la estabilidad.",
        r"Método directo: termina en un número finito de pasos, salvo singularidad.",
        [
            "Muy usado para sistemas pequeños y medianos.",
            "Produce una solución directa sin iterar.",
            "El pivoteo evita divisiones peligrosas por números pequeños.",
        ],
        [
            "Sin pivoteo puede fallar aunque el sistema tenga solución.",
            "Matrices casi singulares son sensibles.",
            "Para sistemas gigantes puede ser costoso.",
        ],
        [
            "Formar la matriz aumentada.",
            "Elegir un pivote por columna.",
            "Anular los elementos debajo del pivote.",
            "Obtener una matriz triangular superior.",
            "Aplicar sustitución regresiva.",
        ],
        _demo(
            "matrix",
            "Sistema 3x3 aumentado",
            4,
            [
                "Matriz aumentada inicial.",
                "Primer pivote: se eliminan entradas debajo.",
                "Segundo pivote: queda estructura triangular.",
                "La última ecuación se resuelve primero.",
                "La sustitución regresiva completa x, y, z.",
            ],
        ),
        "/gauss",
    ),
    _metodo(
        "gauss_jordan",
        "sistema",
        "Gauss-Jordan",
        "Reducir hasta identidad",
        "bi-layout-three-columns",
        "#5eead4",
        r"[A|b]\longrightarrow[I|x]",
        (
            "Continúa la eliminación hasta convertir A en la identidad. Al final, el "
            "vector solución queda directamente en la columna aumentada."
        ),
        "Requiere sistema cuadrado con matriz invertible.",
        "Método directo; más trabajo que Gauss, pero deja la solución explícita.",
        [
            "La solución se lee sin sustitución regresiva.",
            "Útil para calcular inversas y revisar consistencia.",
            "Didácticamente muestra toda la reducción.",
        ],
        [
            "Hace más operaciones que Gauss simple.",
            "Necesita pivotes adecuados.",
            "Puede acumular más error si no se pivotea.",
        ],
        [
            "Formar la matriz aumentada.",
            "Normalizar cada fila pivote.",
            "Anular arriba y abajo de cada pivote.",
            "Llegar a la forma reducida por filas.",
            "Leer la solución en la última columna.",
        ],
        _demo(
            "matrix-reduction",
            "Matriz aumentada → identidad",
            4,
            [
                "Se ubica un pivote.",
                "La fila pivote se normaliza.",
                "Se limpian las entradas de toda la columna.",
                "A se convierte poco a poco en I.",
                "La columna derecha queda como solución.",
            ],
        ),
        "/gauss_jordan",
    ),
    _metodo(
        "lu",
        "sistema",
        "Factorización LU",
        "Separar A en L y U",
        "bi-layers",
        "#a7f3d0",
        r"A=LU,\qquad Ly=b,\qquad Ux=y",
        (
            "Descompone la matriz en una triangular inferior L y una triangular superior U. "
            "Luego resuelve dos sistemas triangulares más simples."
        ),
        "Requiere que la factorización exista; con pivoteo se usa PA = LU.",
        "Método directo. Muy eficiente si se resuelven varios b con la misma A.",
        [
            "Reutiliza la factorización para varios vectores b.",
            "Organiza la eliminación de Gauss en dos matrices.",
            "Los sistemas triangulares son rápidos de resolver.",
        ],
        [
            "Puede necesitar pivoteo parcial.",
            "No conviene si solo se resuelve una vez y el sistema es muy pequeño.",
            "Matrices singulares o casi singulares complican la factorización.",
        ],
        [
            "Factorizar A como LU.",
            "Resolver Ly = b por sustitución progresiva.",
            "Resolver Ux = y por sustitución regresiva.",
            r"Verificar el residuo \( Ax-b \).",
        ],
        _demo(
            "lu",
            "A se separa en L y U",
            4,
            [
                "A contiene toda la información del sistema.",
                "L guarda multiplicadores de eliminación.",
                "U guarda la forma triangular superior.",
                "Primero se resuelve Ly=b.",
                "Luego se resuelve Ux=y.",
            ],
        ),
        "/lu",
    ),
    _metodo(
        "jacobi",
        "sistema",
        "Jacobi",
        "Iterar con valores anteriores",
        "bi-arrow-repeat",
        "#93c5fd",
        r"x^{(k+1)}=D^{-1}(b-(L+U)x^{(k)})",
        (
            "Despeja cada incógnita usando solo los valores de la iteración anterior. "
            "Por eso todas las nuevas componentes pueden calcularse en paralelo."
        ),
        r"Converge si \( \rho(T_J)<1 \); diagonal dominante ayuda mucho.",
        "Iterativo lineal; suele ser más lento que Gauss-Seidel.",
        [
            "Fácil de entender y programar.",
            "Puede paralelizarse porque no usa valores recién calculados.",
            "El radio espectral permite anticipar convergencia.",
        ],
        [
            "Puede divergir si el sistema no cumple condiciones.",
            "Converge lento en muchos casos.",
            "Necesita una semilla inicial y tolerancia.",
        ],
        [
            "Separar A en D, L y U.",
            "Despejar cada variable.",
            "Evaluar todas las nuevas variables con la iteración anterior.",
            "Calcular el error entre iteraciones.",
            "Detenerse si el error cae bajo la tolerancia.",
        ],
        _demo(
            "iterative-linear",
            "x(k+1) usa únicamente x(k)",
            5,
            [
                "Se parte de una semilla inicial.",
                "Cada ecuación produce una nueva componente.",
                "Todas usan los valores viejos.",
                "Se mide el cambio entre iteraciones.",
                "Si el radio espectral es menor que 1, el proceso se contrae.",
                "La solución aparece cuando los cambios son pequeños.",
            ],
        ),
        "/jacobi",
    ),
    _metodo(
        "gauss_seidel",
        "sistema",
        "Gauss-Seidel",
        "Iterar usando lo nuevo al instante",
        "bi-arrow-down-up",
        "#60a5fa",
        r"x^{(k+1)}=(D+L)^{-1}(b-Ux^{(k)})",
        (
            "También despeja variable por variable, pero reutiliza inmediatamente los "
            "valores nuevos. Eso suele acelerar la convergencia respecto a Jacobi."
        ),
        r"Converge si \( \rho(T_{GS})<1 \); diagonal dominante o SPD son condiciones favorables.",
        "Iterativo lineal; normalmente requiere menos iteraciones que Jacobi.",
        [
            "Aprovecha los valores recién calculados.",
            "Suele converger más rápido que Jacobi.",
            "Ideal para comparar radio espectral y diagonal dominante.",
        ],
        [
            "No se paraleliza tan fácil como Jacobi.",
            "El orden de las ecuaciones puede afectar la convergencia.",
            "Sin condiciones adecuadas puede divergir.",
        ],
        [
            "Separar A en D, L y U.",
            "Despejar la primera variable.",
            "Usar ese valor nuevo para calcular la segunda.",
            "Continuar secuencialmente.",
            "Comparar la nueva aproximación con la anterior.",
        ],
        _demo(
            "iterative-linear",
            "x(k+1) usa datos nuevos en la misma vuelta",
            5,
            [
                "Se parte de una semilla.",
                "La primera componente se actualiza.",
                "La segunda ya usa la primera nueva.",
                "La tercera usa las anteriores nuevas.",
                "El error se mide contra la vuelta anterior.",
                "La convergencia depende del radio espectral.",
            ],
        ),
        "/gauss_seidel",
    ),
    _metodo(
        "newton_sistemas",
        "sistema",
        "Newton para Sistemas",
        "Linealizar varias ecuaciones",
        "bi-diagram-3",
        "#fb7185",
        (
            r"\begin{pmatrix}x_{i+1}\\y_{i+1}\end{pmatrix}"
            r"=\begin{pmatrix}x_i\\y_i\end{pmatrix}"
            r"-[J(x_i,y_i)]^{-1}"
            r"\begin{pmatrix}f_1(x_i,y_i)\\f_2(x_i,y_i)\end{pmatrix}"
        ),
        (
            "Extiende Newton a varias variables. En cada iteración evalúa el vector F, "
            "arma la jacobiana y resuelve un sistema lineal para corregir la aproximación."
        ),
        "Necesita jacobiana definida y no singular cerca de la solución.",
        "Cuadrática cerca de una solución regular y con buena semilla.",
        [
            "Muy potente para sistemas no lineales.",
            "La jacobiana muestra cómo interactúan las variables.",
            r"La forma matricial \( X_{i+1}=X_i-J(X_i)^{-1}F(X_i) \) conecta no linealidad con álgebra lineal.",
        ],
        [
            "Puede fallar si la jacobiana es singular.",
            "Es sensible a la semilla inicial.",
            "Resolver un sistema lineal en cada iteración puede ser costoso.",
        ],
        [
            r"Definir \( F(x)=0 \).",
            "Elegir el vector inicial.",
            "Evaluar F en la semilla.",
            "Evaluar la matriz jacobiana.",
            r"Calcular la corrección con \( J(X_i)^{-1}F(X_i) \) y restarla al vector actual.",
        ],
        _demo(
            "newton-system",
            "F(x,y)=0 con corrección vectorial",
            5,
            [
                "Se inicia en un punto del plano.",
                "F mide cuánto falta para cumplir las ecuaciones.",
                "La jacobiana linealiza el sistema alrededor del punto.",
                "Se resuelve el sistema lineal de corrección.",
                "El punto se mueve hacia la intersección.",
                "Cerca de la solución el avance se acelera.",
            ],
        ),
        "/newton_sistemas",
    ),
    _metodo(
        "newton_diferencias",
        "datos",
        "Newton Dif. Divididas",
        "Interpolar construyendo tabla",
        "bi-list-columns",
        "#34d399",
        r"P_n(x)=a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+\cdots",
        (
            "Construye un polinomio interpolante usando una tabla triangular de diferencias "
            "divididas. Los coeficientes salen de la primera fila de esa tabla."
        ),
        "Requiere puntos con valores de x no repetidos.",
        "Interpola exactamente los puntos; el error depende de la suavidad y distribución de datos.",
        [
            "Permite agregar puntos sin rehacer todo desde cero.",
            "Muestra de dónde sale cada coeficiente.",
            "Es más incremental que Lagrange.",
        ],
        [
            "Con muchos puntos puede oscilar.",
            "Puntos x muy cercanos pueden causar inestabilidad.",
            "No extrapola de forma confiable.",
        ],
        [
            "Ordenar o registrar los puntos.",
            "Calcular diferencias divididas de primer orden.",
            "Seguir con órdenes superiores.",
            "Tomar la primera entrada de cada columna como coeficiente.",
            "Construir el polinomio acumulado.",
        ],
        _demo(
            "interpolation-newton",
            "Tabla triangular de diferencias",
            5,
            [
                "Los puntos base ocupan la primera columna.",
                "La columna Δ1 mide pendientes entre pares.",
                "La columna Δ2 mide cambios de pendiente.",
                "Los coeficientes se toman de la parte superior.",
                "Cada coeficiente agrega un factor acumulado.",
                "El polinomio pasa por todos los puntos.",
            ],
        ),
        "/newton_diferencias",
    ),
    _metodo(
        "lagrange",
        "datos",
        "Lagrange",
        "Sumar bases que valen 1 y 0",
        "bi-intersect",
        "#facc15",
        r"P(x)=\sum_{i=0}^{n} y_iL_i(x),\qquad L_i(x)=\prod_{j\ne i}\frac{x-x_j}{x_i-x_j}",
        (
            r"Crea una base \( L_i(x) \) para cada punto. Cada base vale 1 en su propio "
            r"xᵢ y 0 en los demás; al sumar \( y_iL_i(x) \), el polinomio pasa por todos."
        ),
        "Requiere x distintos. Cada base usa todos los puntos.",
        "Interpola exactamente; no es ideal para agregar puntos porque cambia toda la base.",
        [
            "La idea de las bases es muy clara.",
            "No necesita resolver un sistema si se usa la forma clásica.",
            "Sirve para demostrar interpolación polinómica.",
        ],
        [
            "Con muchos puntos puede aparecer fenómeno de Runge.",
            "Agregar un punto obliga a recalcular todas las bases.",
            "Puede generar expresiones largas.",
        ],
        [
            r"Construir una base \( L_i(x) \) por cada punto.",
            "Hacer que cada base valga 1 en su punto y 0 en los otros.",
            "Multiplicar cada base por su yᵢ.",
            "Sumar todos los términos.",
            "Simplificar el polinomio final.",
        ],
        _demo(
            "interpolation-lagrange",
            "Bases L₀, L₁, L₂ sumadas",
            5,
            [
                "Cada punto recibe una base propia.",
                "L₀ vale 1 en x₀ y 0 en los demás.",
                "L₁ hace lo mismo para x₁.",
                "Las bases se escalan por yᵢ.",
                "La suma reconstruye el polinomio.",
                "El resultado coincide exactamente con los puntos.",
            ],
        ),
        "/lagrange",
    ),
    _metodo(
        "trazadores_cubicos",
        "datos",
        "Trazadores Cúbicos",
        "Suavidad por tramos",
        "bi-bezier",
        "#38bdf8",
        r"S_i(x)=a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3",
        (
            "En vez de usar un solo polinomio enorme, usa un cúbico por intervalo. "
            "Los tramos se conectan cuidando continuidad de valor, pendiente y curvatura."
        ),
        "Requiere x ordenados y sin repetidos; para spline natural se fija curvatura cero en extremos.",
        "Interpola por tramos. Reduce oscilaciones frente a polinomios globales grandes.",
        [
            "Muy bueno para muchos puntos.",
            "Produce curvas suaves por partes.",
            "Evita polinomios de grado exagerado.",
        ],
        [
            "El armado algebraico es más largo.",
            "Fuera del rango de datos no conviene extrapolar.",
            "Condiciones de frontera distintas producen curvas distintas.",
        ],
        [
            "Ordenar los puntos.",
            "Asignar un cúbico a cada intervalo.",
            "Exigir que cada tramo pase por sus extremos.",
            "Igualar primeras y segundas derivadas en nodos internos.",
            "Resolver el sistema para los coeficientes.",
        ],
        _demo(
            "splines",
            "Cúbicos conectados con suavidad",
            5,
            [
                "Los datos se separan por intervalos.",
                "Cada intervalo recibe su propio cúbico.",
                "Los tramos se pegan en los nodos.",
                "La pendiente se mantiene continua.",
                "La curvatura también se suaviza.",
                "La curva final evita oscilaciones fuertes.",
            ],
        ),
        "/trazadores_cubicos",
    ),
    _metodo(
        "regresion_lineal",
        "datos",
        "Regresión Lineal",
        "Ajustar tendencia, no interpolar",
        "bi-graph-up-arrow",
        "#22d3ee",
        r"y=a+bx,\qquad b=\frac{n\sum xy-\sum x\sum y}{n\sum x^2-(\sum x)^2}",
        (
            "Busca la recta que mejor representa la tendencia de los datos. No tiene que pasar "
            "por todos los puntos; minimiza la suma de errores cuadrados."
        ),
        "Necesita variación en x; si todos los x son iguales, la pendiente no existe.",
        r"El ajuste se evalúa con residuos, error cuadrático y \( R^2 \).",
        [
            "Ideal para datos con ruido.",
            "Entrega pendiente, intercepto y medida de ajuste.",
            "Distingue tendencia de interpolación exacta.",
        ],
        [
            "No sirve si la relación es claramente curva.",
            "Outliers pueden mover mucho la recta.",
            "Un R² alto no prueba causalidad.",
        ],
        [
            r"Calcular sumas: \( \sum x, \sum y, \sum xy, \sum x^2 \).",
            "Obtener pendiente b.",
            "Obtener intercepto a.",
            r"Calcular predicciones \( \hat y \).",
            "Medir residuos y calidad del ajuste.",
        ],
        _demo(
            "regression",
            "Puntos con tendencia lineal",
            5,
            [
                "Los puntos no caen exactamente sobre una recta.",
                "Se propone una recta candidata.",
                "Cada residuo mide distancia vertical al ajuste.",
                "La recta final minimiza errores cuadrados.",
                "R² resume qué tanto explica la recta.",
                "El modelo puede predecir dentro del rango razonable.",
            ],
        ),
        "/regresion_lineal",
    ),
    _metodo(
        "trapecio",
        "calculo",
        "Regla del Trapecio",
        "Integración por segmentos rectos",
        "bi-border-style",
        "#2ee59d",
        r"I\approx\frac{h}{2}\left[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\right]",
        (
            "Aproxima el área bajo la curva dividiendo el intervalo en partes iguales. "
            "En cada subintervalo reemplaza la curva por una recta y forma un trapecio."
        ),
        "Necesita una función evaluable en todos los nodos del intervalo y n >= 1.",
        "Al aumentar n, h disminuye y la aproximación mejora si la función es suave.",
        [
            "Es muy fácil de aplicar y revisar a mano.",
            "Funciona con cualquier cantidad de subintervalos.",
            "La tabla de pesos 1, 2, 2, ..., 2, 1 es directa.",
        ],
        [
            "Puede ser menos preciso que Romberg si se necesita mucha exactitud.",
            "Si la función tiene discontinuidades, el área aproximada pierde sentido.",
            "Curvas muy pronunciadas requieren más subintervalos.",
        ],
        [
            r"Calcular \( h=(b-a)/n \).",
            r"Construir los nodos \( x_i=a+ih \).",
            r"Evaluar \( f(x_i) \) en todos los nodos.",
            "Multiplicar extremos por 1 e interiores por 2.",
            r"Aplicar \( I\approx h/2 \) por la suma ponderada.",
        ],
        _demo(
            "integration-trapezoid",
            "Área aproximada con trapecios",
            5,
            [
                "Se fija el intervalo [a,b].",
                "El intervalo se parte en subintervalos iguales.",
                "Cada par de nodos define un trapecio.",
                "Los extremos tienen peso 1.",
                "Los nodos interiores tienen peso 2.",
                "La suma ponderada produce el área aproximada.",
            ],
        ),
        "/trapecio",
    ),
    _metodo(
        "romberg",
        "calculo",
        "Integración de Romberg",
        "Trapecio mejorado con Richardson",
        "bi-layers-half",
        "#6ea8ff",
        r"R_{k,j}=R_{k,j-1}+\frac{R_{k,j-1}-R_{k-1,j-1}}{4^j-1}",
        (
            "Primero calcula trapecios cada vez más finos. Después usa extrapolación "
            "de Richardson para cancelar parte del error y acelerar la precisión."
        ),
        "Requiere función suave y evaluable en todos los nodos generados por n = 1, 2, 4, ...",
        "Con funciones suaves suele mejorar muy rápido al avanzar por la diagonal R(k,k).",
        [
            "Aprovecha el trapecio, pero lo refina de forma inteligente.",
            "Muestra una tabla triangular muy clara para estudiar.",
            "Puede lograr alta precisión con pocos niveles.",
        ],
        [
            "No conviene si la función tiene discontinuidades internas.",
            "Funciones muy oscilantes pueden necesitar más niveles.",
            "La tabla crece y puede volverse pesada si se piden demasiados niveles.",
        ],
        [
            r"Calcular \( R_{0,0} \) con trapecio simple.",
            r"Duplicar subintervalos y calcular \( R_{k,0}=T_{2^k} \).",
            r"Aplicar \( R_{k,j}=R_{k,j-1}+\frac{R_{k,j-1}-R_{k-1,j-1}}{4^j-1} \).",
            "Avanzar por la tabla triangular.",
            r"Tomar \( R_{m,m} \) como la mejor aproximación.",
        ],
        _demo(
            "integration-romberg",
            "Trapecios refinados + tabla R(k,j)",
            5,
            [
                "Se calcula el primer trapecio.",
                "Se duplica la cantidad de subintervalos.",
                "Se llena la primera columna R(k,0).",
                "Richardson combina estimaciones vecinas.",
                "La diagonal mejora la aproximación.",
                "El último R(k,k) se toma como resultado.",
            ],
        ),
        "/romberg",
    ),
    _metodo(
        "diferenciacion_numerica",
        "calculo",
        "Diferenciación Numérica",
        "Derivadas con diferencias finitas",
        "bi-graph-up",
        "#c792ff",
        r"f'(x_0)\approx\frac{f(x_0+h)-f(x_0-h)}{2h}",
        (
            "Aproxima la pendiente local usando valores cercanos a x0. "
            "No necesita derivar simbólicamente, solo evaluar la función."
        ),
        "Necesita h > 0 y que la función exista en los puntos requeridos por el esquema.",
        "El error depende de h: h grande aumenta truncamiento; h diminuto aumenta redondeo.",
        [
            "Sirve cuando solo se tienen evaluaciones de la función.",
            "La fórmula centrada suele mejorar la precisión.",
            "También permite aproximar segunda derivada.",
        ],
        [
            "Un h mal elegido puede empeorar el resultado.",
            "Cerca de discontinuidades no representa una derivada real.",
            "Si f(x0+h) y f(x0-h) son muy parecidos puede haber cancelación.",
        ],
        [
            "Elegir el punto x0 y el paso h.",
            "Seleccionar adelante, atrás, centrada o segunda derivada.",
            "Evaluar la función en los puntos vecinos.",
            "Multiplicar cada valor por su coeficiente.",
            "Dividir el numerador entre h, 2h o h² según el esquema.",
        ],
        _demo(
            "differentiation",
            "Pendiente local por puntos cercanos",
            5,
            [
                "Se elige el punto x0.",
                "Se marca un paso pequeño h.",
                "Se evalúan puntos vecinos.",
                "La recta secante aproxima la tangente.",
                "La fórmula centrada usa información de ambos lados.",
                "El resultado estima la derivada en x0.",
            ],
        ),
        "/diferenciacion_numerica",
    ),
]


def obtener_teoria_metodos():
    """Retorna todos los métodos disponibles para renderizar la teoría."""
    return METODOS_TEORIA


def obtener_familias_teoria():
    """Retorna las familias usadas por el filtro de la interfaz."""
    return FAMILIAS_TEORIA
