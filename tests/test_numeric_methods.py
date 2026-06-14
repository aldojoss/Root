import base64
import math
import unittest

import home


def _png_ok(datos):
    grafica = datos.get("grafica")
    if not grafica:
        return False
    raw = base64.b64decode(grafica)
    return raw.startswith(b"\x89PNG") and len(raw) > 1000


class NumericMethodsTest(unittest.TestCase):
    def assertSuccessWithGraph(self, datos):
        self.assertFalse(datos.get("error"), datos)
        self.assertTrue(_png_ok(datos))
        self.assertTrue(datos.get("resultados") or datos.get("grupos"))

    def test_closed_methods_converge_and_plot(self):
        biseccion = home.metodo_biseccion("x^2-4", 0, 3, 0.001, 80)
        falsa = home.metodo_falsa_posicion("x^2-4", 0, 3, 0.001, 80)

        self.assertSuccessWithGraph(biseccion)
        self.assertSuccessWithGraph(falsa)
        self.assertAlmostEqual(float(biseccion["raiz"]), 2.0, places=3)
        self.assertAlmostEqual(float(falsa["raiz"]), 2.0, places=3)

    def test_open_methods_converge_and_plot(self):
        newton = home.metodo_newton_raphson("x^2-4", 3, 0.001, 80)
        secante = home.metodo_secante("x^2-4", 0, 3, 0.001, 80)
        punto_fijo = home.metodo_punto_fijo("x^2-4", 3, 0.001, 80)

        self.assertSuccessWithGraph(newton)
        self.assertSuccessWithGraph(secante)
        self.assertSuccessWithGraph(punto_fijo)
        self.assertAlmostEqual(float(newton["raiz"]), 2.0, places=3)
        self.assertAlmostEqual(float(secante["raiz"]), 2.0, places=3)
        self.assertAlmostEqual(abs(float(punto_fijo["raiz"])), 2.0, places=3)

    def test_polynomial_methods_converge_and_plot(self):
        poly = "x^3-6*x^2+11*x-6"
        horner = home.metodo_horner(poly, 2)
        horner_newton = home.metodo_horner_newton(poly, 1.5, 0.001, 80)
        muller = home.metodo_muller("x^2-4", 0, 1, 3, 0.001, 80)
        bairstow = home.metodo_bairstow(poly, 0.5, -0.5, 0.001, 100)

        for datos in [horner, horner_newton, muller, bairstow]:
            self.assertSuccessWithGraph(datos)

        self.assertAlmostEqual(float(horner["raiz"]), 0.0, places=8)
        self.assertIn(round(float(horner_newton["raiz"])), {1, 2, 3})
        self.assertAlmostEqual(abs(float(muller["raiz"])), 2.0, places=3)

        roots = []
        for grupo in bairstow["grupos"]:
            roots.extend(round(root.real) for root in grupo["raices_complejas"])
        self.assertEqual(sorted(roots), [1, 2, 3])

    def test_taylor_approximates_exp_and_plots(self):
        datos = home.metodo_taylor("e^x", 0, 1, 12)

        self.assertSuccessWithGraph(datos)
        self.assertAlmostEqual(float(datos["aprox_final"]), math.e, places=7)

    def test_taylor_keeps_small_nonzero_terms(self):
        datos = home.metodo_taylor(r"\sqrt{x}", 25, 36, 10)

        self.assertSuccessWithGraph(datos)
        self.assertEqual(len(datos["resultados"]), 10)
        self.assertAlmostEqual(float(datos["aprox_final"]), 6.0, places=4)

    def test_destructive_scenarios_are_controlled(self):
        punto_complejo = home.metodo_punto_fijo("x^2+x+1", 0, 0.01, 50)
        punto_suave = home.metodo_punto_fijo("e^{-x}-x", 0, 0.01, 80)
        muller_lineal = home.metodo_muller("2*x-4", 1, 2, 3, 0.01, 50)
        biseccion_asintota = home.metodo_biseccion(r"\tan(x)", 1, 2, 0.01, 80)
        falsa_asintota = home.metodo_falsa_posicion(r"\tan(x)", 1, 2, 0.01, 80)
        biseccion_log = home.metodo_biseccion(r"x*\log(x)-1", 1, 3, 0.01, 80)
        falsa_log = home.metodo_falsa_posicion(r"x*\log(x)-1", 1, 3, 0.01, 80)
        taylor_pesado = home.metodo_taylor(r"\tan(\sin(x^2))", 0, 1, 18)
        taylor_cos = home.metodo_taylor(r"\cos(x)", 0, 3.14159, 8)

        self.assertTrue(punto_complejo.get("error"))
        self.assertEqual(punto_complejo.get("titulo"), "🛑 Salto al plano complejo")

        self.assertSuccessWithGraph(punto_suave)
        self.assertAlmostEqual(float(punto_suave["raiz"]), 0.567143, places=3)

        self.assertTrue(muller_lineal.get("error"))
        self.assertEqual(muller_lineal.get("titulo"), "⚠️ Denominador Müller = 0")

        for datos in [biseccion_asintota, falsa_asintota]:
            self.assertTrue(datos.get("error"))
            self.assertEqual(datos.get("titulo"), "⚠️ Discontinuidad en el intervalo")

        self.assertSuccessWithGraph(biseccion_log)
        self.assertSuccessWithGraph(falsa_log)
        self.assertAlmostEqual(float(biseccion_log["raiz"]), 1.76322, places=3)
        self.assertAlmostEqual(float(falsa_log["raiz"]), 1.76322, places=3)

        self.assertSuccessWithGraph(taylor_pesado)
        self.assertEqual(len(taylor_pesado["resultados"]), 18)

        self.assertSuccessWithGraph(taylor_cos)
        self.assertAlmostEqual(float(taylor_cos["aprox_final"]), -1.0, places=3)

    def test_invalid_numeric_inputs_are_controlled_errors(self):
        cases = [
            home.metodo_biseccion("x^2-4", 0, 3, 0.01, 0),
            home.metodo_newton_raphson("x^2-4", 3, 0, 20),
            home.metodo_secante("x^2-4", 0, 3, 0.01, 2000),
            home.metodo_taylor("e^x", 0, 1, 3.5),
            home.metodo_horner("x^2-4", float("nan")),
            home.metodo_horner("x^2-4", 1e308),
            home.metodo_bairstow("x^3-6*x^2+11*x-6", 0.5, -0.5, 0.01, 0),
        ]

        for datos in cases:
            self.assertTrue(datos.get("error"), datos)
            self.assertEqual(datos.get("titulo"), "⚠️ Entrada inválida")

    def test_exact_roots_are_success_not_errors(self):
        cases = [
            home.metodo_biseccion("x^2-4", 2, 3, 0.01, 20),
            home.metodo_falsa_posicion("x^2-4", 2, 3, 0.01, 20),
            home.metodo_newton_raphson("x^2", 0, 0.01, 20),
            home.metodo_horner_newton("x^2", 0, 0.01, 20),
            home.metodo_muller("x^2-4", 2, 2, 2, 0.01, 20),
        ]

        for datos in cases:
            self.assertSuccessWithGraph(datos)

    def test_linear_system_methods_solve_sec(self):
        matrix = "2 1 -1 8\n-3 -1 2 -11\n-2 1 2 -3"
        expected = [2.0, 3.0, -1.0]

        for solver in [home.metodo_gauss, home.metodo_gauss_jordan, home.metodo_lu]:
            with self.subTest(solver=solver.__name__):
                datos = solver(matrix)
                self.assertFalse(datos.get("error"), datos)
                self.assertEqual(datos.get("tipo"), "sistema_lineal")
                self.assertTrue(datos.get("pasos"))
                for got, exp in zip(datos["solucion"], expected):
                    self.assertAlmostEqual(float(got), exp, places=6)

    def test_iterative_linear_system_methods_solve_sec(self):
        matrix = "10 -1 2 6\n-1 11 -1 22\n2 -1 10 -10"
        expected = [1.0, 2.0, -1.0]

        for solver in [home.metodo_jacobi, home.metodo_gauss_seidel]:
            with self.subTest(solver=solver.__name__):
                datos = solver(matrix, "0,0,0", 0.0001, 100)
                self.assertFalse(datos.get("error"), datos)
                self.assertEqual(datos.get("tipo"), "sistema_iterativo")
                self.assertTrue(datos.get("resultados"))
                self.assertTrue(_png_ok(datos))
                for got, exp in zip(datos["solucion"], expected):
                    self.assertAlmostEqual(float(got), exp, places=4)

    def test_linear_system_errors_are_controlled(self):
        malformed = home.metodo_gauss("1 2\n3 4")
        singular = home.metodo_gauss("1 1 2\n2 2 4")
        bad_initial = home.metodo_jacobi("10 1 1\n1 10 1", "0,0,0", 0.01, 20)
        divergent = home.metodo_gauss_seidel("1 2 3\n1 2 3", "0,0", 0.01, 20)

        self.assertTrue(malformed.get("error"))
        self.assertTrue(singular.get("error"))
        self.assertTrue(bad_initial.get("error"))
        self.assertTrue(divergent.get("error"))
        self.assertIn(singular.get("titulo"), {"⚠️ Sistema singular", "⚠️ Sustitución imposible"})

    def test_newton_systems_solves_and_rejects_singular_jacobian(self):
        datos = home.metodo_newton_sistemas(
            "x^2 + y^2 - 4\nx - y",
            "x,y",
            "1,1",
            0.0001,
            50,
        )

        self.assertFalse(datos.get("error"), datos)
        self.assertEqual(datos.get("tipo"), "newton_sistemas")
        self.assertTrue(_png_ok(datos))
        self.assertAlmostEqual(float(datos["solucion"][0]), math.sqrt(2), places=4)
        self.assertAlmostEqual(float(datos["solucion"][1]), math.sqrt(2), places=4)

        exact = home.metodo_newton_sistemas(
            "x^2\n y^2",
            "x,y",
            "0,0",
            0.0001,
            10,
        )
        self.assertFalse(exact.get("error"), exact)

        singular = home.metodo_newton_sistemas(
            "x^2 + 1\n y^2 + 1",
            "x,y",
            "0,0",
            0.0001,
            10,
        )
        self.assertTrue(singular.get("error"))
        self.assertEqual(singular.get("titulo"), "⚠️ Jacobiano singular")

    def test_newton_systems_accepts_math_constants(self):
        casos = [
            "3*x - cos(x*y) - 0.5\n"
            "x^2 - 81*(y + 0.1)^2 + sin(z) + 1.06\n"
            "exp(-x*y) + 20*z + (10*pi - 3)/3",
            r"3x-\cos(xy)-0.5" "\n"
            r"x^2-81(y+0.1)^2+\sin(z)+1.06" "\n"
            r"e^{-xy}+20z+\frac{10\pi-3}{3}",
        ]
        for funciones in casos:
            with self.subTest(funciones=funciones):
                datos = home.metodo_newton_sistemas(
                    funciones,
                    "x,y,z",
                    "0.1,0.1,-0.1",
                    0.0005,
                    3,
                )

                self.assertFalse(datos.get("error"), datos)
                self.assertEqual(datos.get("tipo"), "newton_sistemas")

    def test_interpolation_and_regression_methods(self):
        puntos_cuadraticos = "0 1\n1 2\n2 5\n3 10"
        newton_diff = home.metodo_newton_diferencias(puntos_cuadraticos, "1.5")
        lagrange = home.metodo_lagrange(puntos_cuadraticos, "1.5")
        lagrange_vander = home.metodo_lagrange(puntos_cuadraticos, "1.5", "vandermonde")
        spline = home.metodo_trazadores_cubicos("0 0\n1 1\n2 2\n3 3", "1.5")
        regresion = home.metodo_regresion_lineal("0 1\n1 3\n2 5\n3 7", "4")

        for datos in [newton_diff, lagrange, lagrange_vander, spline, regresion]:
            self.assertSuccessWithGraph(datos)

        self.assertAlmostEqual(float(newton_diff["valor_eval"]), 3.25, places=6)
        self.assertAlmostEqual(float(lagrange["valor_eval"]), 3.25, places=6)
        self.assertEqual(lagrange["metodo_resolucion"], "lagrange")
        self.assertAlmostEqual(float(lagrange_vander["valor_eval"]), 3.25, places=6)
        self.assertEqual(lagrange_vander["metodo_resolucion"], "vandermonde")
        self.assertAlmostEqual(float(spline["valor_eval"]), 1.5, places=6)
        self.assertAlmostEqual(float(regresion["valor_eval"]), 9.0, places=6)
        self.assertAlmostEqual(float(regresion["r2"]), 1.0, places=8)

    def test_data_methods_control_bad_points(self):
        duplicate = home.metodo_newton_diferencias("0 1\n0 2", "1")
        spline_out = home.metodo_trazadores_cubicos("0 0\n1 1\n2 0", "3")
        vertical_regression = home.metodo_regresion_lineal("1 2\n1 3", "")

        self.assertTrue(duplicate.get("error"))
        self.assertTrue(spline_out.get("error"))
        self.assertTrue(vertical_regression.get("error"))


class FlaskRoutesTest(unittest.TestCase):
    def setUp(self):
        self.client = home.app.test_client()

    def test_get_routes_render(self):
        for path in [
            "/", "/biseccion", "/falsa_posicion", "/newton", "/secante",
            "/taylor", "/punto_fijo", "/horner", "/horner_newton",
            "/muller", "/bairstow", "/gauss", "/gauss_jordan", "/lu",
            "/jacobi", "/gauss_seidel", "/newton_sistemas",
            "/newton_diferencias", "/lagrange",
            "/trazadores_cubicos", "/regresion_lineal",
            "/analisis_inteligente",
            "/teoria_metodos",
        ]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertGreater(len(response.data), 1000)

    def test_linear_system_routes_use_matrix_editor(self):
        for path in ["/gauss", "/gauss_jordan", "/lu", "/jacobi", "/gauss_seidel"]:
            with self.subTest(path=path):
                response = self.client.get(path)
                html = response.data.decode("utf-8", errors="replace")
                self.assertIn("matrix-editor", html)
                self.assertIn("matrix-grid", html)
                self.assertIn('name="matriz"', html)
                self.assertNotIn("<textarea name=\"matriz\"", html)

    def test_valid_posts_render_results_without_server_errors(self):
        forms = {
            "/biseccion": {"ecuacion_latex": "x^2-4", "xl": "0", "xu": "3", "tol": "0.001", "max_iter": "80"},
            "/falsa_posicion": {"ecuacion_latex": "x^2-4", "xl": "0", "xu": "3", "tol": "0.001", "max_iter": "80"},
            "/newton": {"ecuacion_latex": "x^2-4", "x0": "3", "tol": "0.001", "max_iter": "80"},
            "/secante": {"ecuacion_latex": "x^2-4", "x0": "0", "x1": "3", "tol": "0.001", "max_iter": "80"},
            "/taylor": {"ecuacion_latex": "e^x", "x0": "0", "x_eval": "1", "n_terminos": "12"},
            "/punto_fijo": {"ecuacion_latex": "x^2-4", "x0": "3", "tol": "0.001", "max_iter": "80"},
            "/horner": {"ecuacion_latex": "x^3-6*x^2+11*x-6", "x0": "2"},
            "/horner_newton": {"ecuacion_latex": "x^3-6*x^2+11*x-6", "x0": "1.5", "tol": "0.001", "max_iter": "80"},
            "/muller": {"ecuacion_latex": "x^2-4", "x0": "0", "x1": "1", "x2": "3", "tol": "0.001", "max_iter": "80"},
            "/bairstow": {"ecuacion_latex": "x^3-6*x^2+11*x-6", "r0": "0.5", "s0": "-0.5", "tol": "0.001", "max_iter": "100"},
            "/gauss": {"matriz": "2 1 -1 8\n-3 -1 2 -11\n-2 1 2 -3"},
            "/gauss_jordan": {"matriz": "2 1 -1 8\n-3 -1 2 -11\n-2 1 2 -3"},
            "/lu": {"matriz": "2 1 -1 8\n-3 -1 2 -11\n-2 1 2 -3"},
            "/jacobi": {"matriz": "10 -1 2 6\n-1 11 -1 22\n2 -1 10 -10", "inicial": "0,0,0", "tol": "0.0001", "max_iter": "100"},
            "/gauss_seidel": {"matriz": "10 -1 2 6\n-1 11 -1 22\n2 -1 10 -10", "inicial": "0,0,0", "tol": "0.0001", "max_iter": "100"},
            "/newton_sistemas": {
                "funciones": "x^2+y^2-4\nx-y",
                "variables": "x,y",
                "inicial": "1,1",
                "tol": "0.0001",
                "max_iter": "50",
            },
            "/newton_diferencias": {"puntos": "0 1\n1 2\n2 5\n3 10", "x_eval": "1.5"},
            "/lagrange": {"puntos": "0 1\n1 2\n2 5\n3 10", "x_eval": "1.5", "metodo_resolucion": "vandermonde"},
            "/trazadores_cubicos": {"puntos": "0 0\n1 1\n2 2\n3 3", "x_eval": "1.5"},
            "/regresion_lineal": {"puntos": "0 1\n1 3\n2 5\n3 7", "x_eval": "4"},
        }

        for path, data in forms.items():
            with self.subTest(path=path):
                response = self.client.post(path, data=data)
                html = response.data.decode("utf-8", errors="replace")
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("Error inesperado", html)
                if path in {"/gauss", "/gauss_jordan", "/lu"}:
                    self.assertIn("Solución", html)
                else:
                    self.assertIn("data:image/png;base64", html)

    def test_bad_form_numbers_are_friendly_errors(self):
        response = self.client.post(
            "/newton",
            data={"ecuacion_latex": "x^2-4", "x0": "abc", "tol": "0.01", "max_iter": "20"},
        )
        html = response.data.decode("utf-8", errors="replace")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Entrada inválida", html)
        self.assertNotIn("Error inesperado", html)

    def test_interactive_theory_route_contains_all_project_methods(self):
        response = self.client.get("/teoria_metodos")
        html = response.data.decode("utf-8", errors="replace")

        self.assertEqual(response.status_code, 200)
        for text in [
            "Teoría Interactiva de Métodos",
            "Bisección",
            "Regla Falsa",
            "Newton-Raphson",
            "Secante",
            "Punto Fijo",
            "Series de Taylor",
            "Horner",
            "Horner-Newton",
            "Müller",
            "Bairstow",
            "Eliminación de Gauss",
            "Gauss-Jordan",
            "Factorización LU",
            "Jacobi",
            "Gauss-Seidel",
            "Newton para Sistemas",
            "Newton Dif. Divididas",
            "Lagrange",
            "Trazadores Cúbicos",
            "Regresión Lineal",
            "methodSearch",
            "methodDemoCanvas",
        ]:
            self.assertIn(text, html)
        self.assertNotIn("Error inesperado", html)

    def test_smart_analysis_posts_compare_methods(self):
        forms = {
            "raices": {
                "tipo_problema": "raices",
                "ecuacion_latex": "x^2-4",
                "xl": "0",
                "xu": "3",
                "x0": "3",
                "x1": "0",
                "raiz_tol": "0.001",
                "raiz_max_iter": "80",
            },
            "sistemas_lineales": {
                "tipo_problema": "sistemas_lineales",
                "matriz": "10 -1 2 6\n-1 11 -1 22\n2 -1 10 -10",
                "inicial": "0,0,0",
                "lineal_tol": "0.0001",
                "lineal_max_iter": "100",
            },
            "interpolacion": {
                "tipo_problema": "interpolacion",
                "puntos": "0 1\n1 2\n2 5\n3 10",
                "x_eval_datos": "1.5",
                "objetivo_interpolacion": "auto",
            },
            "regresion": {
                "tipo_problema": "regresion",
                "puntos": "0 1\n1 3\n2 5\n3 7",
                "x_eval_datos": "4",
            },
        }

        for problem, data in forms.items():
            with self.subTest(problem=problem):
                response = self.client.post("/analisis_inteligente", data=data)
                html = response.data.decode("utf-8", errors="replace")
                self.assertEqual(response.status_code, 200)
                self.assertIn("Comparación automática", html)
                self.assertIn("Diagnóstico general", html)
                self.assertIn(f'data-result-problem="{problem}"', html)
                self.assertNotIn("Error inesperado", html)

    def test_smart_analysis_handles_invalid_input(self):
        response = self.client.post(
            "/analisis_inteligente",
            data={
                "tipo_problema": "raices",
                "ecuacion_latex": "x^2-4",
                "xl": "abc",
                "xu": "3",
                "raiz_tol": "0.001",
                "raiz_max_iter": "80",
            },
        )
        html = response.data.decode("utf-8", errors="replace")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Entrada inválida", html)
        self.assertNotIn("Error inesperado", html)


if __name__ == "__main__":
    unittest.main()
