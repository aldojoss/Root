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


class FlaskRoutesTest(unittest.TestCase):
    def setUp(self):
        self.client = home.app.test_client()

    def test_get_routes_render(self):
        for path in [
            "/", "/biseccion", "/falsa_posicion", "/newton", "/secante",
            "/taylor", "/punto_fijo", "/horner", "/horner_newton",
            "/muller", "/bairstow",
        ]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertGreater(len(response.data), 1000)

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
        }

        for path, data in forms.items():
            with self.subTest(path=path):
                response = self.client.post(path, data=data)
                html = response.data.decode("utf-8", errors="replace")
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("Error inesperado", html)
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


if __name__ == "__main__":
    unittest.main()
