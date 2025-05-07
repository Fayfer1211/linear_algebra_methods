import numpy as np
import sys

# --- Función Principal del Algoritmo (Refinada) ---
def steepest_descent(A, b, x0, tol=1e-6, max_iter=1000, verbose=True, return_history=False):
    """
    Resuelve Ax = b usando Descenso Máximo. Optimizada para A simétrica y definida positiva.

    Args:
        A (np.ndarray): Matriz cuadrada (n x n).
        b (np.ndarray): Vector del lado derecho (n,).
        x0 (np.ndarray): Vector de estimación inicial (n,).
        tol (float): Tolerancia para convergencia (norma L2 del residual).
        max_iter (int): Número máximo de iteraciones permitidas.
        verbose (bool): Imprimir mensajes de estado y advertencias.
        return_history (bool): Si True, retorna también el historial de la norma del residual.

    Returns:
        tuple: Depende de `return_history`:
               Si False: (x, k, status) donde 'x' es la solución, 'k' las iteraciones, 'status' un string indicando el resultado.
               Si True: (x, k, status, history) donde 'history' es una lista de normas L2 del residual.
               Si no converge, x es la última aproximación calculada.
    """
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,) or x0.shape != (n,):
         raise ValueError("Las dimensiones internas de A, b, o x0 no coinciden.")

    # Tolerancia pequeña para comprobaciones numéricas (basada en el tipo de dato de A)
    TINY_EPS = np.finfo(A.dtype).eps * 10

    # Advertencia si A no es simétrica
    if not np.allclose(A, A.T):
        if verbose:
            print("\n[Advertencia]: La matriz A no parece ser simétrica (A != A^T).")
            print("             La convergencia o las propiedades de optimalidad pueden no cumplirse.")

    x = x0.copy()
    r = b - np.dot(A, x)
    history = []
    status = "Maximum Iterations Reached" # Estado por defecto
    iterations_done = 0

    for k in range(max_iter + 1): # Bucle hasta max_iter+1 para comprobar estado inicial y final
        residual_norm = np.linalg.norm(r)

        if return_history:
            history.append(residual_norm)

        # --- Comprobar condición de parada ---
        if residual_norm < tol:
            status = "Converged"
            iterations_done = k
            if verbose:
                print(f"\n{status} en {k} iteraciones (Norma Residual = {residual_norm:.4e} < {tol:.1e}).")
            break # Salir del bucle

        if k == max_iter: # Si hemos llegado al límite de iteraciones sin converger antes
            iterations_done = k
            # El status ya es "Maximum Iterations Reached" por defecto
            if verbose:
                 print(f"\n{status} después de {k} iteraciones (Norma Residual = {residual_norm:.4e}).")
            break # Salir del bucle

        # --- Si no paramos, calcular el siguiente paso ---
        if verbose and k == 0: # Imprimir estado inicial
             print(f"Iteración {k:4d}: Norma Residual = {residual_norm:.6e}")

        # Calcular Ar = A*r_k
        Ar = np.dot(A, r)

        # Calcular alpha_k = (r_k^T * r_k) / (r_k^T * A * r_k)
        r_dot_r = np.dot(r, r)  # Es residual_norm**2
        r_dot_Ar = np.dot(r, Ar) # Denominador de alpha

        # Comprobar denominador alpha (r^T*A*r)
        # Debe ser > 0 si A es definida positiva y r != 0
        if abs(r_dot_Ar) < TINY_EPS:
            status = "Stalled (r^T*A*r near zero)"
            iterations_done = k
            if verbose: print(f"\n[Error/Warning]: {status} en iteración {k}. A podría ser singular o no def. pos.")
            break

        if r_dot_Ar < 0:
            status = "Stalled (r^T*A*r < 0)"
            iterations_done = k
            if verbose: print(f"\n[Error]: {status} en iteración {k}. La matriz A NO es definida positiva.")
            break

        alpha = r_dot_r / r_dot_Ar

        # Actualizar solución: x_{k+1} = x_k + alpha_k * r_k
        x = x + alpha * r

        # Actualizar residual: r_{k+1} = r_k - alpha_k * A * r_k
        r = r - alpha * Ar

        if verbose and (k+1) % 10 == 0 and k < max_iter: # Imprimir progreso
             # Imprime la norma del *nuevo* residual que se comprobará en la siguiente iteración
             print(f"Iteración {k+1:4d}: Norma Residual = {np.linalg.norm(r):.6e}")


    # Retornar resultados
    if return_history:
        return x, iterations_done, status, history
    else:
        return x, iterations_done, status

# --- Funciones Auxiliares para Entrada de Datos (Sin cambios respecto a la versión anterior) ---
def input_dimension():
    """Pide al usuario la dimensión de la matriz."""
    while True:
        try:
            n_str = input("Ingrese la dimensión (entero > 0) de la matriz cuadrada A (n): ")
            n = int(n_str)
            if n > 0: return n
            else: print("Error: La dimensión debe ser un entero positivo.")
        except ValueError: print("Error: Entrada inválida. Ingrese un número entero.")
        except EOFError: print("\nEntrada cancelada."); sys.exit()

def input_matrix(n):
    """Pide al usuario los elementos de la matriz A."""
    print(f"\nIngrese los elementos de la matriz A ({n}x{n}), fila por fila.")
    print(f"Use espacios para separar números. Use '.' como separador decimal.")
    A_list = []
    for i in range(n):
        while True:
            try:
                row_str = input(f"  Fila {i+1} ({n} números): ")
                row = list(map(float, row_str.split()))
                if len(row) == n: A_list.append(row); break
                else: print(f"  Error: Se esperaban {n} elementos, ingresó {len(row)}. Intente de nuevo.")
            except ValueError: print("  Error: Ingrese solo números válidos separados por espacios.")
            except EOFError: print("\nEntrada cancelada."); sys.exit()
    return np.array(A_list)

def input_vector(n, vector_name):
    """Pide al usuario los elementos de un vector."""
    print(f"\nIngrese los {n} elementos del vector {vector_name}.")
    print(f"Use espacios para separar números. Use '.' como separador decimal.")
    while True:
        try:
            vec_str = input(f"  Vector {vector_name} ({n} números): ")
            vec = np.array(list(map(float, vec_str.split())))
            if len(vec) == n: return vec
            else: print(f"  Error: Se esperaban {n} elementos, ingresó {len(vec)}. Intente de nuevo.")
        except ValueError: print("  Error: Ingrese solo números válidos separados por espacios.")
        except EOFError: print("\nEntrada cancelada."); sys.exit()

def input_parameters():
    """Pide al usuario la tolerancia, iteraciones y si desea historial."""
    while True:
        try:
            tol_str = input(f"Tolerancia [ej: 1e-6, default]: ")
            if not tol_str: tol = 1e-6; break
            tol = float(tol_str)
            if tol > 0: break
            else: print("Error: Tolerancia > 0.")
        except ValueError: print("Error: Entrada inválida.")
        except EOFError: print("\nEntrada cancelada."); sys.exit()
    while True:
        try:
            max_iter_str = input(f"Máx. Iteraciones [ej: 1000, default]: ")
            if not max_iter_str: max_iter = 1000; break
            max_iter = int(max_iter_str)
            if max_iter > 0: break
            else: print("Error: Max. Iteraciones > 0.")
        except ValueError: print("Error: Entrada inválida.")
        except EOFError: print("\nEntrada cancelada."); sys.exit()
    while True:
        hist_str = input("Guardar historial de residuales? (s/n) [default: n]: ").lower()
        if not hist_str or hist_str == 'n': return tol, max_iter, False
        if hist_str == 's': return tol, max_iter, True
        print("Respuesta inválida. Ingrese 's' o 'n'.")

# --- Bloque Principal (Mejorado) ---
if __name__ == "__main__":
    print("="*55)
    print("--- Solución Numérica de Ax = b (Descenso Máximo) ---")
    print("="*55)

    try:
        # Obtener entradas
        n = input_dimension()
        A = input_matrix(n)
        b = input_vector(n, "b")
        x0 = input_vector(n, "x0 (inicial)")
        tol, max_iter, track_history = input_parameters()

        print("\n--- Confirmación de Datos ---" + "-"*25)
        print(f"A = \n{np.array2string(A, prefix='    ', precision=4, suppress_small=True)}")
        print(f"b = {np.array2string(b, precision=4, suppress_small=True)}")
        print(f"x0 = {np.array2string(x0, precision=4, suppress_small=True)}")
        print(f"Tolerancia = {tol:.1e}, Max Iter = {max_iter}, Historial = {track_history}")
        print("-" * 49)

        # Ejecutar
        print("\n--- Ejecutando Descenso Máximo ---")
        result = steepest_descent(A, b, x0, tol=tol, max_iter=max_iter,
                                   verbose=True, return_history=track_history)
        print("-" * 30)

        # Procesar resultados
        if track_history:
            solution, iterations, status, history = result
        else:
            solution, iterations, status = result
            history = None

        # Mostrar resultados
        print(f"\n--- Resultados ---")
        print(f"Estado Final: {status}")
        print(f"Iteraciones realizadas: {iterations}")
        print(f"Solución encontrada (x):")
        print(np.array2string(solution, prefix='   ', formatter={'float_kind':lambda f: f"{f:12.6f}"}))

        # Verificación
        if isinstance(solution, np.ndarray):
            print(f"\n--- Verificación ---")
            Ax = np.dot(A, solution)
            # print(f"A @ x ≈ {np.array2string(Ax, formatter={'float_kind':lambda f: f'{f:12.6f}'})}")
            # print(f"b     = {np.array2string(b, formatter={'float_kind':lambda f: f'{f:12.6f}'})}")
            residual_final = b - Ax
            print(f"Residual final (b - Ax):")
            print(np.array2string(residual_final, prefix='   ', formatter={'float_kind':lambda f: f"{f:12.6e}"}))
            print(f"Norma L2 del residual final: {np.linalg.norm(residual_final):.6e}")

        # Mostrar Historial
        if history:
            print("\n--- Historial de Norma Residual ---")
            display_step = max(1, (iterations + 1) // 15) # ~15 puntos + el último
            for i, norm_val in enumerate(history):
                # Imprimir el primero(i=0), el último(i=iterations_done), y cada display_step
                if i == 0 or i == iterations or (i > 0 and i % display_step == 0) :
                    print(f" Iteración {i:4d}: {norm_val:.6e}")
            # Asegurarse que el último valor (si no se imprimió ya) aparezca
            if iterations not in [0] and iterations % display_step != 0:
                 print(f" Iteración {iterations:4d}: {history[iterations]:.6e}")


        # Comparar con NumPy
        print("\n--- Comparación con np.linalg.solve ---")
        try:
            numpy_solution = np.linalg.solve(A, b)
            print("Solución de NumPy:")
            print(np.array2string(numpy_solution, prefix='   ', formatter={'float_kind':lambda f: f"{f:12.6f}"}))
            if isinstance(solution, np.ndarray):
                diff_norm = np.linalg.norm(solution - numpy_solution)
                print(f"Norma L2 de la diferencia |x - x_numpy|: {diff_norm:.6e}")
        except np.linalg.LinAlgError:
            print("No se pudo calcular la solución directa (np.linalg.solve): La matriz A puede ser singular.")
        except Exception as e:
             print(f"Error durante la comparación con NumPy: {e}")

    except ValueError as ve:
        print(f"\nERROR de Valor/Dimensión: {ve}")
    except Exception as e:
        print(f"\nERROR Inesperado: {e}")
        import traceback
        traceback.print_exc() # Imprimir traza para depuración
    finally:
        print("\n=== Fin del programa ===")