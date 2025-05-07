# -*- coding: utf-8 -*-
import numpy as np
import time # Para medir tiempo (opcional)
import math # Para usar funciones matemáticas en eval

# Configuración de impresión de NumPy para mejor legibilidad
np.set_printoptions(precision=6, suppress=True)

# ------------------------------
# Entrada de datos
# ------------------------------
def input_matrix_manual(require_square=False):
    """
    Solicita al usuario que ingrese manualmente una matriz A y, opcionalmente, un vector b.

    Args:
        require_square (bool): Si True, solo pide una matriz cuadrada A.
                               Si False, pide A y b para un sistema Ax=b.

    Returns:
        Si require_square=True: A (ndarray)
        Si require_square=False: A (ndarray), b (ndarray)
    """
    while True:
        try:
            n_rows_str = input("Ingrese el número de filas de A (n): ")
            n_rows = int(n_rows_str)
            if require_square:
                n_cols = n_rows
                print(f"La matriz será de tamaño {n_rows} x {n_rows}.")
            else:
                n_cols_str = input("Ingrese el número de columnas de A (m, usualmente n): ")
                n_cols = int(n_cols_str)

            if n_rows <= 0 or n_cols <= 0:
                print("Error: Las dimensiones deben ser enteros positivos.")
                continue
            break
        except ValueError:
            print("Error: Entrada inválida. Ingrese números enteros.")

    print(f"Ingrese la matriz A ({n_rows} x {n_cols}), una fila por línea, elementos separados por espacios:")
    A = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(n_rows):
        while True:
            try:
                row_input = input(f"Fila {i+1}: ").split()
                if len(row_input) != n_cols:
                    print(f"Error: La fila debe tener exactamente {n_cols} elementos. Intente de nuevo.")
                    continue
                A[i, :] = list(map(float, row_input))
                # Validar ceros en diagonal si es cuadrada
                if n_rows == n_cols and i < n_rows and abs(A[i, i]) < 1e-15:
                     print(f"Advertencia: El elemento diagonal A[{i+1},{i+1}] es cero o muy pequeño.")
                break # Fila válida ingresada
            except ValueError:
                print("Error: Entrada inválida. Ingrese solo números separados por espacios.")

    if require_square:
        return A

    # Si no se requiere cuadrada, pedir vector b
    print(f"Ingrese el vector b ({n_rows} elementos separados por espacios):")
    while True:
        try:
            b_input = input().split()
            if len(b_input) != n_rows:
                print(f"Error: El vector b debe tener exactamente {n_rows} elementos. Intente de nuevo.")
                continue
            b = np.array(list(map(float, b_input)))
            break # Vector b válido ingresado
        except ValueError:
            print("Error: Entrada inválida. Ingrese solo números separados por espacios.")

    return A, b

def input_matrix_by_rule(require_square=False):
    """
    Genera una matriz A y opcionalmente un vector b usando reglas de formación.
    ¡ADVERTENCIA! Usa eval(), lo que puede ser inseguro si se ingresan expresiones maliciosas.

    Args:
        require_square (bool): Si True, solo genera una matriz cuadrada A.
                               Si False, genera A y b para un sistema Ax=b.

    Returns:
        Si require_square=True: A (ndarray)
        Si require_square=False: A (ndarray), b (ndarray)
    """
    print("\n¡ADVERTENCIA! Esta función usa eval() para interpretar las reglas.")
    print("Esto puede ser un RIESGO DE SEGURIDAD si se ingresa código malicioso.")
    print("Ejecute solo expresiones matemáticas confiables de fuentes conocidas.")
    confirm = input("¿Desea continuar bajo su propio riesgo? (s/N): ").lower()
    if confirm != 's':
        raise ValueError("Generación por regla cancelada por el usuario.")

    while True:
        try:
            n_rows_str = input("Ingrese el número de filas de A (n): ")
            n_rows = int(n_rows_str)
            if require_square:
                n_cols = n_rows
                print(f"La matriz será de tamaño {n_rows} x {n_rows}.")
            else:
                n_cols_str = input("Ingrese el número de columnas de A (m, usualmente n): ")
                n_cols = int(n_cols_str)

            if n_rows <= 0 or n_cols <= 0:
                print("Error: Las dimensiones deben ser enteros positivos.")
                continue
            break
        except ValueError:
            print("Error: Entrada inválida. Ingrese números enteros.")

    # --- Regla para la Matriz A ---
    print("\nIngrese la regla de formación para los elementos A[i, j].")
    print("Use 'i' para fila (0..n-1), 'j' para columna (0..m-1).")
    print("Variables disponibles: i, j, n_rows, n_cols.")
    print("Funciones permitidas: np.*, math.*, sin, cos, tan, sqrt, exp, log, log10, pi, e, abs, pow, max, min.")
    print("Ej: 'i + j + 1', 'np.sin(i*np.pi/n_rows)', '1 / (i + j + 1)'")
    rule_A = input("Regla para A[i, j]: ")

    A = np.zeros((n_rows, n_cols), dtype=float)
    # Crear un entorno seguro para eval
    safe_globals = {"__builtins__": None}
    safe_locals = {
        'np': np, 'math': math, 'i': 0, 'j': 0, 'n_rows': n_rows, 'n_cols': n_cols,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'sqrt': math.sqrt,
        'exp': math.exp, 'log': math.log, 'log10': math.log10, 'pi': math.pi,
        'e': math.e, 'abs': abs, 'pow': pow, 'max': max, 'min': min,
    }

    print("Generando matriz A...")
    try:
        for i_idx in range(n_rows):
            for j_idx in range(n_cols):
                safe_locals['i'] = i_idx
                safe_locals['j'] = j_idx
                # Evaluar la regla en un entorno controlado
                A[i_idx, j_idx] = float(eval(rule_A, safe_globals, safe_locals))
        print("Matriz A generada:")
        print(A)
        # Validar ceros en diagonal si es cuadrada
        if n_rows == n_cols:
            for i_idx in range(n_rows):
                 if abs(A[i_idx, i_idx]) < 1e-15:
                     print(f"Advertencia: El elemento diagonal A[{i_idx+1},{i_idx+1}] generado es cero o muy pequeño.")

    except Exception as e:
        print(f"\nError al evaluar la regla para A[i, j]: {e}")
        print("Verifique la sintaxis y las variables/funciones usadas.")
        raise ValueError("Error en la regla de formación de A.") from e

    if require_square:
        return A

    # --- Regla para el Vector b ---
    print("\nIngrese la regla de formación para los elementos b[i].")
    print("Use 'i' para fila (0..n-1). Puede usar 'A' para referirse a la matriz generada.")
    print("Variables disponibles: i, A, n_rows, n_cols.")
    print("Ej: 'i * 2', 'np.cos(i)', 'sum(A[i,:])', 'np.dot(A[i,:], np.ones(n_cols))'")
    rule_b = input("Regla para b[i]: ")

    b = np.zeros(n_rows, dtype=float)
    print("Generando vector b...")
    try:
        # Añadir A al diccionario seguro por si la regla de b depende de A
        safe_locals['A'] = A
        for i_idx in range(n_rows):
            safe_locals['i'] = i_idx
            # Evaluar la regla en un entorno controlado
            b[i_idx] = float(eval(rule_b, safe_globals, safe_locals))
        print("Vector b generado:")
        print(b)
    except Exception as e:
        print(f"\nError al evaluar la regla para b[i]: {e}")
        print("Verifique la sintaxis y las variables usadas.")
        raise ValueError("Error en la regla de formación de b.") from e

    return A, b


# ------------------------------
# Validaciones y Análisis Matricial
# ------------------------------
def es_simetrica(A, tol=1e-8):
    """Verifica si la matriz A es simétrica."""
    if A.shape[0] != A.shape[1]: return False
    return np.allclose(A, A.T, atol=tol)

def es_definida_positiva(A):
    """Verifica si la matriz A es definida positiva."""
    if not es_simetrica(A): return False
    try:
        # Cholesky falla para matrices no definidas positivas
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def es_diagonalmente_dominante_estricta_filas(A):
    """Verifica dominancia diagonal estricta por filas."""
    if A.shape[0] != A.shape[1]: return False, {}
    n = A.shape[0]
    ops = {'compare': 0, 'add_sub': 0, 'abs': 0} # Contador solo para esta función
    for i in range(n):
        diag_element = abs(A[i, i])
        ops['abs'] += 1
        if diag_element < 1e-15: return False, ops # Diagonal no puede ser cero
        # Suma de valores absolutos de elementos fuera de la diagonal
        off_diag_sum = np.sum(np.abs(A[i, :i])) + np.sum(np.abs(A[i, i+1:]))
        ops['abs'] += (n - 1)
        ops['add_sub'] += (n - 2) if n > 1 else 0 # n-1 elementos sumados
        ops['compare'] += 1
        if diag_element <= off_diag_sum:
            return False, ops # No es estrictamente dominante
    return True, ops

def es_diagonalmente_dominante_estricta_columnas(A):
    """Verifica dominancia diagonal estricta por columnas."""
    if A.shape[0] != A.shape[1]: return False, {}
    n = A.shape[0]
    ops = {'compare': 0, 'add_sub': 0, 'abs': 0}
    for j in range(n):
        diag_element = abs(A[j, j])
        ops['abs'] += 1
        if diag_element < 1e-15: return False, ops
        # Suma de valores absolutos de elementos fuera de la diagonal en la columna j
        off_diag_sum = np.sum(np.abs(A[:j, j])) + np.sum(np.abs(A[j+1:, j]))
        ops['abs'] += (n - 1)
        ops['add_sub'] += (n - 2) if n > 1 else 0
        ops['compare'] += 1
        if diag_element <= off_diag_sum:
            return False, ops
    return True, ops

def check_convergence_conditions(A):
    """Verifica e imprime condiciones comunes de convergencia."""
    if A.shape[0] != A.shape[1]:
        print("La matriz no es cuadrada, no se aplican condiciones de convergencia estándar.")
        return False # Indica que no se pudo verificar

    print("\n--- Verificando Condiciones de Convergencia (Jacobi/Gauss-Seidel/SOR) ---")
    is_dd_filas, _ = es_diagonalmente_dominante_estricta_filas(A)
    is_dd_cols, _ = es_diagonalmente_dominante_estricta_columnas(A)
    is_spd = False
    # Solo verificar SPD si es cuadrada
    if A.shape[0] == A.shape[1]:
        is_spd = es_definida_positiva(A)

    conditions_met = False
    if is_dd_filas:
        print("- ES estrictamente diagonalmente dominante por filas (Garantiza convergencia J, GS, SOR(0<w<2)).")
        conditions_met = True
    else:
        print("- NO es estrictamente diagonalmente dominante por filas.")

    if is_dd_cols:
         print("- ES estrictamente diagonalmente dominante por columnas (Garantiza convergencia J, GS).")
         conditions_met = True # Es una condición fuerte
    else:
        print("- NO es estrictamente diagonalmente dominante por columnas.")

    if is_spd:
        print("- ES Simétrica Definida Positiva (Garantiza convergencia GS, SOR(0<w<2), CG).")
        conditions_met = True
    else:
        # No imprimir si ya sabemos que no es simétrica por la prueba SPD
        if es_simetrica(A):
             print("- ES Simétrica, pero NO Definida Positiva.")
        else:
             print("- NO es Simétrica Definida Positiva.")


    if not conditions_met:
         print("\n- Advertencia: No se cumplen las condiciones comunes (Dominancia Diagonal Estricta, SPD).")
         print("  La convergencia de los métodos iterativos no está garantizada, aunque podría ocurrir.")
    print("--------------------------------------------------------------------")
    return conditions_met # Devuelve True si al menos una condición común se cumple

def analyze_matrix(A):
    """Calcula e imprime autovalores, radio espectral y autovectores."""
    if A.shape[0] != A.shape[1]:
        print("Error: El análisis matricial requiere una matriz cuadrada.")
        return

    print("\n--- Análisis Matricial ---")
    try:
        start_time = time.time()
        # Usar eig para matrices generales (puede dar complejos)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        end_time = time.time()
        print(f"(Cálculo de autovalores/vectores realizado en {end_time - start_time:.4f} segundos)")

        print("\nAutovalores (ordenados por magnitud descendente):")
        # Ordenar por magnitud para identificar el radio espectral fácilmente
        eigenvalues_sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues_sorted = eigenvalues[eigenvalues_sorted_indices]
        eigenvectors_sorted = eigenvectors[:, eigenvalues_sorted_indices]

        max_magnitude = 0.0
        for i, val in enumerate(eigenvalues_sorted):
            magnitude = np.abs(val)
            if i == 0: max_magnitude = magnitude # El primero es el radio espectral
            # Imprimir con formato claro para complejos
            if abs(val.imag) > 1e-10: # Considerar complejo si la parte imaginaria no es despreciable
                 print(f"  λ_{i+1} = {val.real:.6f} {val.imag:+.6f}j  (Magnitud: {magnitude:.6f})")
            else:
                 print(f"  λ_{i+1} = {val.real:.6f}                  (Magnitud: {magnitude:.6f})")

        print(f"\nRadio Espectral (ρ(A) = max|λ_i|): {max_magnitude:.6f}")

        print("\nAutovectores (columnas correspondientes a los autovalores ordenados):")
        np.set_printoptions(precision=4, suppress=True) # Formato compacto
        print(eigenvectors_sorted)
        np.set_printoptions(suppress=False) # Restaurar configuración por defecto

    except np.linalg.LinAlgError as e:
        print(f"\nError al calcular autovalores/autovectores: {e}")
    print("------------------------")


# ===================================================
# MÉTODOS DIRECTOS (con conteo de operaciones)
# ===================================================
# Nota: El conteo de operaciones es una ESTIMACIÓN, especialmente para
#       operaciones vectorizadas de NumPy (@, dot, slicing).
#       Se enfoca en las operaciones aritméticas principales.
# ------------------------------
def gaussian_elimination(A, b):
    """Resuelve Ax=b con Eliminación Gaussiana y Pivoteo Parcial. Cuenta operaciones."""
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    # Trabajar con copia para no modificar originales
    Ab = np.concatenate((A.copy(), b.reshape(n, 1).copy()), axis=1)
    ops = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0}

    # Eliminación hacia adelante O(n^3)
    for i in range(n):
        # Pivoteo parcial: encontrar max en columna i desde fila i
        k = i + np.argmax(np.abs(Ab[i:, i]))
        ops['abs'] += (n - i)
        ops['compare'] += (n - i - 1) if (n - i > 1) else 0
        if i != k:
            Ab[[i, k]] = Ab[[k, i]] # Intercambio (sin costo aritmético directo)

        pivot = Ab[i, i]
        if abs(pivot) < 1e-15: raise ValueError(f"Matriz singular o numéricamente singular (pivote cercano a cero en columna {i+1}).")
        ops['abs'] += 1

        for j in range(i + 1, n): # Para cada fila debajo del pivote
            factor = Ab[j, i] / pivot
            ops['div'] += 1
            # Actualizar fila j: Ab[j, k] = Ab[j, k] - factor * Ab[i, k] para k=i..n
            # Operaciones: (n - i + 1) multiplicaciones y (n - i + 1) restas
            Ab[j, i:] -= factor * Ab[i, i:]
            ops['mul'] += (n - i + 1) # n-i cols en A + 1 col en b
            ops['add_sub'] += (n - i + 1)

    # Sustitución hacia atrás O(n^2)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        diag = Ab[i, i]
        if abs(diag) < 1e-15: raise ValueError("Matriz singular (elemento diagonal cero en forma escalonada).")
        ops['abs'] += 1
        # Suma: Ab[i, i+1]*x[i+1] + ... + Ab[i, n-1]*x[n-1]
        sum_ax = np.dot(Ab[i, i + 1:n], x[i + 1:n])
        num_terms = n - 1 - i
        if num_terms > 0:
            ops['mul'] += num_terms
            ops['add_sub'] += (num_terms - 1) if num_terms > 1 else 0 # Sumas entre términos

        # x[i] = (Ab[i, n] - sum_ax) / diag
        ops['add_sub'] += 1 # Resta
        ops['div'] += 1     # División
        x[i] = (Ab[i, n] - sum_ax) / diag

    return x, ops

def gauss_jordan_elimination(A, b):
    """Resuelve Ax=b con Gauss-Jordan y Pivoteo Parcial. Cuenta operaciones."""
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    Ab = np.concatenate((A.copy(), b.reshape(n, 1).copy()), axis=1)
    ops = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0}

    # Eliminación O(n^3)
    for i in range(n):
        # Pivoteo parcial
        k = i + np.argmax(np.abs(Ab[i:, i]))
        ops['abs'] += (n - i)
        ops['compare'] += (n - i - 1) if (n - i > 1) else 0
        if i != k:
            Ab[[i, k]] = Ab[[k, i]]

        pivot = Ab[i, i]
        if abs(pivot) < 1e-15: raise ValueError(f"Matriz singular o numéricamente singular (pivote cercano a cero en columna {i+1}).")
        ops['abs'] += 1

        # Normalizar fila pivote: Ab[i, k] /= pivot para k=i..n
        Ab[i, i:] /= pivot
        ops['div'] += (n - i + 1) # n-i cols en A + 1 col en b

        # Eliminar otros elementos en la columna pivote (arriba y abajo)
        for j in range(n):
            if i != j:
                factor = Ab[j, i]
                # Actualizar fila j: Ab[j, k] = Ab[j, k] - factor * Ab[i, k] para k=i..n
                Ab[j, i:] -= factor * Ab[i, i:]
                ops['mul'] += (n - i + 1)
                ops['add_sub'] += (n - i + 1)

    # La solución está en la última columna
    return Ab[:, n], ops

def doolittle_factorization(A):
    """Factorización LU (Doolittle) sin pivoteo. Cuenta operaciones."""
    n = A.shape[0]
    if n != A.shape[1]: raise ValueError("La matriz debe ser cuadrada para LU.")
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    A = A.astype(float) # Trabajar con floats
    ops = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0}

    # Costo O(n^3)
    for i in range(n):
        # Calcular U (fila i)
        for k in range(i, n): # Columnas desde i hasta n-1
            # sum_val = L[i,0]*U[0,k] + ... + L[i,i-1]*U[i-1,k]
            sum_val = np.dot(L[i, :i], U[:i, k])
            if i > 0:
                ops['mul'] += i
                ops['add_sub'] += (i - 1) if i > 1 else 0
            ops['add_sub'] += 1 # Resta A[i,k] - sum_val
            U[i, k] = A[i, k] - sum_val

        # Calcular L (columna i)
        L[i, i] = 1.0
        pivot_u = U[i, i]
        if abs(pivot_u) < 1e-15:
             raise ValueError(f"Factorización LU (Doolittle) falló (pivote U[{i},{i}] cero). Considere usar pivoteo.")
        ops['abs'] += 1
        ops['compare'] +=1 # Comparación implícita con cero

        for k in range(i + 1, n): # Filas debajo de i
            # sum_val = L[k,0]*U[0,i] + ... + L[k,i-1]*U[i-1,i]
            sum_val = np.dot(L[k, :i], U[:i, i])
            if i > 0:
                ops['mul'] += i
                ops['add_sub'] += (i - 1) if i > 1 else 0
            ops['add_sub'] += 1 # Resta A[k,i] - sum_val
            ops['div'] += 1     # División por pivot_u
            L[k, i] = (A[k, i] - sum_val) / pivot_u

    return L, U, ops

def lu_solve(L, U, b):
    """Resuelve LUx=b. Cuenta operaciones de la sustitución O(n^2)."""
    n = len(b)
    if L.shape[0] != n or L.shape[1] != n or U.shape[0] != n or U.shape[1] != n:
         raise ValueError("Dimensiones inconsistentes para L, U, b.")
    y = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    b = b.astype(float)
    ops = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0}

    # 1. Sustitución hacia adelante (Ly = b)
    for i in range(n):
        # sum_ly = L[i,0]*y[0] + ... + L[i,i-1]*y[i-1]
        sum_ly = np.dot(L[i, :i], y[:i])
        if i > 0:
            ops['mul'] += i
            ops['add_sub'] += (i - 1) if i > 1 else 0
        diag_l = L[i, i]
        if abs(diag_l) < 1e-15: raise ValueError(f"Matriz L singular (elemento L[{i},{i}] cero).")
        ops['abs'] += 1
        ops['compare'] +=1
        # y[i] = (b[i] - sum_ly) / diag_l
        ops['add_sub'] += 1 # Resta
        ops['div'] += 1     # División (si diag_l no es 1)
        y[i] = (b[i] - sum_ly) / diag_l

    # 2. Sustitución hacia atrás (Ux = y)
    for i in range(n - 1, -1, -1):
        # sum_ux = U[i,i+1]*x[i+1] + ... + U[i,n-1]*x[n-1]
        sum_ux = np.dot(U[i, i + 1:], x[i + 1:])
        num_terms = n - 1 - i
        if num_terms > 0:
            ops['mul'] += num_terms
            ops['add_sub'] += (num_terms - 1) if num_terms > 1 else 0
        diag_u = U[i, i]
        if abs(diag_u) < 1e-15: raise ValueError(f"Matriz U singular (elemento U[{i},{i}] cero).")
        ops['abs'] += 1
        ops['compare'] +=1
        # x[i] = (y[i] - sum_ux) / diag_u
        ops['add_sub'] += 1 # Resta
        ops['div'] += 1     # División
        x[i] = (y[i] - sum_ux) / diag_u
    return x, ops

def crout_factorization(A):
    """Factorización LU (Crout) sin pivoteo. Cuenta operaciones."""
    n = A.shape[0]
    if n != A.shape[1]: raise ValueError("La matriz debe ser cuadrada para LU.")
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    A = A.astype(float)
    ops = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0}

    # Costo O(n^3)
    for j in range(n): # Iterar por columnas
        # Calcular L (columna j)
        for i in range(j, n): # Filas desde j hasta n-1
            # sum_val = L[i,0]*U[0,j] + ... + L[i,j-1]*U[j-1,j]
            sum_val = np.dot(L[i, :j], U[:j, j])
            if j > 0:
                ops['mul'] += j
                ops['add_sub'] += (j - 1) if j > 1 else 0
            ops['add_sub'] += 1 # Resta A[i,j] - sum_val
            L[i, j] = A[i, j] - sum_val

        # Calcular U (fila j)
        U[j, j] = 1.0 # Diagonal de U es 1
        pivot_l = L[j, j]
        if abs(pivot_l) < 1e-15:
            raise ValueError(f"Factorización LU (Crout) falló (pivote L[{j},{j}] cero).")
        ops['abs'] += 1
        ops['compare'] += 1

        for i in range(j + 1, n): # Columnas a la derecha de j
            # sum_val = L[j,0]*U[0,i] + ... + L[j,j-1]*U[j-1,i]
            sum_val = np.dot(L[j, :j], U[:j, i])
            if j > 0:
                ops['mul'] += j
                ops['add_sub'] += (j - 1) if j > 1 else 0
            ops['add_sub'] += 1 # Resta A[j,i] - sum_val
            ops['div'] += 1     # División por pivot_l
            U[j, i] = (A[j, i] - sum_val) / pivot_l

    return L, U, ops

def cholesky_factorization(A):
    """Factorización de Cholesky (A=LL^T). Cuenta operaciones."""
    n = A.shape[0]
    # Verificación inicial (importante para Cholesky)
    if not es_simetrica(A):
         raise np.linalg.LinAlgError("La matriz no es simétrica. No se puede aplicar Cholesky.")

    L = np.zeros_like(A, dtype=float)
    A = A.astype(float)
    ops = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'sqrt': 0}

    # Costo O(n^3)
    for i in range(n):
        for j in range(i + 1): # Solo calcular la parte inferior, incluyendo diagonal
            # Suma L[i,k]*L[j,k] para k < j
            suma = np.dot(L[i, :j], L[j, :j])
            if j > 0:
                ops['mul'] += j
                ops['add_sub'] += (j - 1) if j > 1 else 0

            if i == j: # Elementos de la diagonal L[i,i]
                diag_val_squared = A[i, i] - suma
                ops['add_sub'] += 1
                # Comprobar si es positivo (numéricamente) antes de sqrt
                if diag_val_squared < 1e-15:
                    raise np.linalg.LinAlgError(f"Matriz no definida positiva (falló en L[{i},{i}]^2 = {diag_val_squared:.2e} <= 0).")
                ops['compare'] += 1
                L[i, j] = np.sqrt(diag_val_squared)
                ops['sqrt'] += 1
            else: # Elementos fuera de la diagonal L[i,j] con i > j
                pivot_l_jj = L[j, j] # Ya calculado
                # No debería ser cero si la matriz es SPD y el cálculo anterior fue correcto
                if abs(pivot_l_jj) < 1e-15:
                     raise ValueError(f"División por cero en Cholesky (L[{j},{j}] es cero).")
                ops['abs'] += 1
                ops['compare'] += 1
                # L[i, j] = (A[i, j] - suma) / pivot_l_jj
                ops['add_sub'] += 1 # Resta
                ops['div'] += 1     # División
                L[i, j] = (A[i, j] - suma) / pivot_l_jj
    return L, ops

def cholesky_solve(L, b):
    """Resuelve Ax=b usando Cholesky (Ly=b, L^Tx=y). Cuenta ops de sustitución O(n^2)."""
    # Reutiliza lu_solve ya que las operaciones son las mismas para
    # resolver sistemas triangulares inferiores y superiores.
    # Aquí U es L.T. L.T no tiene costo aritmético significativo.
    x, ops = lu_solve(L, L.T, b)
    return x, ops

# ===================================================
# MÉTODOS ITERATIVOS (con conteo de operaciones)
# ===================================================
# Nota: El conteo de operaciones por iteración es O(n^2).
#       El conteo total depende del número de iteraciones K: O(K*n^2).
# ------------------------------
def jacobi_method(A, b, x0=None, tol=1e-10, max_iter=1000, check_convergence=False):
    """Método de Jacobi. Cuenta operaciones."""
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    # Contadores totales
    ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}

    if check_convergence: # Controlado desde el menú
        check_convergence_conditions(A)

    # Bucle de iteraciones
    for k in range(max_iter):
        ops_iter = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0} # Ops de esta iteración
        x_new = np.zeros_like(x, dtype=float)
        for i in range(n):
            # s = sum(A[i,j]*x[j] for j != i)
            # Usar x de la iteración anterior (k) para calcular todos los x_new[i]
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            if n > 1:
                ops_iter['mul'] += (n - 1) # n-1 multiplicaciones
                ops_iter['add_sub'] += (n - 2) if n > 2 else 0 # n-2 sumas internas
                ops_iter['add_sub'] += 1 # Suma de los dos dot

            diag = A[i, i]
            if abs(diag) < 1e-15: raise ValueError(f"Elemento diagonal A[{i},{i}] cero o muy pequeño.")
            ops_iter['abs'] += 1
            ops_iter['compare'] += 1
            # x_new[i] = (b[i] - s) / diag
            ops_iter['add_sub'] += 1 # Resta
            ops_iter['div'] += 1     # División
            x_new[i] = (b[i] - s) / diag

        # Criterio de parada: ||x_new - x||_inf < tol
        diff_norm = np.linalg.norm(x_new - x, ord=np.inf)
        # Costo aproximado de la norma infinito:
        ops_iter['add_sub'] += n # n restas
        ops_iter['abs'] += n # n valores absolutos
        ops_iter['compare'] += (n-1) if n > 1 else 0 # n-1 comparaciones para max
        ops_iter['norm'] += 1 # Contar la llamada a la función norma
        ops_iter['compare'] += 1 # Comparación final con tol

        # Acumular operaciones de la iteración
        for key in ops_total: ops_total[key] += ops_iter[key]

        # Comprobar convergencia
        if diff_norm < tol:
            return x_new, k + 1, True, ops_total # Solución, iteraciones, convergió, ops

        # Actualizar x para la siguiente iteración
        x = x_new

    # Si sale del bucle, no convergió en max_iter
    return x, max_iter, False, ops_total # Última aproximación, iteraciones, no convergió, ops

def gauss_seidel_method(A, b, x0=None, tol=1e-10, max_iter=1000, check_convergence=False):
    """Método de Gauss-Seidel. Cuenta operaciones."""
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}

    if check_convergence:
        check_convergence_conditions(A)

    for k in range(max_iter):
        ops_iter = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
        x_old = x.copy() # Guardar x de la iteración anterior para la comprobación de convergencia
        for i in range(n):
            # s1 = sum(A[i,j]*x[j] for j < i) -> Usa x ya actualizado en esta iteración k+1
            s1 = np.dot(A[i, :i], x[:i])
            if i > 0:
                ops_iter['mul'] += i
                ops_iter['add_sub'] += (i - 1) if i > 1 else 0
            # s2 = sum(A[i,j]*x_old[j] for j > i) -> Usa x de la iteración anterior k
            num_terms_s2 = n - 1 - i
            s2 = np.dot(A[i, i+1:], x_old[i+1:]) # ¡Usa x_old aquí!
            if num_terms_s2 > 0:
                ops_iter['mul'] += num_terms_s2
                ops_iter['add_sub'] += (num_terms_s2 - 1) if num_terms_s2 > 1 else 0

            diag = A[i, i]
            if abs(diag) < 1e-15: raise ValueError(f"Elemento diagonal A[{i},{i}] cero o muy pequeño.")
            ops_iter['abs'] += 1
            ops_iter['compare'] += 1
            # x[i] = (b[i] - s1 - s2) / diag
            ops_iter['add_sub'] += 2 # Dos restas
            ops_iter['div'] += 1     # División
            x[i] = (b[i] - s1 - s2) / diag # Actualiza x[i] inmediatamente

        # Criterio de parada: ||x^(k+1) - x^(k)||_inf < tol
        diff_norm = np.linalg.norm(x - x_old, ord=np.inf)
        ops_iter['add_sub'] += n
        ops_iter['abs'] += n
        ops_iter['compare'] += (n-1) if n > 1 else 0
        ops_iter['norm'] += 1
        ops_iter['compare'] += 1

        for key in ops_total: ops_total[key] += ops_iter[key]

        if diff_norm < tol:
            return x, k + 1, True, ops_total

    return x, max_iter, False, ops_total

def sor_method(A, b, omega, x0=None, tol=1e-10, max_iter=1000, check_convergence=False):
    """Método SOR. Cuenta operaciones."""
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}

    if not (0 < omega < 2):
        print(f"Advertencia: omega={omega} está fuera del rango (0, 2) recomendado para convergencia.")
    if check_convergence:
        check_convergence_conditions(A)

    for k in range(max_iter):
        ops_iter = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
        x_old = x.copy()
        for i in range(n):
            # s1, s2 como en Gauss-Seidel
            s1 = np.dot(A[i, :i], x[:i]) # Usa x actualizado
            if i > 0:
                ops_iter['mul'] += i
                ops_iter['add_sub'] += (i - 1) if i > 1 else 0
            num_terms_s2 = n - 1 - i
            s2 = np.dot(A[i, i+1:], x_old[i+1:]) # Usa x_old
            if num_terms_s2 > 0:
                ops_iter['mul'] += num_terms_s2
                ops_iter['add_sub'] += (num_terms_s2 - 1) if num_terms_s2 > 1 else 0

            diag = A[i, i]
            if abs(diag) < 1e-15: raise ValueError(f"Elemento diagonal A[{i},{i}] cero o muy pequeño.")
            ops_iter['abs'] += 1
            ops_iter['compare'] += 1
            # x_gs = (b[i] - s1 - s2) / diag (Valor intermedio de Gauss-Seidel)
            ops_iter['add_sub'] += 2
            ops_iter['div'] += 1
            x_gs = (b[i] - s1 - s2) / diag

            # Aplicar relajación SOR: x[i] = (1 - omega) * x_old[i] + omega * x_gs
            ops_iter['add_sub'] += 1 # 1 - omega
            ops_iter['mul'] += 1     # (1-omega)*x_old[i]
            ops_iter['mul'] += 1     # omega*x_gs
            ops_iter['add_sub'] += 1 # Suma final
            x[i] = (1 - omega) * x_old[i] + omega * x_gs # Actualiza x[i]

        # Criterio de parada
        diff_norm = np.linalg.norm(x - x_old, ord=np.inf)
        ops_iter['add_sub'] += n
        ops_iter['abs'] += n
        ops_iter['compare'] += (n-1) if n > 1 else 0
        ops_iter['norm'] += 1
        ops_iter['compare'] += 1

        for key in ops_total: ops_total[key] += ops_iter[key]

        if diff_norm < tol:
            return x, k + 1, True, ops_total

    return x, max_iter, False, ops_total

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    """Método del Gradiente Conjugado. Cuenta operaciones."""
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    if not es_simetrica(A):
        print("Advertencia: La matriz A no es simétrica. CG podría no funcionar correctamente.")
        # La no SPD se detectará si pAp <= 0.

    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    # Contadores: add_sub, mul, div, compare, abs, norm, matmul (A@vector), dot (vector@vector)
    ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0, 'matmul': 0, 'dot': 0}

    # --- Costo Inicial ---
    # r = b - A @ x
    Ax = A @ x
    ops_total['matmul'] += 1
    ops_total['mul'] += n * n # Estimación matmul
    ops_total['add_sub'] += n * (n - 1) if n > 1 else 0 # Estimación matmul
    r = b - Ax
    ops_total['add_sub'] += n # Resta vectorial

    # Norma inicial y chequeo
    norm_r = np.linalg.norm(r)
    ops_total['norm'] += 1
    ops_total['mul'] += n # Estimación norma (n mul para cuadrados)
    ops_total['add_sub'] += (n-1) if n > 1 else 0 # Estimación norma (n-1 sumas)
    # La raíz cuadrada no se cuenta como operación aritmética estándar
    ops_total['compare'] += 1 # Comparación con tol
    if norm_r < tol:
        print("Gradiente Conjugado: La solución inicial ya cumple la tolerancia.")
        return x, 0, True, ops_total

    p = r.copy() # Copia vectorial (sin costo aritmético)
    rs_old = r @ r # Producto punto rTr
    ops_total['dot'] += 1
    ops_total['mul'] += n # Estimación dot
    ops_total['add_sub'] += (n-1) if n > 1 else 0 # Estimación dot

    # --- Bucle Principal de Iteraciones ---
    for k in range(max_iter):
        # Ap = A @ p
        Ap = A @ p
        ops_total['matmul'] += 1
        ops_total['mul'] += n * n
        ops_total['add_sub'] += n * (n - 1) if n > 1 else 0

        # pAp = p @ Ap
        pAp = p @ Ap
        ops_total['dot'] += 1
        ops_total['mul'] += n
        ops_total['add_sub'] += (n-1) if n > 1 else 0

        # alpha = rs_old / pAp
        if abs(pAp) < 1e-15: # Evitar división por cero y verificar definida positiva
             print(f"\nCG Error: Denominador pAp = {pAp:.2e} cercano a cero en iteración {k+1}.")
             if pAp <= 0: print("  La matriz podría no ser definida positiva.")
             # Devolver estado actual; las operaciones hasta este punto ya están contadas.
             return x, k + 1, False, ops_total
        ops_total['abs'] += 1
        ops_total['compare'] += 1
        alpha = rs_old / pAp
        ops_total['div'] += 1

        # x = x + alpha * p (Actualización de la solución)
        # alpha * p: n multiplicaciones
        # x + ... : n sumas
        x = x + alpha * p
        ops_total['mul'] += n
        ops_total['add_sub'] += n

        # r_new = r - alpha * Ap (Actualización eficiente del residuo)
        # alpha * Ap: n multiplicaciones
        # r - ... : n restas
        r_new = r - alpha * Ap
        ops_total['mul'] += n
        ops_total['add_sub'] += n

        # Criterio de parada: norm(r_new) < tol
        norm_r_new = np.linalg.norm(r_new)
        ops_total['norm'] += 1
        ops_total['mul'] += n
        ops_total['add_sub'] += (n-1) if n > 1 else 0
        ops_total['compare'] += 1
        if norm_r_new < tol:
            return x, k + 1, True, ops_total # Convergió

        # rs_new = r_new @ r_new (Para la siguiente iteración)
        rs_new = r_new @ r_new
        ops_total['dot'] += 1
        ops_total['mul'] += n
        ops_total['add_sub'] += (n-1) if n > 1 else 0

        # beta = rs_new / rs_old
        # Comprobar rs_old antes de dividir
        if abs(rs_old) < 1e-15:
             # Si rs_old es cero, r era cero, deberíamos haber parado antes. Salvaguarda.
             print(f"\nCG Advertencia: rs_old = {rs_old:.2e} cercano a cero en iteración {k+1} antes de calcular beta.")
             return x, k + 1, False, ops_total # Devolver estado actual
        ops_total['abs'] += 1
        ops_total['compare'] += 1
        beta = rs_new / rs_old
        ops_total['div'] += 1

        # p = r_new + beta * p (Nueva dirección de búsqueda)
        # beta * p : n multiplicaciones
        # r_new + ... : n sumas
        p = r_new + beta * p
        ops_total['mul'] += n
        ops_total['add_sub'] += n

        # Actualizar r y rs_old para la siguiente iteración
        r = r_new
        rs_old = rs_new
        # --- Fin de operaciones dentro del bucle ---

    # Si sale del bucle, no convergió en max_iter
    return x, max_iter, False, ops_total


# ===================================================
# FUNCIÓN PARA EL PROBLEMA ESPECÍFICO (n=14,15,16,17)
# ===================================================
def generate_matrix_problem(n):
    """
    Genera la matriz A y el vector b para el problema específico
    con a_ij = (1+i)^(j-1) y b_i = ((1+i)^n - 1)^i.
    Indices i, j van de 0 a n-1. Usa floats estándar.
    Retorna A, b. Puede contener NaN si hay overflow.
    """
    n = int(n)
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    n_float = float(n)

    # Usar float64 estándar, np.power maneja errores
    print(f"Generando con n={n} usando float64.")
    # Desactivar temporalmente advertencias de overflow/invalid para np.power
    # ya que las manejaremos explícitamente con NaN/inf checks.
    old_settings = np.seterr(over='ignore', invalid='ignore')

    for i in range(n):
        i_float = float(i)
        one_plus_i = 1.0 + i_float

        # Calcular b[i] = ((1+i)^n - 1)^i
        try:
            # Calcular (1+i)^n
            if one_plus_i == 1.0: # Caso i=0
                one_plus_i_pow_n = 1.0
            else:
                one_plus_i_pow_n = np.power(one_plus_i, n_float)

            base_b = one_plus_i_pow_n - 1.0

            # Manejar 0^0 = 1
            if i == 0: # b_0 = (1^n - 1)^0 = 0^0 = 1
                b[i] = 1.0
            else:
                b[i] = np.power(base_b, i_float)

            # Verificar si el resultado es inf o nan
            if not np.isfinite(b[i]):
                 print(f"Advertencia: Resultado no finito (NaN/Inf) para b[{i}].")
                 b[i] = np.nan # Marcar como NaN

        except Exception as e: # Captura más amplia por si acaso
             print(f"Error inesperado calculando b[{i}]: {e}. Estableciendo a NaN.")
             b[i] = np.nan

        # Calcular A[i, j] = (1+i)^(j-1)
        for j in range(n):
            j_float = float(j)
            exponent = j_float - 1.0
            try:
                # Manejar 1^x = 1
                if i == 0: # Caso i=0 -> 1+i = 1
                    A[i, j] = 1.0
                else:
                    A[i, j] = np.power(one_plus_i, exponent)

                # Verificar si el resultado es inf o nan
                if not np.isfinite(A[i, j]):
                    print(f"Advertencia: Resultado no finito (NaN/Inf) para A[{i},{j}].")
                    A[i, j] = np.nan # Marcar como NaN

            except Exception as e:
                 print(f"Error inesperado calculando A[{i},{j}]: {e}. Estableciendo a NaN.")
                 A[i, j] = np.nan

    # Restaurar configuración de errores de NumPy
    np.seterr(**old_settings)
    return A, b

def run_specific_problem(n_values):
    """
    Ejecuta la generación y resolución (usando Gauss) para la lista de n dada.
    """
    print("\n" + "="*30 + " EJECUCIÓN PROBLEMA ESPECÍFICO " + "="*30)
    print(f"Reglas: a_ij = (1+i)^(j-1), b_i = ((1+i)^n - 1)^i  (índices i,j desde 0)")
    print(f"Valores de n a probar: {n_values}")
    print("Usando Eliminación Gaussiana para resolver.")
    print("="*86)

    results = {}

    for n in n_values:
        print(f"\n--- Procesando n = {n} ---")
        status = "No procesado"
        solve_time = None
        gen_time = None
        solution = None
        ops = {}
        A, b = None, None # Inicializar A y b para este n

        try:
            # 1. Generar Matriz y Vector
            print("Generando A y b...")
            start_gen = time.time()
            A, b = generate_matrix_problem(n)
            end_gen = time.time()
            gen_time = end_gen - start_gen
            print(f"(Generación completada en {gen_time:.4f} segundos)")

            # Imprimir solo una parte si son muy grandes
            print("Matriz A (primeras 5x5 si n>5):")
            print(A[:min(n, 5), :min(n, 5)])
            print("\nVector b (primeros 5 si n>5):")
            print(b[:min(n, 5)])


            # Verificar si hay NaN en A o b ANTES de resolver
            if np.isnan(A).any() or np.isnan(b).any():
                print("\nError: Se generaron valores NaN en A o b. No se puede resolver.")
                status = "Error (NaN en generación)"
                # Guardar resultado parcial y continuar al siguiente n
                results[n] = {"status": status, "gen_time": gen_time, "solve_time": None, "solution": None, "ops": {}}
                print("-"*(20 + len(str(n))))
                continue

            # 2. Resolver usando Eliminación Gaussiana
            print("\nResolviendo con Eliminación Gaussiana...")
            start_solve = time.time()
            solution, ops = gaussian_elimination(A.copy(), b.copy()) # Usar copias
            end_solve = time.time()
            solve_time = end_solve - start_solve
            print(f"(Resolución completada en {solve_time:.4f} segundos)")
            status = "Resuelto"

            # Verificar si la solución contiene NaN (puede indicar problemas numéricos)
            if np.isnan(solution).any():
                 print("\nAdvertencia: La solución contiene valores NaN. Indica problemas numéricos severos.")
                 status = "Resuelto (con NaN)"


            # 3. Mostrar resultados
            print(f"\nSolución x para n={n} (primeros/últimos 5 elementos si n>10):")
            if n <= 10:
                 print(solution)
            else:
                 # Usar formato para manejar NaN si existen
                 np.set_printoptions(precision=6, suppress=True, nanstr='NaN')
                 print(f"  Primeros 5: {solution[:5]}")
                 print(f"  Últimos 5:  {solution[-5:]}")
                 np.set_printoptions() # Restaurar default

            print_ops(ops)


        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"\nError al procesar n={n} durante la resolución: {e}")
            status = f"Error en Solución ({type(e).__name__})"
        except Exception as e:
            print(f"\nOcurrió un error inesperado para n={n}: {e}")
            status = f"Error inesperado ({type(e).__name__})"
        finally:
            # Guardar resultados independientemente de si hubo error
             results[n] = {"status": status, "gen_time": gen_time, "solve_time": solve_time, "solution": solution, "ops": ops}
             print("-"*(20 + len(str(n))))


    print("\n" + "="*30 + " FIN EJECUCIÓN PROBLEMA ESPECÍFICO " + "="*30)
    # Imprimir resumen de resultados
    print("\nResumen de Resultados:")
    for n_res, res in results.items():
        time_str = f"{res['solve_time']:.4f}s" if res['solve_time'] is not None else "N/A"
        print(f" n={n_res}: Estado='{res['status']}', Tiempo Solución={time_str}")
    print("="*86)
    return results


# ===================================================
# FUNCION PRINCIPAL Y MENÚ
# ===================================================
def print_ops(ops):
    """Imprime el diccionario de operaciones de forma legible."""
    print("  Operaciones realizadas (estimación):")
    # Filtrar y ordenar claves para una presentación consistente
    keys_ordered = sorted([k for k, v in ops.items() if v > 0])
    if not keys_ordered:
        print("    (No se contaron operaciones significativas)")
        return
    for key in keys_ordered:
        # Formato especial para operaciones NumPy estimadas
        if key in ['matmul', 'dot', 'norm']:
             print(f"    - {key.capitalize()} (NumPy): {ops[key]} llamadas")
        else:
             print(f"    - {key.replace('_',' ').capitalize()}: {ops[key]}")
    print("    (Nota: El conteo para matmul/dot/norm es de llamadas; las ops aritméticas están incluidas en add_sub, mul, etc.)")


def main_menu():
    """Muestra el menú principal y maneja la selección del usuario."""
    A = None
    b = None
    L = None # Para LU Doolittle/Crout
    U = None # Para LU Doolittle/Crout
    L_chol = None # Para Cholesky

    while True:
        print("\n" + "="*30 + " MENÚ PRINCIPAL " + "="*30)
        print("1. Ingresar/Generar Matriz y Vector")
        print("2. Resolver Sistema Ax=b (Métodos Directos)")
        print("3. Resolver Sistema Ax=b (Métodos Iterativos)")
        print("4. Realizar Factorización (LU, Cholesky)")
        print("5. Analizar Matriz A (Autovalores, Radio Espectral, etc.)")
        print("6. Ejecutar Problema Específico (n=14,15,16,17)") # Nueva opción
        print("7. Salir") # Opción Salir ahora es 7
        print("="*78)

        choice = input("Seleccione una opción: ")

        # --- Opción 1: Ingresar/Generar Datos ---
        if choice == '1':
            print("\n--- Ingresar/Generar Matriz y Vector ---")
            print("  a) Ingresar Manualmente Sistema Ax=b")
            print("  b) Ingresar Manualmente Matriz A (para análisis)")
            print("  c) Generar Sistema Ax=b por Regla de Formación")
            print("  d) Generar Matriz A por Regla de Formación (para análisis)")
            sub_choice = input("Opción: ").lower()

            try:
                A_new, b_new = None, None # Variables temporales
                require_sq = False
                input_func = None

                if sub_choice == 'a':
                    input_func = input_matrix_manual
                    require_sq = False
                elif sub_choice == 'b':
                    input_func = input_matrix_manual
                    require_sq = True
                elif sub_choice == 'c':
                    input_func = input_matrix_by_rule
                    require_sq = False
                elif sub_choice == 'd':
                    input_func = input_matrix_by_rule
                    require_sq = True
                else:
                    print("Opción inválida.")
                    continue # Volver al menú principal

                # Llamar a la función de entrada seleccionada
                if require_sq:
                    A_new = input_func(require_square=True)
                    b_new = None
                else:
                    A_new, b_new = input_func(require_square=False)

                # Actualizar A y b globales si la entrada fue exitosa
                A = A_new
                b = b_new
                L, U, L_chol = None, None, None # Resetear factorizaciones previas
                print("\nDatos ingresados/generados correctamente.")
                if A is not None and b is not None and A.shape[0] != A.shape[1]:
                     print("Advertencia: La matriz A no es cuadrada. Algunos métodos no aplicarán.")

            except (ValueError, TypeError) as e: # Capturar errores de entrada/regla
                print(f"\nError durante la entrada/generación de datos: {e}")
            except Exception as e: # Capturar otros errores inesperados
                 print(f"\nOcurrió un error inesperado: {e}")


        # --- Opción 2: Métodos Directos ---
        elif choice == '2':
            if A is None or b is None:
                print("\nError: Primero debe ingresar/generar un sistema Ax=b (Opción 1a o 1c).")
                continue
            if A.shape[0] != A.shape[1]:
                 print("\nError: Los métodos directos implementados requieren una matriz A cuadrada.")
                 continue
            # Verificar si hay NaN antes de proceder
            if np.isnan(A).any() or np.isnan(b).any():
                 print("\nError: La matriz A o el vector b contienen NaN. No se puede resolver.")
                 continue

            print("\n--- Seleccione Método Directo ---")
            print("  a) Eliminación Gaussiana")
            print("  b) Eliminación Gauss-Jordan")
            print("  c) Usar Factorización LU (previamente calculada o Doolittle)")
            print("  d) Usar Factorización Cholesky (previamente calculada)")
            sub_choice = input("Opción: ").lower()

            try:
                start_time = time.time()
                ops_fact = {}
                ops_solve = {}
                x = None
                method_name = "Desconocido"
                executed = False # Flag para saber si se ejecutó un método

                if sub_choice == 'a':
                    x, ops_solve = gaussian_elimination(A, b) # Pasa copias internamente
                    method_name = "Eliminación Gaussiana"
                    executed = True
                elif sub_choice == 'b':
                    x, ops_solve = gauss_jordan_elimination(A, b) # Pasa copias internamente
                    method_name = "Gauss-Jordan"
                    executed = True
                elif sub_choice == 'c':
                    method_name = "LU Solve"
                    temp_L, temp_U = L, U # Usar las guardadas si existen
                    if temp_L is None or temp_U is None:
                         print("  Factorización LU no disponible. Calculando Doolittle...")
                         try:
                             temp_L, temp_U, ops_fact = doolittle_factorization(A.copy()) # Usar copia
                             L, U = temp_L, temp_U # Guardar si se calcula
                             print("  Factorización Doolittle realizada y guardada.")
                         except (ValueError, np.linalg.LinAlgError) as e_fact:
                              print(f"  Error en factorización Doolittle: {e_fact}")
                              continue
                    # Resolver usando temp_L y temp_U
                    x, ops_solve = lu_solve(temp_L, temp_U, b.copy()) # Usar copia de b
                    executed = True

                elif sub_choice == 'd':
                     method_name = "Cholesky Solve"
                     temp_L_chol = L_chol # Usar la guardada si existe
                     if temp_L_chol is None:
                         print("  Factorización Cholesky no disponible. Calculando...")
                         try:
                             # Cholesky requiere SPD, verificar antes o dejar que falle
                             temp_L_chol, ops_fact = cholesky_factorization(A.copy()) # Usar copia
                             L_chol = temp_L_chol # Guardar si se calcula
                             print("  Factorización Cholesky realizada y guardada.")
                         except (ValueError, np.linalg.LinAlgError) as e_fact:
                              print(f"  Error en factorización Cholesky: {e_fact}")
                              continue
                     # Resolver usando temp_L_chol
                     x, ops_solve = cholesky_solve(temp_L_chol, b.copy()) # Usar copia de b
                     executed = True
                else:
                    print("Opción inválida.")
                    continue

                # Mostrar resultados si se ejecutó un método
                if executed:
                    end_time = time.time()
                    ops = {k: ops_fact.get(k, 0) + ops_solve.get(k, 0) for k in set(ops_fact) | set(ops_solve)}
                    print(f"\n--- Resultado ({method_name}) ---")
                    # Verificar si la solución contiene NaN
                    if np.isnan(x).any():
                         print("\nAdvertencia: La solución contiene valores NaN. Indica problemas numéricos.")
                         status = "Resuelto (con NaN)"
                    # Imprimir solo parte de la solución si es muy larga
                    if x.size <= 20:
                        print(f"Solución x: {x}")
                    else:
                        np.set_printoptions(precision=6, suppress=True, nanstr='NaN')
                        print(f"Solución x (primeros/últimos 5): {x[:5]} ... {x[-5:]}")
                        np.set_printoptions() # Restaurar
                    print_ops(ops)
                    print(f"Tiempo de ejecución total: {end_time - start_time:.6f} segundos")
                    print("-"*(len(method_name) + 16))

            except (ValueError, np.linalg.LinAlgError) as e:
                # Capturar errores de los métodos de solución/factorización
                print(f"\nError durante la ejecución del método '{method_name}': {e}")
            except Exception as e:
                 print(f"\nOcurrió un error inesperado en Opción 2: {e}")


        # --- Opción 3: Métodos Iterativos ---
        elif choice == '3':
            if A is None or b is None:
                print("\nError: Primero debe ingresar/generar un sistema Ax=b (Opción 1a o 1c).")
                continue
            if A.shape[0] != A.shape[1]:
                 print("\nError: Los métodos iterativos implementados requieren una matriz A cuadrada.")
                 continue
            # Verificar si hay NaN antes de proceder
            if np.isnan(A).any() or np.isnan(b).any():
                 print("\nError: La matriz A o el vector b contienen NaN. No se puede resolver.")
                 continue

            # Verificar condiciones de convergencia ANTES de pedir el método
            check_convergence_conditions(A)

            print("\n--- Seleccione Método Iterativo ---")
            print("  a) Jacobi")
            print("  b) Gauss-Seidel")
            print("  c) SOR (Successive Over-Relaxation)")
            print("  d) Gradiente Conjugado (para A simétrica definida positiva)")
            sub_choice = input("Opción: ").lower()

            try:
                # Pedir parámetros comunes para métodos iterativos
                while True:
                    try:
                        tol_str = input("  Ingrese la tolerancia (ej: 1e-6): ")
                        tol = float(tol_str)
                        if tol <= 0: print("La tolerancia debe ser positiva."); continue
                        break
                    except ValueError: print("Entrada inválida.")
                while True:
                     try:
                        max_iter_str = input("  Ingrese el número máximo de iteraciones (ej: 100): ")
                        max_iter = int(max_iter_str)
                        if max_iter <= 0: print("Máx. iteraciones debe ser positivo."); continue
                        break
                     except ValueError: print("Entrada inválida.")

                x0_input = input(f"  Ingrese aprox. inicial x0 ({A.shape[0]} elems. sep. por espacios, [Enter] para ceros): ")
                x0 = None
                if x0_input.strip():
                    try:
                        x0 = np.array(list(map(float, x0_input.split())))
                        if len(x0) != A.shape[0]:
                            print(f"Error: Tamaño de x0 ({len(x0)}) != n ({A.shape[0]}). Usando ceros."); x0 = None
                    except ValueError: print("Error al leer x0. Usando ceros."); x0 = None

                start_time = time.time()
                result = None
                method_name = "Desconocido"

                # Usar copias de A y b para no modificarlas
                A_copy = A.copy()
                b_copy = b.copy()

                if sub_choice == 'a':
                    result = jacobi_method(A_copy, b_copy, x0, tol, max_iter, check_convergence=False)
                    method_name = "Jacobi"
                elif sub_choice == 'b':
                    result = gauss_seidel_method(A_copy, b_copy, x0, tol, max_iter, check_convergence=False)
                    method_name = "Gauss-Seidel"
                elif sub_choice == 'c':
                    while True:
                        try:
                            omega_str = input("  Ingrese el factor de relajación omega (0 < omega < 2): ")
                            omega = float(omega_str)
                            break
                        except ValueError: print("Entrada inválida.")
                    result = sor_method(A_copy, b_copy, omega, x0, tol, max_iter, check_convergence=False)
                    method_name = f"SOR (omega={omega})"
                elif sub_choice == 'd':
                     # La verificación SPD ya se hizo antes, pero CG puede fallar si no lo es
                     result = conjugate_gradient(A_copy, b_copy, x0, tol, max_iter)
                     method_name = "Gradiente Conjugado"
                else:
                    print("Opción inválida.")
                    continue

                # Desempaquetar resultado si el método fue válido
                if result:
                    x, iters, converged, ops = result
                    end_time = time.time()

                    print(f"\n--- Resultado ({method_name}) ---")
                    if converged:
                         # Imprimir solo parte de la solución si es muy larga
                        if x.size <= 20:
                            print(f"Solución x: {x} (Convergió en {iters} iteraciones)")
                        else:
                            np.set_printoptions(precision=6, suppress=True, nanstr='NaN')
                            print(f"Solución x (primeros/últimos 5): {x[:5]} ... {x[-5:]} (Convergió en {iters} iteraciones)")
                            np.set_printoptions() # Restaurar
                    else:
                         if x.size <= 20:
                            print(f"No convergió en {iters} iteraciones. Última aproximación x: {x}")
                         else:
                            np.set_printoptions(precision=6, suppress=True, nanstr='NaN')
                            print(f"No convergió en {iters} iteraciones. Última aprox. x (primeros/últimos 5): {x[:5]} ... {x[-5:]}")
                            np.set_printoptions() # Restaurar
                    # Verificar si la solución contiene NaN
                    if np.isnan(x).any():
                         print("\nAdvertencia: La solución contiene valores NaN. Indica problemas numéricos.")

                    print_ops(ops)
                    print(f"Tiempo de ejecución: {end_time - start_time:.6f} segundos")
                    print("-"*(len(method_name) + 16))

            except (ValueError, np.linalg.LinAlgError) as e:
                 print(f"\nError durante la ejecución del método iterativo '{method_name}': {e}")
            except Exception as e:
                 print(f"\nOcurrió un error inesperado en Opción 3: {e}")


        # --- Opción 4: Factorización ---
        elif choice == '4':
            if A is None:
                print("\nError: Primero debe ingresar/generar una matriz A (Opción 1).")
                continue
            if A.shape[0] != A.shape[1]:
                 print("\nError: Las factorizaciones implementadas requieren una matriz A cuadrada.")
                 continue
            # Verificar si hay NaN antes de proceder
            if np.isnan(A).any():
                 print("\nError: La matriz A contiene NaN. No se puede factorizar.")
                 continue

            print("\n--- Seleccione Factorización ---")
            print("  a) LU Doolittle")
            print("  b) LU Crout")
            print("  c) Cholesky (para A simétrica definida positiva)")
            sub_choice = input("Opción: ").lower()

            try:
                start_time = time.time()
                ops = {}
                method_name = "Desconocido"
                factorization_done = False
                temp_L, temp_U, temp_L_chol = None, None, None # Para guardar resultado local

                # Usar copia de A para no modificar la original
                A_copy_fact = A.copy()

                if sub_choice == 'a':
                    temp_L, temp_U, ops = doolittle_factorization(A_copy_fact)
                    method_name = "LU Doolittle"
                    print(f"\n--- Factorización ({method_name}) ---")
                    print("Matriz L:\n", temp_L)
                    print("Matriz U:\n", temp_U)
                    L, U = temp_L, temp_U # Guardar globalmente
                    L_chol = None # Invalidar Cholesky
                    factorization_done = True
                elif sub_choice == 'b':
                    temp_L, temp_U, ops = crout_factorization(A_copy_fact)
                    method_name = "LU Crout"
                    print(f"\n--- Factorización ({method_name}) ---")
                    print("Matriz L:\n", temp_L)
                    print("Matriz U:\n", temp_U)
                    L, U = temp_L, temp_U # Guardar globalmente
                    L_chol = None # Invalidar Cholesky
                    factorization_done = True
                elif sub_choice == 'c':
                     # Verificar SPD antes de intentar
                     if not es_definida_positiva(A_copy_fact):
                          print("\nError: La matriz no es Simétrica Definida Positiva. No se puede aplicar Cholesky.")
                          continue # Volver al menú
                     temp_L_chol, ops = cholesky_factorization(A_copy_fact)
                     method_name = "Cholesky"
                     print(f"\n--- Factorización ({method_name}) ---")
                     print("Matriz L (tal que A = LL^T):\n", temp_L_chol)
                     L_chol = temp_L_chol # Guardar globalmente
                     L, U = None, None # Invalidar LU
                     factorization_done = True
                else:
                    print("Opción inválida.")
                    continue

                # Mostrar resultados si la factorización se hizo
                if factorization_done:
                    end_time = time.time()
                    print_ops(ops)
                    print(f"Tiempo de ejecución: {end_time - start_time:.6f} segundos")
                    print("-"*(len(method_name) + 18))

            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"\nError durante la factorización '{method_name}': {e}")
            except Exception as e:
                 print(f"\nOcurrió un error inesperado en Opción 4: {e}")


        # --- Opción 5: Analizar Matriz ---
        elif choice == '5':
            if A is None:
                print("\nError: Primero debe ingresar/generar una matriz A (Opción 1).")
                continue
            if A.shape[0] != A.shape[1]:
                 print("\nError: El análisis matricial requiere una matriz A cuadrada.")
                 continue
            # Verificar si hay NaN antes de proceder
            if np.isnan(A).any():
                 print("\nError: La matriz A contiene NaN. No se puede analizar.")
                 continue
            analyze_matrix(A)

        # --- Opción 6: Ejecutar Problema Específico ---
        elif choice == '6':
            n_list = [14, 15, 16, 17]
            try:
                run_specific_problem(n_list)
            except Exception as e:
                 print(f"\nOcurrió un error inesperado durante la ejecución del problema específico: {e}")
            # Resetear A, b, L, U, L_chol globales después de la ejecución específica
            # para evitar usar accidentalmente estas matrices grandes en otras opciones.
            A, b, L, U, L_chol = None, None, None, None, None
            print("\n(Variables globales A, b y factorizaciones reseteadas)")


        # --- Opción 7: Salir ---
        elif choice == '7':
            print("\nSaliendo del programa.")
            break

        # --- Opción Inválida ---
        else:
            print("\nOpción no válida. Por favor, intente de nuevo.")


if __name__ == "__main__":
    # Mensaje de bienvenida opcional
    print("="*78)
    print(" Calculadora de Métodos Numéricos para Sistemas Lineales y Análisis Matricial")
    print("="*78)

    # Descomentar la siguiente línea para ejecutar directamente el problema específico al iniciar
    # run_specific_problem([14, 15, 16, 17])

    # Iniciar el menú principal
    main_menu()
