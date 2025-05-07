import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font as tkFont
import numpy as np
import sys
import time

# --- Importar los métodos del otro archivo (para métodos directos y validaciones) ---
try:
    import solution_methods_linear_systems_equations as methods
except ImportError:
    print("Error: No se pudo encontrar el archivo 'solution_methods_linear_systems_equations.py'.")
    print("Asegúrate de que esté en el mismo directorio que este script.")
    print("Se usarán implementaciones básicas de validación si están disponibles.")


    class methods:
        @staticmethod
        def es_simetrica(A, tol=1e-8):
            if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]: return False
            return np.allclose(A, A.T, atol=tol)

        # Placeholders más robustos
        gaussian_elimination = lambda A, b: (None, {})
        gauss_jordan_elimination = lambda A, b: (None, {})
        doolittle_factorization = lambda A: (None, None, {})
        lu_solve = lambda L, U, b: (None, {})
        cholesky_factorization = lambda A: (None, {})
        cholesky_solve = lambda L, b: (None, {})
        run_specific_problem = lambda n_list: ({n: {"status": "Error (Módulo no importado)"} for n in n_list})
        # Los métodos iterativos se definen dentro de la clase GUI ahora


# --- Implementación del Descenso Pronunciado (con callback mejorado) ---
def steepest_descent(A, b, x0=None, tol=1e-10, max_iter=1000, callback=None):
    """Método del Descenso Pronunciado (Steepest Descent) con callback opcional."""
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n) para este método.")
    if not methods.es_simetrica(A):
        print(
            "Advertencia: La matriz A no es simétrica. El Descenso Pronunciado podría no converger o hacerlo lentamente.")

    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0, 'matmul': 0, 'dot': 0}

    Ax = A @ x
    ops_total['matmul'] += 1;
    ops_total['mul'] += n * n;
    ops_total['add_sub'] += n * (n - 1) if n > 1 else 0
    r = b - Ax
    ops_total['add_sub'] += n

    # Pasar alpha=None en la iteración 0
    if callback: callback(k=0, x=x, r=r, alpha=None)  # Pasar k=0 explícitamente

    for k in range(max_iter):
        norm_r = np.linalg.norm(r)
        ops_total['norm'] += 1;
        ops_total['mul'] += n;
        ops_total['add_sub'] += (n - 1) if n > 1 else 0
        ops_total['compare'] += 1
        if norm_r < tol:
            return x, k, True, ops_total  # Convergió en la iteración k (antes de calcular k+1)

        Ar = A @ r
        ops_total['matmul'] += 1;
        ops_total['mul'] += n * n;
        ops_total['add_sub'] += n * (n - 1) if n > 1 else 0

        rTr = r @ r
        ops_total['dot'] += 1;
        ops_total['mul'] += n;
        ops_total['add_sub'] += (n - 1) if n > 1 else 0

        rTAr = r @ Ar
        ops_total['dot'] += 1;
        ops_total['mul'] += n;
        ops_total['add_sub'] += (n - 1) if n > 1 else 0

        if abs(rTAr) < 1e-15:
            print(f"\nSteepest Descent Error: Denominador rTAr = {rTAr:.2e} cercano a cero en iteración {k + 1}.")
            if rTAr <= 0 and methods.es_simetrica(A):
                print("  Esto puede indicar que la matriz no es definida positiva.")
            # Retorna el estado de la iteración k, ya que no se pudo completar k+1
            return x, k, False, ops_total  # Indicar 0 iteraciones completadas si falla en la primera
        ops_total['abs'] += 1;
        ops_total['compare'] += 1
        alpha = rTr / rTAr  # Este es t_k
        ops_total['div'] += 1

        x_prev = x.copy()
        x = x + alpha * r
        ops_total['mul'] += n;
        ops_total['add_sub'] += n

        r = r - alpha * Ar
        ops_total['mul'] += n;
        ops_total['add_sub'] += n

        # Pasar alpha calculado en esta iteración (k+1)
        if callback: callback(k + 1, x, r, alpha=alpha)

    norm_r_final = np.linalg.norm(r)
    ops_total['norm'] += 1;
    ops_total['mul'] += n;
    ops_total['add_sub'] += (n - 1) if n > 1 else 0
    print(f"Steepest Descent no convergió en {max_iter} iteraciones. Norma residuo final: {norm_r_final:.4e}")
    return x, max_iter, False, ops_total


class AlgebraLinealTkinterGUI:
    def __init__(self, master):
        self.master = master
        master.title("Calculadora de Álgebra Lineal Profesional (Tkinter + ttk)")
        master.geometry("1100x900")
        master.configure(bg="#f0f2f5")

        # --- Variables de instancia ---
        self.A = None;
        self.b = None;
        self.vec_a = None;
        self.vec_b = None
        self.L = None;
        self.U = None;
        self.L_chol = None
        self.matrix_a_entries = None;
        self.matrix_b_entries = None
        self.vector_a_entries = None;
        self.vector_b_entries = None
        self.vector_x0_entries = None
        self.show_iterations_var = tk.BooleanVar(value=False)

        # --- Definición de Colores y Fuentes ---
        self.colors = {
            "bg_main": "#f0f2f5", "bg_frame": "#ffffff", "bg_entry": "#ffffff",
            "text_main": "#2c3e50", "text_label": "#34495e", "button_bg": "#3498db",
            "button_fg": "#ffffff", "button_hover_bg": "#2980b9", "clear_button_bg": "#e74c3c",
            "clear_button_hover_bg": "#c0392b", "result_bg": "#ecf0f1", "result_fg": "#2c3e50",
            "border": "#bdc3c7", "groupbox_title_fg": "#2980b9", "error_fg": "#c0392b",
            "iter_fg": "#555555"  # Color para texto de iteración
        }
        self.fonts = {
            "main": tkFont.Font(family="Helvetica", size=11),
            "label": tkFont.Font(family="Helvetica", size=10, weight="bold"),
            "entry": tkFont.Font(family="Helvetica", size=10),
            "button": tkFont.Font(family="Helvetica", size=9, weight="bold"),
            "result": tkFont.Font(family="Consolas", size=10),
            "group_title": tkFont.Font(family="Helvetica", size=13, weight="bold"),
            "entry_grid": tkFont.Font(family="Helvetica", size=9),
            "iter": tkFont.Font(family="Consolas", size=9)  # Fuente para iteraciones
        }

        # --- Estilo ttk ---
        style = ttk.Style()
        style.theme_use('clam')
        # ... (Configuraciones de estilo ttk idénticas) ...
        style.configure("TLabelFrame", background=self.colors["bg_frame"], padding=15, relief=tk.GROOVE, borderwidth=1)
        style.configure("TLabelFrame.Label", font=self.fonts["group_title"],
                        foreground=self.colors["groupbox_title_fg"], background=self.colors["bg_frame"])
        style.configure("TLabel", font=self.fonts["label"], foreground=self.colors["text_label"],
                        background=self.colors["bg_frame"], padding=(5, 5, 5, 2))
        style.configure("TEntry", font=self.fonts["entry"], padding=7, relief=tk.SOLID, borderwidth=1,
                        fieldbackground=self.colors["bg_entry"])
        style.map("TEntry", bordercolor=[('focus', self.colors["button_bg"]), ('!focus', self.colors["border"])])
        style.configure("Accent.TButton", font=self.fonts["button"], background=self.colors["button_bg"],
                        foreground=self.colors["button_fg"], padding=(10, 5), relief=tk.RAISED, borderwidth=1)
        style.map("Accent.TButton", background=[('active', self.colors["button_hover_bg"])])
        style.configure("Clear.TButton", font=self.fonts["button"], background=self.colors["clear_button_bg"],
                        foreground=self.colors["button_fg"], padding=(10, 5), relief=tk.RAISED, borderwidth=1)
        style.map("Clear.TButton", background=[('active', self.colors["clear_button_hover_bg"])])
        style.configure("MainBG.TFrame", background=self.colors["bg_main"])
        style.configure("FrameBG.TFrame", background=self.colors["bg_frame"])
        style.configure("TCheckbutton", background=self.colors["bg_frame"], font=self.fonts["label"],
                        foreground=self.colors["text_label"])

        # --- Scrollable Frame Setup ---
        main_app_frame = ttk.Frame(master, style="MainBG.TFrame")
        main_app_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main_app_frame, borderwidth=0, background=self.colors["bg_main"])
        vsb = ttk.Scrollbar(main_app_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = ttk.Frame(self.canvas, style="MainBG.TFrame")
        self.canvas_frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # --- Layout Principal ---
        main_content_frame = ttk.Frame(self.scrollable_frame, padding="20 20 20 20", style="MainBG.TFrame")
        main_content_frame.pack(fill=tk.BOTH, expand=True)

        # --- Contenedor para Entradas ---
        input_frame = ttk.Frame(main_content_frame, style="MainBG.TFrame")
        input_frame.pack(fill=tk.X, pady=(0, 15))

        # --- Columna Izquierda ---
        left_column_frame = ttk.Frame(input_frame, style="MainBG.TFrame")
        left_column_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 10))

        # --- Sección de Vectores (Entrada por Grid) ---
        vector_groupbox = ttk.LabelFrame(left_column_frame, text="Vectores (Entrada por Grid)", padding="15")
        vector_groupbox.pack(fill=tk.X, pady=(0, 15))
        vec_control_frame = ttk.Frame(vector_groupbox, style="FrameBG.TFrame")
        vec_control_frame.pack(fill=tk.X, pady=5)
        # Controles Vector A
        vec_a_dim_frame = ttk.Frame(vec_control_frame, style="FrameBG.TFrame")
        vec_a_dim_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(vec_a_dim_frame, text="Vector A - Tamaño:").pack(side=tk.LEFT, padx=5)
        self.vec_a_size_entry = ttk.Entry(vec_a_dim_frame, width=5, font=self.fonts["entry"])
        self.vec_a_size_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(vec_a_dim_frame, text="Generar Grid A", command=lambda: self._generate_grid('VA'),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.vector_a_grid_container = ttk.Frame(vec_control_frame, style="FrameBG.TFrame", borderwidth=1,
                                                 relief=tk.SUNKEN)
        self.vector_a_grid_container.pack(fill=tk.X, expand=True, pady=5, ipady=2)
        # Controles Vector B
        vec_b_dim_frame = ttk.Frame(vec_control_frame, style="FrameBG.TFrame")
        vec_b_dim_frame.pack(fill=tk.X, pady=(5, 5))
        ttk.Label(vec_b_dim_frame, text="Vector B (o b p/ Ax=b):").pack(side=tk.LEFT, padx=5)
        self.vec_b_size_entry = ttk.Entry(vec_b_dim_frame, width=5, font=self.fonts["entry"])
        self.vec_b_size_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(vec_b_dim_frame, text="Generar Grid B/b", command=lambda: self._generate_grid('VB'),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.vector_b_grid_container = ttk.Frame(vec_control_frame, style="FrameBG.TFrame", borderwidth=1,
                                                 relief=tk.SUNKEN)
        self.vector_b_grid_container.pack(fill=tk.X, expand=True, pady=5, ipady=2)
        # Escalar Vectores
        scalar_v_frame = ttk.Frame(vec_control_frame, style="FrameBG.TFrame")
        scalar_v_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(scalar_v_frame, text="Escalar (Vectores):").pack(side=tk.LEFT, padx=5)
        self.scalar_v_entry = ttk.Entry(scalar_v_frame, width=20, font=self.fonts["entry"])
        self.scalar_v_entry.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.W)

        # --- Sección Parámetros Métodos Iterativos ---
        iter_params_groupbox = ttk.LabelFrame(left_column_frame, text="Parámetros Métodos Iterativos", padding="15")
        iter_params_groupbox.pack(fill=tk.X)
        iter_grid = ttk.Frame(iter_params_groupbox, style="FrameBG.TFrame")
        iter_grid.pack(fill=tk.X)
        # Tolerancia y Max Iter
        ttk.Label(iter_grid, text="Tolerancia (ej: 1e-6):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.tol_entry = ttk.Entry(iter_grid, width=20, font=self.fonts["entry"])
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(iter_grid, text="Max. Iteraciones (ej: 1000):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_iter_entry = ttk.Entry(iter_grid, width=20, font=self.fonts["entry"])
        self.max_iter_entry.insert(0, "1000")
        self.max_iter_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        # Omega
        ttk.Label(iter_grid, text="Omega (SOR, 0<ω<2):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.omega_entry = ttk.Entry(iter_grid, width=20, font=self.fonts["entry"])
        self.omega_entry.insert(0, "1.1")
        self.omega_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        # --- Grid para x0 ---
        x0_frame = ttk.Frame(iter_grid, style="FrameBG.TFrame")
        x0_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=5)
        ttk.Label(x0_frame, text="Aprox. Inicial x0 - Tamaño:").pack(side=tk.LEFT, padx=5)
        self.vec_x0_size_entry = ttk.Entry(x0_frame, width=5, font=self.fonts["entry"])
        self.vec_x0_size_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(x0_frame, text="Generar Grid x0", command=lambda: self._generate_grid('VX0'),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.vector_x0_grid_container = ttk.Frame(iter_grid, style="FrameBG.TFrame", borderwidth=1, relief=tk.SUNKEN)
        self.vector_x0_grid_container.grid(row=4, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        # Checkbox
        self.show_iter_check = ttk.Checkbutton(iter_grid, text="Mostrar Iteraciones",
                                               variable=self.show_iterations_var, style="TCheckbutton")
        self.show_iter_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        iter_grid.columnconfigure(1, weight=1)

        # --- Columna Derecha ---
        right_column_frame = ttk.Frame(input_frame, style="MainBG.TFrame")
        right_column_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Sección de Matrices (Entrada por Grid) ---
        matrix_groupbox = ttk.LabelFrame(right_column_frame, text="Matrices (Entrada por Grid)", padding="15")
        matrix_groupbox.pack(fill=tk.BOTH, expand=True)
        mat_control_frame = ttk.Frame(matrix_groupbox, style="FrameBG.TFrame")
        mat_control_frame.pack(fill=tk.X, pady=5)
        # ... (Controles de Dimensiones y Grids sin cambios) ...
        mat_a_dim_frame = ttk.Frame(mat_control_frame, style="FrameBG.TFrame")
        mat_a_dim_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(mat_a_dim_frame, text="Matriz A - Dimensiones:").pack(side=tk.LEFT, padx=5)
        self.mat_a_rows_entry = ttk.Entry(mat_a_dim_frame, width=5, font=self.fonts["entry"])
        self.mat_a_rows_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(mat_a_dim_frame, text="x").pack(side=tk.LEFT, padx=2)
        self.mat_a_cols_entry = ttk.Entry(mat_a_dim_frame, width=5, font=self.fonts["entry"])
        self.mat_a_cols_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(mat_a_dim_frame, text="Generar Grid A", command=lambda: self._generate_grid('A'),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.matrix_a_grid_container = ttk.Frame(mat_control_frame, style="FrameBG.TFrame", borderwidth=1,
                                                 relief=tk.SUNKEN)
        self.matrix_a_grid_container.pack(fill=tk.BOTH, expand=True, pady=5, ipady=5)

        mat_b_dim_frame = ttk.Frame(mat_control_frame, style="FrameBG.TFrame")
        mat_b_dim_frame.pack(fill=tk.X, pady=(5, 5))
        ttk.Label(mat_b_dim_frame, text="Matriz B - Dimensiones:").pack(side=tk.LEFT, padx=5)
        self.mat_b_rows_entry = ttk.Entry(mat_b_dim_frame, width=5, font=self.fonts["entry"])
        self.mat_b_rows_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(mat_b_dim_frame, text="x").pack(side=tk.LEFT, padx=2)
        self.mat_b_cols_entry = ttk.Entry(mat_b_dim_frame, width=5, font=self.fonts["entry"])
        self.mat_b_cols_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(mat_b_dim_frame, text="Generar Grid B", command=lambda: self._generate_grid('B'),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        self.matrix_b_grid_container = ttk.Frame(mat_control_frame, style="FrameBG.TFrame", borderwidth=1,
                                                 relief=tk.SUNKEN)
        self.matrix_b_grid_container.pack(fill=tk.BOTH, expand=True, pady=5, ipady=5)

        scalar_m_frame = ttk.Frame(mat_control_frame, style="FrameBG.TFrame")
        scalar_m_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(scalar_m_frame, text="Escalar (Matrices):").pack(side=tk.LEFT, padx=5)
        self.scalar_m_entry = ttk.Entry(scalar_m_frame, width=20, font=self.fonts["entry"])
        self.scalar_m_entry.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.W)

        # --- Sección de Operaciones (Botones) ---
        ops_groupbox = ttk.LabelFrame(main_content_frame, text="Operaciones a Realizar", padding="15")
        ops_groupbox.pack(fill=tk.X, pady=15)
        ops_grid = ttk.Frame(ops_groupbox, style="FrameBG.TFrame")
        ops_grid.pack(fill=tk.X)
        # ... (Layout de botones sin cambios lógicos, solo textos) ...
        vec_ops_label = ttk.Label(ops_grid, text="Vectores:", font=self.fonts["label"])
        vec_ops_label.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(5, 2))
        vec_buttons_layout = ttk.Frame(ops_grid, style="FrameBG.TFrame")
        vec_buttons_layout.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(0, 10))
        vec_buttons_data = [
            ("A + B", lambda: self.perform_vector_op('add')), ("A - B", lambda: self.perform_vector_op('subtract')),
            ("Producto Punto", lambda: self.perform_vector_op('dot')),
            ("Producto Cruz", lambda: self.perform_vector_op('cross')),
            ("Escalar * A", lambda: self.perform_vector_op('scalar_mult_a')),
            ("Escalar * B", lambda: self.perform_vector_op('scalar_mult_b')),
            ("Magnitud A", lambda: self.perform_vector_op('magnitude_a')),
            ("Magnitud B", lambda: self.perform_vector_op('magnitude_b')),
        ]
        buttons_per_row_vec = 4
        for i, (text, command) in enumerate(vec_buttons_data):
            btn = ttk.Button(vec_buttons_layout, text=text, command=command, style="Accent.TButton", width=15)
            btn.grid(row=i // buttons_per_row_vec, column=i % buttons_per_row_vec, padx=4, pady=4, sticky=tk.EW)
            vec_buttons_layout.columnconfigure(i % buttons_per_row_vec, weight=1)

        mat_ops_label1 = ttk.Label(ops_grid, text="Matrices (Análisis y Ops Básicas):", font=self.fonts["label"])
        mat_ops_label1.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(10, 2))
        mat_buttons_layout_1 = ttk.Frame(ops_grid, style="FrameBG.TFrame")
        mat_buttons_layout_1.grid(row=3, column=0, columnspan=4, sticky=tk.EW, pady=(0, 10))
        mat_buttons_data_1 = [
            ("A + B", lambda: self.perform_matrix_op('add')), ("A - B", lambda: self.perform_matrix_op('subtract')),
            ("A * B", lambda: self.perform_matrix_op('multiply')),
            ("Escalar * A", lambda: self.perform_matrix_op('scalar_mult_a')),
            ("Escalar * B", lambda: self.perform_matrix_op('scalar_mult_b')),
            ("Transpuesta A", lambda: self.perform_matrix_op('transpose_a')),
            ("Transpuesta B", lambda: self.perform_matrix_op('transpose_b')),
            ("det(A)", lambda: self.perform_matrix_op('determinant_a')),
            ("det(B)", lambda: self.perform_matrix_op('determinant_b')),
            ("Inversa A", lambda: self.perform_matrix_op('inverse_a')),
            ("Inversa B", lambda: self.perform_matrix_op('inverse_b')),
            ("Condición A", lambda: self.perform_matrix_op('condition_a')),
            ("Condición B", lambda: self.perform_matrix_op('condition_b')),
            ("Val/Vec Propios A", lambda: self.perform_matrix_op('eigen_a')),
            ("Val/Vec Propios B", lambda: self.perform_matrix_op('eigen_b')),
            ("Dom. Diagonal A", lambda: self.perform_matrix_op('diag_dominance_a')),
            ("Dom. Diagonal B", lambda: self.perform_matrix_op('diag_dominance_b')),
        ]
        buttons_per_row_mat1 = 4
        for i, (text, command) in enumerate(mat_buttons_data_1):
            btn = ttk.Button(mat_buttons_layout_1, text=text, command=command, style="Accent.TButton", width=15)
            btn.grid(row=i // buttons_per_row_mat1, column=i % buttons_per_row_mat1, padx=4, pady=4, sticky=tk.EW)
            mat_buttons_layout_1.columnconfigure(i % buttons_per_row_mat1, weight=1)

        solve_ops_label = ttk.Label(ops_grid, text="Solución Sistemas Ax=b:", font=self.fonts["label"])
        solve_ops_label.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(10, 2))
        solve_buttons_layout = ttk.Frame(ops_grid, style="FrameBG.TFrame")
        solve_buttons_layout.grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=(0, 10))
        solve_buttons_data = [
            ("Gauss", lambda: self.perform_matrix_op('gauss')),
            ("Gauss-Jordan", lambda: self.perform_matrix_op('gauss_jordan')),
            ("LU Solve", lambda: self.perform_matrix_op('lu_doolittle_solve')),
            ("Cholesky Solve", lambda: self.perform_matrix_op('cholesky_solve')),
            ("Jacobi", lambda: self.perform_matrix_op('jacobi')),
            ("Gauss-Seidel", lambda: self.perform_matrix_op('gauss_seidel')),
            ("SOR", lambda: self.perform_matrix_op('sor')),
            ("Grad. Conjugado", lambda: self.perform_matrix_op('conjugate_gradient')),
            ("Desc. Pronunciado", lambda: self.perform_matrix_op('steepest_descent')),
        ]
        buttons_per_row_solve = 3
        for i, (text, command) in enumerate(solve_buttons_data):
            btn = ttk.Button(solve_buttons_layout, text=text, command=command, style="Accent.TButton", width=15)
            btn.grid(row=i // buttons_per_row_solve, column=i % buttons_per_row_solve, padx=4, pady=4, sticky=tk.EW)
            solve_buttons_layout.columnconfigure(i % buttons_per_row_solve, weight=1)

        other_ops_label = ttk.Label(ops_grid, text="Otros / Limpiar:", font=self.fonts["label"])
        other_ops_label.grid(row=6, column=0, columnspan=4, sticky=tk.W, pady=(10, 2))
        other_buttons_layout = ttk.Frame(ops_grid, style="FrameBG.TFrame")
        other_buttons_layout.grid(row=7, column=0, columnspan=4, sticky=tk.EW, pady=(0, 10))
        specific_problem_btn = ttk.Button(other_buttons_layout, text="Ejecutar Problema Específico (n=14..17)",
                                          command=self.run_specific_problem_gui, style="Accent.TButton")
        specific_problem_btn.pack(side=tk.LEFT, padx=4, pady=4)
        clear_all_btn = ttk.Button(other_buttons_layout, text="Limpiar Todo",
                                   command=self.clear_all_fields, style="Clear.TButton")
        clear_all_btn.pack(side=tk.LEFT, padx=4, pady=4)
        clear_mat_btn = ttk.Button(other_buttons_layout, text="Limpiar Matrices",
                                   command=self.clear_matrix_fields, style="Clear.TButton")
        clear_mat_btn.pack(side=tk.LEFT, padx=4, pady=4)
        clear_vec_btn = ttk.Button(other_buttons_layout, text="Limpiar Vectores",
                                   command=self.clear_vector_fields, style="Clear.TButton")
        clear_vec_btn.pack(side=tk.LEFT, padx=4, pady=4)
        ops_grid.columnconfigure(0, weight=1)

        # --- Área de Resultados ---
        result_groupbox = ttk.LabelFrame(main_content_frame, text="Resultado", padding="15")
        result_groupbox.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.result_text = scrolledtext.ScrolledText(result_groupbox, width=70, height=12, wrap=tk.WORD,
                                                     state=tk.DISABLED, font=self.fonts["result"],
                                                     bg=self.colors["result_bg"], fg=self.colors["result_fg"],
                                                     relief=tk.SOLID, borderwidth=1, padx=10, pady=10)
        # Configurar Tags para formato
        self.result_text.tag_configure("bold", font=tkFont.Font(family="Consolas", size=10, weight="bold"))
        self.result_text.tag_configure("error", foreground=self.colors["error_fg"],
                                       font=tkFont.Font(family="Consolas", size=10, weight="bold"))
        self.result_text.tag_configure("info", foreground=self.colors["text_label"],
                                       font=tkFont.Font(family="Consolas", size=10, slant="italic"))
        self.result_text.tag_configure("iter", foreground=self.colors["iter_fg"],
                                       font=self.fonts["iter"])  # Tag para iteraciones

        self.result_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

    # --- Funciones Auxiliares para Scroll ---
    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_frame_id, width=canvas_width)

    # --- Funciones de Parseo ---
    def _parse_vector(self, vec_str):
        # ... (sin cambios) ...
        if not vec_str.strip(): return None
        try:
            vec_str = vec_str.replace(' ', ',')
            parts = [p for p in vec_str.split(',') if p.strip()]
            return np.array([float(x.strip()) for x in parts])
        except ValueError:
            # No mostrar messagebox aquí, retornar None y que la función llamante decida
            print("Error de formato en vector. Use números separados por comas (o espacios).")
            return None

    # --- MODIFICADA: Generar Grid de Entradas para Matriz o Vector ---
    def _generate_grid(self, target):
        # ... (sin cambios lógicos) ...
        is_vector = target.startswith('V')
        is_matrix = not is_vector

        if is_vector:
            if target == 'VA':
                size_entry = self.vec_a_size_entry;
                container = self.vector_a_grid_container
                entry_list_attr = "vector_a_entries";
                rows = 1
            elif target == 'VB':
                size_entry = self.vec_b_size_entry;
                container = self.vector_b_grid_container
                entry_list_attr = "vector_b_entries";
                rows = 1
            elif target == 'VX0':  # NUEVO para x0
                size_entry = self.vec_x0_size_entry;
                container = self.vector_x0_grid_container
                entry_list_attr = "vector_x0_entries";
                rows = 1
            else:
                return

            try:
                cols = int(size_entry.get())
                if cols <= 0: raise ValueError("El tamaño debe ser positivo.")
                if cols > 50:
                    if not messagebox.askyesno("Advertencia",
                                               f"Generar un grid para vector de tamaño {cols} puede ocupar mucho espacio horizontal. ¿Continuar?"): return
            except ValueError as e:
                self.show_error(f"Tamaño inválido para {target}: {e}");
                return

        elif is_matrix:
            if target == 'A':
                rows_entry = self.mat_a_rows_entry;
                cols_entry = self.mat_a_cols_entry
                container = self.matrix_a_grid_container;
                entry_list_attr = "matrix_a_entries"
            elif target == 'B':
                rows_entry = self.mat_b_rows_entry;
                cols_entry = self.mat_b_cols_entry
                container = self.matrix_b_grid_container;
                entry_list_attr = "matrix_b_entries"
            else:
                return

            try:
                rows = int(rows_entry.get());
                cols = int(cols_entry.get())
                if rows <= 0 or cols <= 0: raise ValueError("Las dimensiones deben ser positivas.")
                if rows * cols > 150:
                    if not messagebox.askyesno("Advertencia",
                                               f"Generar un grid de {rows}x{cols} puede ser lento y ocupar mucho espacio. ¿Continuar?"): return
            except ValueError as e:
                self.show_error(f"Dimensiones inválidas para Matriz {target}: {e}");
                return
        else:
            return

        for widget in container.winfo_children(): widget.destroy()
        setattr(self, entry_list_attr, None)

        entries = []
        grid_frame = ttk.Frame(container, style="FrameBG.TFrame")
        grid_frame.pack(padx=1, pady=1, anchor='w')

        entry_width = 6 if rows * cols <= 50 else 5
        for r in range(rows):
            row_entries = []
            for c in range(cols):
                entry = ttk.Entry(grid_frame, width=entry_width, font=self.fonts["entry_grid"], justify='right')
                entry.grid(row=r, column=c, padx=1, pady=1)
                row_entries.append(entry)
            if is_vector:
                entries = row_entries
            else:
                entries.append(row_entries)

        setattr(self, entry_list_attr, entries)
        self.master.update_idletasks();
        self._on_frame_configure()

        # --- NUEVA: Leer Vector desde el Grid ---

    def _read_vector_from_grid(self, vector_target):
        # ... (Validación mejorada) ...
        if vector_target == 'VA':
            entries = self.vector_a_entries;
            size_entry = self.vec_a_size_entry;
            target_name = "Vector A"
        elif vector_target == 'VB':
            entries = self.vector_b_entries;
            size_entry = self.vec_b_size_entry;
            target_name = "Vector B/b"
        elif vector_target == 'VX0':  # NUEVO
            entries = self.vector_x0_entries;
            size_entry = self.vec_x0_size_entry;
            target_name = "Vector x0"
        else:
            return None

        if not entries: return None

        try:
            size = int(size_entry.get())
            if size != len(entries):
                # No mostrar error aquí, podría ser que el grid no se generó intencionalmente
                # print(f"Advertencia: Tamaño especificado para {target_name} ({size}) no coincide con grid ({len(entries)}).")
                return None  # Indicar que no se pudo leer

            vector_data = np.zeros(size, dtype=float)
            for i in range(size):
                val_str = entries[i].get().strip()
                if not val_str:
                    # Ser estricto: requerir un valor
                    raise ValueError(f"Entrada vacía en {target_name}[{i + 1}]. Ingrese un número.")
                    # vector_data[i] = 0.0 # O asumir cero
                else:
                    # Validar que sea un solo número
                    parts = val_str.split()
                    if len(parts) > 1:
                        raise ValueError(
                            f"Ingrese solo un número por celda (encontrado: '{val_str}' en {target_name}[{i + 1}]).")
                    vector_data[i] = float(parts[0])
            return vector_data
        except ValueError as e:
            self.show_error(f"Error al leer {target_name} del grid: {e}. Verifique las entradas.");
            return None
        except (TypeError, IndexError):
            self.show_error(f"Error interno al leer grid de {target_name}. ¿Generó el grid correctamente?");
            return None

    # --- MODIFICADA: Leer Matriz desde el Grid ---
    def _read_matrix_from_grid(self, matrix_target):
        # ... (Validación mejorada) ...
        if matrix_target == 'A':
            entries = self.matrix_a_entries;
            rows_entry = self.mat_a_rows_entry;
            cols_entry = self.mat_a_cols_entry;
            target_name = "Matriz A"
        elif matrix_target == 'B':
            entries = self.matrix_b_entries;
            rows_entry = self.mat_b_rows_entry;
            cols_entry = self.mat_b_cols_entry;
            target_name = "Matriz B"
        else:
            return None
        if not entries: return None
        try:
            rows = int(rows_entry.get());
            cols = int(cols_entry.get())
            if rows != len(entries) or (rows > 0 and cols != len(entries[0])):
                # print(f"Advertencia: Dimensiones especificadas para {target_name} ({rows}x{cols}) no coinciden con grid ({len(entries)}x{len(entries[0]) if rows>0 else 0}).")
                return None  # No se pudo leer

            matrix_data = np.zeros((rows, cols), dtype=float)
            for r in range(rows):
                for c in range(cols):
                    val_str = entries[r][c].get().strip()
                    if not val_str:
                        raise ValueError(f"Entrada vacía en {target_name}[{r + 1},{c + 1}]. Ingrese un número.")
                        # matrix_data[r, c] = 0.0 # O asumir cero
                    else:
                        parts = val_str.split()
                        if len(parts) > 1:
                            raise ValueError(
                                f"Ingrese solo un número por celda (encontrado: '{val_str}' en {target_name}[{r + 1},{c + 1}]).")
                        matrix_data[r, c] = float(parts[0])
            return matrix_data
        except ValueError as e:
            self.show_error(f"Error al leer {target_name} del grid: {e}. Verifique las entradas.");
            return None
        except (TypeError, IndexError):
            self.show_error(f"Error interno al leer grid de {target_name}. ¿Generó el grid correctamente?");
            return None

    def _get_scalar(self, entry_widget):
        # ... (sin cambios) ...
        scalar_str = entry_widget.get()
        if not scalar_str.strip(): return None
        try:
            return float(scalar_str)
        except ValueError:
            self.show_error("Escalar inválido. Debe ser un número.");
            return None

    def _get_iter_params(self):
        # Modificado para leer x0 desde su propio grid
        try:
            tol = float(self.tol_entry.get())
            if tol <= 0: raise ValueError("La tolerancia debe ser positiva.")
        except ValueError:
            self.show_error("Tolerancia inválida. Usando 1e-6.");
            tol = 1e-6
            self.tol_entry.delete(0, tk.END);
            self.tol_entry.insert(0, "1e-6")
        try:
            max_iter = int(self.max_iter_entry.get())
            if max_iter <= 0: raise ValueError("Máx. iteraciones debe ser positivo.")
        except ValueError:
            self.show_error("Máx. iteraciones inválido. Usando 1000.");
            max_iter = 1000
            self.max_iter_entry.delete(0, tk.END);
            self.max_iter_entry.insert(0, "1000")

        # Leer x0 desde su grid dedicado
        x0 = self._read_vector_from_grid('VX0')
        if x0 is None:
            # Si no se generó grid x0, usar ceros por defecto
            print("No se generó grid para x0 o está vacío. Se usará vector de ceros.")

        try:
            omega = float(self.omega_entry.get())
            if not (0 < omega < 2):
                messagebox.showwarning("Advertencia Omega", f"Omega = {omega} está fuera del rango (0, 2).")
        except ValueError:
            self.show_error("Omega inválido. Usando 1.1.");
            omega = 1.1
            self.omega_entry.delete(0, tk.END);
            self.omega_entry.insert(0, "1.1")
        return tol, max_iter, x0, omega

    # --- Funciones de Visualización y Limpieza ---
    def _insert_result_text(self, text, tags=None):
        """Inserta texto con tags opcionales y hace scroll, forzando update."""
        self.result_text.config(state=tk.NORMAL)
        if tags:
            # Asegurarse que el tag existe antes de usarlo
            if tags not in self.result_text.tag_names():
                # Configurar tag si no existe
                if tags == "iter":
                    self.result_text.tag_configure("iter", foreground=self.colors["iter_fg"], font=self.fonts["iter"])
                elif tags == "bold":
                    self.result_text.tag_configure("bold", font=tkFont.Font(family="Consolas", size=10, weight="bold"))
                elif tags == "error":
                    self.result_text.tag_configure("error", foreground=self.colors["error_fg"],
                                                   font=tkFont.Font(family="Consolas", size=10, weight="bold"))
                elif tags == "info":
                    self.result_text.tag_configure("info", foreground=self.colors["text_label"],
                                                   font=tkFont.Font(family="Consolas", size=10, slant="italic"))
            # Insertar con el tag
            self.result_text.insert(tk.END, text + "\n", tags)
        else:
            self.result_text.insert(tk.END, text + "\n")
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)
        self.master.update_idletasks()  # Usar update_idletasks es generalmente más seguro

    def show_result(self, result_data):
        """Muestra el resultado final. No borra si se mostraron iteraciones."""
        is_iter_dict = isinstance(result_data, dict) and 'solution' in result_data

        # Solo borrar si NO es un diccionario de resultado iterativo O si NO se mostraron iteraciones
        if not (is_iter_dict and self.show_iterations_var.get()):
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)

        self.result_text.config(state=tk.NORMAL)  # Habilitar para insertar

        # Formatear la salida
        if is_iter_dict:
            sol = result_data['solution']
            iters = result_data['iterations']
            converged = result_data['converged']
            ops = result_data.get('ops', {})
            method = result_data.get('method', 'Método Iterativo')
            time_taken = result_data.get('time', None)

            # Añadir encabezado solo si no se mostraron iteraciones
            if not self.show_iterations_var.get():
                self.result_text.insert(tk.END, f"--- Resumen Final ({method}) ---\n", "bold")

            if converged:
                self.result_text.insert(tk.END, f"Convergió en {iters} iteraciones.\n")
            else:
                self.result_text.insert(tk.END, f"NO convergió después de {iters} iteraciones.\n", "error")

            if time_taken is not None:
                self.result_text.insert(tk.END, f"Tiempo total de ejecución: {time_taken:.6f} segundos.\n")

            self.result_text.insert(tk.END, "Solución x final (o última aproximación):\n")
            if isinstance(sol, np.ndarray):
                res_str = np.array2string(sol, precision=6, suppress_small=True, separator=', ',
                                          threshold=np.inf, edgeitems=10,
                                          formatter={'float_kind': lambda x: "%.6f" % x})
                if sol.ndim == 1: res_str = res_str.replace('[[', '[').replace(']]', ']')
                self.result_text.insert(tk.END, res_str)
                if np.isnan(sol).any():
                    self.result_text.insert(tk.END, "\nAdvertencia: La solución contiene valores NaN.", "error")
            else:
                self.result_text.insert(tk.END, str(sol))

            if ops:
                self.result_text.insert(tk.END, "\n\nOperaciones Totales (Estimación):\n", "bold")
                keys_ordered = sorted([k for k, v in ops.items() if v > 0])
                for key in keys_ordered:
                    if key in ['matmul', 'dot', 'norm']:
                        self.result_text.insert(tk.END, f"- {key.capitalize()} (Llamadas NumPy): {ops[key]}\n")
                    else:
                        self.result_text.insert(tk.END, f"- {key.replace('_', ' ').capitalize()}: {ops[key]}\n")

        elif isinstance(result_data, tuple) and len(result_data) == 2 and isinstance(result_data[0], str):
            header, data = result_data
            self.result_text.insert(tk.END, f"{header}\n", "bold")
            if isinstance(data, np.ndarray):
                res_str = np.array2string(data, precision=4, suppress_small=True, separator=', ',
                                          threshold=np.inf, edgeitems=10,
                                          formatter={'float_kind': lambda x: "%.4f" % x})
                if data.ndim == 1: res_str = res_str.replace('[[', '[').replace(']]', ']')
                self.result_text.insert(tk.END, res_str)
            elif isinstance(data, str) and data != "":
                self.result_text.insert(tk.END, data)
            elif isinstance(data, (int, float)):
                # Formato para determinante y condición
                if "Condición" in header:
                    self.result_text.insert(tk.END, f"{data:.4e}")
                else:  # Asumir determinante u otro escalar
                    if abs(data) > 1e5 or (abs(data) < 1e-4 and data != 0):
                        self.result_text.insert(tk.END, f"{data:.4e}")
                    else:
                        self.result_text.insert(tk.END, f"{data:.6f}")
            elif data is not None and data != "":
                self.result_text.insert(tk.END, str(data))
        elif isinstance(result_data, np.ndarray):
            res_str = np.array2string(result_data, precision=4, suppress_small=True, separator=', ',
                                      threshold=np.inf, edgeitems=10,
                                      formatter={'float_kind': lambda x: "%.4f" % x})
            if result_data.ndim == 1: res_str = res_str.replace('[[', '[').replace(']]', ']')
            self.result_text.insert(tk.END, res_str)
        elif result_data is not None:
            if isinstance(result_data, (int, float)):
                if abs(result_data) > 1e5 or (abs(result_data) < 1e-4 and result_data != 0):
                    self.result_text.insert(tk.END, f"{result_data:.4e}")
                else:
                    self.result_text.insert(tk.END, f"{result_data:.6f}")
            else:  # Strings informativos como los de limpiar
                self.result_text.insert(tk.END, str(result_data), "info")

        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)

    def show_error(self, message):
        # ... (Uso de tag 'error') ...
        messagebox.showerror("Error", message)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Error: {message}", "error")
        self.result_text.config(state=tk.DISABLED)

    def clear_vector_fields(self):
        # Modificado para limpiar grids y dimensiones de vectores
        self.vec_a_size_entry.delete(0, tk.END)
        self.vec_b_size_entry.delete(0, tk.END)
        self.scalar_v_entry.delete(0, tk.END)
        self.vec_x0_size_entry.delete(0, tk.END)

        for widget in self.vector_a_grid_container.winfo_children(): widget.destroy()
        for widget in self.vector_b_grid_container.winfo_children(): widget.destroy()
        for widget in self.vector_x0_grid_container.winfo_children(): widget.destroy()

        self.vector_a_entries = None;
        self.vector_b_entries = None;
        self.vector_x0_entries = None
        self.vec_a = None;
        self.b = None

        self.show_result("Campos de vectores y grids limpiados.")
        self.master.update_idletasks();
        self._on_frame_configure()

    def clear_matrix_fields(self):
        # ... (sin cambios lógicos) ...
        self.mat_a_rows_entry.delete(0, tk.END)
        self.mat_a_cols_entry.delete(0, tk.END)
        self.mat_b_rows_entry.delete(0, tk.END)
        self.mat_b_cols_entry.delete(0, tk.END)
        self.scalar_m_entry.delete(0, tk.END)
        for widget in self.matrix_a_grid_container.winfo_children(): widget.destroy()
        for widget in self.matrix_b_grid_container.winfo_children(): widget.destroy()
        self.matrix_a_entries = None;
        self.matrix_b_entries = None
        self.A = None;
        self.L = None;
        self.U = None;
        self.L_chol = None
        self.show_result("Campos de matrices, grids y factorizaciones limpiados.")
        self.master.update_idletasks();
        self._on_frame_configure()

    def clear_all_fields(self):
        # ... (sin cambios lógicos) ...
        self.clear_vector_fields()
        self.clear_matrix_fields()
        self.tol_entry.delete(0, tk.END);
        self.tol_entry.insert(0, "1e-6")
        self.max_iter_entry.delete(0, tk.END);
        self.max_iter_entry.insert(0, "1000")
        self.omega_entry.delete(0, tk.END);
        self.omega_entry.insert(0, "1.1")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Todos los campos limpiados y parámetros reseteados.", "info")
        self.result_text.config(state=tk.DISABLED)

    # --- Lógica de Operaciones ---
    def perform_vector_op(self, operation):
        # Modificado para leer vectores desde el grid
        self.vec_a = self._read_vector_from_grid('VA')
        self.vec_b = self._read_vector_from_grid('VB')

        scalar = self._get_scalar(self.scalar_v_entry)
        result = None
        try:
            if operation == 'add':
                if self.vec_a is None or self.vec_b is None: raise ValueError(
                    "Ambos vectores A y B deben ser generados desde el grid.")
                if self.vec_a.shape != self.vec_b.shape: raise ValueError(
                    "Vectores deben tener la misma dimensión para sumar.")
                result = self.vec_a + self.vec_b
            elif operation == 'subtract':
                if self.vec_a is None or self.vec_b is None: raise ValueError(
                    "Ambos vectores A y B deben ser generados desde el grid.")
                if self.vec_a.shape != self.vec_b.shape: raise ValueError(
                    "Vectores deben tener la misma dimensión para restar.")
                result = self.vec_a - self.vec_b
            elif operation == 'dot':
                if self.vec_a is None or self.vec_b is None: raise ValueError(
                    "Ambos vectores A y B deben ser generados desde el grid.")
                if self.vec_a.shape != self.vec_b.shape: raise ValueError(
                    "Vectores deben tener la misma dimensión para producto punto.")
                result = np.dot(self.vec_a, self.vec_b)
            elif operation == 'cross':
                if self.vec_a is None or self.vec_b is None: raise ValueError(
                    "Ambos vectores A y B deben ser generados desde el grid.")
                if self.vec_a.size != 3 or self.vec_b.size != 3: raise ValueError(
                    "Producto cruz definido solo para vectores 3D.")
                result = np.cross(self.vec_a, self.vec_b)
            elif operation == 'scalar_mult_a':
                if self.vec_a is None: raise ValueError("Genere el grid para Vector A primero.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result = self.vec_a * scalar
            elif operation == 'scalar_mult_b':
                if self.vec_b is None: raise ValueError("Genere el grid para Vector B primero.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result = self.vec_b * scalar
            elif operation == 'magnitude_a':
                if self.vec_a is None: raise ValueError("Genere el grid para Vector A primero.")
                result = np.linalg.norm(self.vec_a)
            elif operation == 'magnitude_b':
                if self.vec_b is None: raise ValueError("Genere el grid para Vector B primero.")
                result = np.linalg.norm(self.vec_b)

            if result is not None:
                self.show_result(result)
        except ValueError as e:
            self.show_error(str(e))
        except Exception as e:
            self.show_error(f"Error inesperado en operación vectorial: {str(e)}")

    def perform_matrix_op(self, operation):
        # Leer matrices desde el grid
        self.A = self._read_matrix_from_grid('A')
        mat_b_gui = self._read_matrix_from_grid('B')

        # Leer vector b (para Ax=b) desde su grid
        self.b = self._read_vector_from_grid('VB')

        scalar = self._get_scalar(self.scalar_m_entry)
        result_data = None
        start_time = time.time()

        # CORRECCIÓN: Inicializar is_iterative aquí
        is_iterative = operation in ['jacobi', 'gauss_seidel', 'sor', 'conjugate_gradient', 'steepest_descent']
        is_solve_op = operation in ['gauss', 'gauss_jordan', 'lu_doolittle_solve', 'cholesky_solve'] or is_iterative

        # Limpiar resultados anteriores si se mostrarán iteraciones
        show_iters = self.show_iterations_var.get()
        if is_iterative and show_iters:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            self.master.update_idletasks()

            # Definir el callback para métodos iterativos
        iter_callback = None
        if is_iterative and show_iters:
            # Modificar callback para aceptar más argumentos
            def iter_callback_func(k, x, r=None, alpha=None, beta=None, p=None):  # Añadir p
                line = f"Iter {k}: x = {np.array2string(x, precision=4, suppress_small=True, separator=', ', threshold=8, edgeitems=2)}"  # Más corto
                if r is not None:
                    r_str = np.array2string(r, precision=3, suppress_small=True, separator=', ', threshold=8,
                                            edgeitems=2)
                    if r.ndim == 1: r_str = r_str.replace('[[', '[').replace(']]', ']')
                    line += f" | r = {r_str}"
                if p is not None:  # Mostrar vector p si está disponible (CG)
                    p_str = np.array2string(p, precision=3, suppress_small=True, separator=', ', threshold=8,
                                            edgeitems=2)
                    if p.ndim == 1: p_str = p_str.replace('[[', '[').replace(']]', ']')
                    line += f" | p = {p_str}"
                if alpha is not None:
                    line += f" | α = {alpha:.4e}"  # t_k
                if beta is not None:
                    line += f" | β = {beta:.4e}"  # s_k
                self._insert_result_text(line, "iter")

            iter_callback = iter_callback_func

        try:
            # --- Operaciones Básicas y Análisis (sin cambios) ---
            if operation == 'add':
                if self.A is None or mat_b_gui is None: raise ValueError(
                    "Ambas matrices A y B deben ser generadas desde el grid.")
                if self.A.shape != mat_b_gui.shape: raise ValueError(
                    "Matrices deben tener las mismas dimensiones para sumar.")
                result_data = self.A + mat_b_gui
            elif operation == 'subtract':
                if self.A is None or mat_b_gui is None: raise ValueError(
                    "Ambas matrices A y B deben ser generadas desde el grid.")
                if self.A.shape != mat_b_gui.shape: raise ValueError(
                    "Matrices deben tener las mismas dimensiones para restar.")
                result_data = self.A - mat_b_gui
            # ... (resto de operaciones básicas y análisis idénticas) ...
            elif operation == 'multiply':
                if self.A is None or mat_b_gui is None: raise ValueError(
                    "Ambas matrices A y B deben ser generadas desde el grid.")
                if self.A.shape[1] != mat_b_gui.shape[0]: raise ValueError(
                    "Columnas de A deben ser igual a filas de B para multiplicar.")
                result_data = np.matmul(self.A, mat_b_gui)
            elif operation == 'scalar_mult_a':
                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result_data = self.A * scalar
            elif operation == 'scalar_mult_b':
                if mat_b_gui is None: raise ValueError("Genere el grid para Matriz B primero.")
                if scalar is None: raise ValueError("Escalar es requerido.")
                result_data = mat_b_gui * scalar
            elif operation == 'transpose_a':
                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                result_data = self.A.T
            elif operation == 'transpose_b':
                if mat_b_gui is None: raise ValueError("Genere el grid para Matriz B primero.")
                result_data = mat_b_gui.T
            elif operation == 'determinant_a':
                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada para determinante.")
                result_data = np.linalg.det(self.A)
            elif operation == 'determinant_b':
                if mat_b_gui is None: raise ValueError("Genere el grid para Matriz B primero.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError(
                    "Matriz B debe ser cuadrada para determinante.")
                result_data = np.linalg.det(mat_b_gui)
            elif operation == 'inverse_a':
                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada para inversa.")
                try:
                    result_data = np.linalg.inv(self.A)
                except np.linalg.LinAlgError:
                    raise ValueError("Matriz A es singular, no se puede invertir.")
            elif operation == 'inverse_b':
                if mat_b_gui is None: raise ValueError("Genere el grid para Matriz B primero.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError(
                    "Matriz B debe ser cuadrada para inversa.")
                try:
                    result_data = np.linalg.inv(mat_b_gui)
                except np.linalg.LinAlgError:
                    raise ValueError("Matriz B es singular, no se puede invertir.")
            elif operation == 'eigen_a':
                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada.")
                eigenvalues, eigenvectors = np.linalg.eig(self.A)
                result_data = (
                    f"Valores Propios A:\n{np.array2string(eigenvalues, precision=4, suppress_small=True, formatter={'float_kind': lambda x: '%.4f' % x})}\n\nVectores Propios A (columnas):\n",
                    eigenvectors)
            elif operation == 'eigen_b':
                if mat_b_gui is None: raise ValueError("Genere el grid para Matriz B primero.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError("Matriz B debe ser cuadrada.")
                eigenvalues, eigenvectors = np.linalg.eig(mat_b_gui)
                result_data = (
                    f"Valores Propios B:\n{np.array2string(eigenvalues, precision=4, suppress_small=True, formatter={'float_kind': lambda x: '%.4f' % x})}\n\nVectores Propios B (columnas):\n",
                    eigenvectors)
            elif operation == 'condition_a':
                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                cond_num = np.linalg.cond(self.A)
                result_data = (f"Número Condición A (norma 2): {cond_num:.4e}", "")
            elif operation == 'condition_b':
                if mat_b_gui is None: raise ValueError("Genere el grid para Matriz B primero.")
                cond_num = np.linalg.cond(mat_b_gui)
                result_data = (f"Número Condición B (norma 2): {cond_num:.4e}", "")
            elif operation == 'diag_dominance_a':
                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada.")
                summary, details = self.check_diagonal_dominance(self.A)
                result_data = (summary, details)
            elif operation == 'diag_dominance_b':
                if mat_b_gui is None: raise ValueError("Genere el grid para Matriz B primero.")
                if mat_b_gui.shape[0] != mat_b_gui.shape[1]: raise ValueError("Matriz B debe ser cuadrada.")
                summary, details = self.check_diagonal_dominance(mat_b_gui)
                result_data = (summary, details)

            # --- Métodos de Solución Ax=b ---
            elif is_solve_op:

                if self.A is None: raise ValueError("Genere el grid para Matriz A primero.")
                if self.b is None: raise ValueError("Genere el grid para Vector b (o use campo Vector B).")
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("Matriz A debe ser cuadrada para Ax=b.")
                if self.A.shape[0] != self.b.shape[0]: raise ValueError("Dimensiones de A y b no compatibles.")
                if np.isnan(self.A).any() or np.isnan(self.b).any(): raise ValueError(
                    "Matriz A o vector b contienen NaN.")

                tol, max_iter, x0, omega = None, None, None, None
                if is_iterative:
                    tol, max_iter, x0, omega = self._get_iter_params()
                    # Usar x0 del grid dedicado si existe, si no, usar ceros
                    x0_grid = self._read_vector_from_grid('VX0')
                    if x0_grid is not None:
                        if x0_grid.shape[0] == self.A.shape[0]:
                            x0 = x0_grid
                            print("Usando x0 desde su grid dedicado.")
                        else:
                            self.show_error(
                                f"Tamaño del grid x0 ({x0_grid.shape[0]}) no coincide con n ({self.A.shape[0]}). Usando ceros.")
                            x0 = None
                    # Si no hay grid x0, x0 seguirá siendo None y los métodos usarán ceros

                solution, ops, iters, converged = None, {}, 0, False
                method_name_full = operation.replace("_", " ").title()
                if operation == 'lu_doolittle_solve': method_name_full = "LU Doolittle + Solve"
                if operation == 'cholesky_solve': method_name_full = "Cholesky + Solve"
                if operation == 'sor': method_name_full = f"SOR (ω={omega})"
                if operation == 'conjugate_gradient': method_name_full = "Gradiente Conjugado"
                if operation == 'steepest_descent': method_name_full = "Descenso Pronunciado"

                try:
                    A_copy, b_copy = self.A.copy(), self.b.copy()

                    # --- Llamadas a los métodos ---
                    if operation == 'gauss':
                        solution, ops = methods.gaussian_elimination(A_copy, b_copy)
                        converged, iters = True, 1
                    elif operation == 'gauss_jordan':
                        solution, ops = methods.gauss_jordan_elimination(A_copy, b_copy)
                        converged, iters = True, 1
                    elif operation == 'lu_doolittle_solve':
                        if self.L is None or self.U is None:
                            print("Calculando factorización Doolittle...")
                            self.L, self.U, ops_fact = methods.doolittle_factorization(A_copy)
                            print("Factorización completada.")
                        else:
                            print("Usando factorización LU Doolittle guardada.")
                            ops_fact = {}
                        solution, ops_solve = methods.lu_solve(self.L, self.U, b_copy)
                        ops = {k: ops_fact.get(k, 0) + ops_solve.get(k, 0) for k in set(ops_fact) | set(ops_solve)}
                        converged, iters = True, 1
                    elif operation == 'cholesky_solve':
                        if self.L_chol is None:
                            print("Calculando factorización Cholesky...")
                            if not methods.es_simetrica(A_copy): raise ValueError("Cholesky requiere matriz simétrica.")
                            self.L_chol, ops_fact = methods.cholesky_factorization(A_copy)
                            print("Factorización completada.")
                        else:
                            print("Usando factorización Cholesky guardada.")
                            ops_fact = {}
                        solution, ops_solve = methods.cholesky_solve(self.L_chol, b_copy)
                        ops = {k: ops_fact.get(k, 0) + ops_solve.get(k, 0) for k in set(ops_fact) | set(ops_solve)}
                        converged, iters = True, 1
                    # --- Llamadas a métodos iterativos INTERNOS (GUI) ---
                    elif operation == 'jacobi':
                        solution, iters, converged, ops = self.jacobi_method_gui(A_copy, b_copy, x0, tol, max_iter,
                                                                                 iter_callback)
                    elif operation == 'gauss_seidel':
                        solution, iters, converged, ops = self.gauss_seidel_method_gui(A_copy, b_copy, x0, tol,
                                                                                       max_iter, iter_callback)
                    elif operation == 'sor':
                        solution, iters, converged, ops = self.sor_method_gui(A_copy, b_copy, omega, x0, tol, max_iter,
                                                                              iter_callback)
                    elif operation == 'conjugate_gradient':
                        solution, iters, converged, ops = self.conjugate_gradient_gui(A_copy, b_copy, x0, tol, max_iter,
                                                                                      iter_callback)
                    elif operation == 'steepest_descent':
                        solution, iters, converged, ops = steepest_descent(A_copy, b_copy, x0, tol, max_iter,
                                                                           iter_callback)  # Usa la local

                    end_time = time.time()
                    result_data = {
                        'method': method_name_full, 'solution': solution,
                        'iterations': iters, 'converged': converged,
                        'ops': ops, 'time': end_time - start_time
                    }

                except (ValueError, np.linalg.LinAlgError) as e_solve:
                    self.show_error(f"Error en {method_name_full}: {e_solve}")
                    return

            else:
                raise ValueError(f"Operación desconocida: {operation}")

            # --- Mostrar resultado ---
            if result_data is not None and (not is_iterative or not show_iters):
                end_time = time.time()
                if not (isinstance(result_data, dict) and 'time' in result_data):
                    print(f"Tiempo de ejecución para '{operation}': {end_time - start_time:.6f} segundos")
                self.show_result(result_data)
            elif result_data is not None and is_iterative and show_iters:
                self._insert_result_text("\n" + "=" * 20 + " RESUMEN FINAL " + "=" * 20, "bold")
                self.show_result(result_data)  # Llama a show_result para formatear el dict final


        except ValueError as e:
            self.show_error(str(e))
        except np.linalg.LinAlgError as e:
            self.show_error(f"Error de Álgebra Lineal (NumPy): {str(e)}")
        except Exception as e:
            import traceback
            print("--- ERROR INESPERADO ---")
            traceback.print_exc()
            print("-----------------------")
            self.show_error(f"Error inesperado: {type(e).__name__} - {str(e)}")

    # --- Reimplementaciones de Métodos Iterativos con Callback ---
    def jacobi_method_gui(self, A, b, x0, tol, max_iter, callback=None):
        n = len(b)
        x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
        ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
        if callback: callback(k=0, x=x)  # Solo x en iter 0

        for k in range(max_iter):
            ops_iter = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
            x_new = np.zeros_like(x, dtype=float)
            for i in range(n):
                s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                if n > 1: ops_iter['mul'] += (n - 1); ops_iter['add_sub'] += (n - 2) if n > 2 else 0; ops_iter[
                    'add_sub'] += 1
                diag = A[i, i]
                if abs(diag) < 1e-15: raise ValueError(f"Elemento diagonal A[{i},{i}] cero.")
                ops_iter['abs'] += 1;
                ops_iter['compare'] += 1
                ops_iter['add_sub'] += 1;
                ops_iter['div'] += 1
                x_new[i] = (b[i] - s) / diag

            diff_norm = np.linalg.norm(x_new - x, ord=np.inf)
            ops_iter['add_sub'] += n;
            ops_iter['abs'] += n;
            ops_iter['compare'] += (n - 1) if n > 1 else 0
            ops_iter['norm'] += 1;
            ops_iter['compare'] += 1
            for key in ops_total: ops_total[key] += ops_iter[key]

            x = x_new
            if callback: callback(k=k + 1, x=x)  # Pasar k+1

            if diff_norm < tol:
                return x, k + 1, True, ops_total
        return x, max_iter, False, ops_total

    def gauss_seidel_method_gui(self, A, b, x0, tol, max_iter, callback=None):
        n = len(b)
        x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
        ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
        if callback: callback(k=0, x=x)

        for k in range(max_iter):
            ops_iter = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
            x_old = x.copy()
            for i in range(n):
                s1 = np.dot(A[i, :i], x[:i])
                if i > 0: ops_iter['mul'] += i; ops_iter['add_sub'] += (i - 1) if i > 1 else 0
                num_terms_s2 = n - 1 - i
                s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
                if num_terms_s2 > 0: ops_iter['mul'] += num_terms_s2; ops_iter['add_sub'] += (
                            num_terms_s2 - 1) if num_terms_s2 > 1 else 0
                diag = A[i, i]
                if abs(diag) < 1e-15: raise ValueError(f"Elemento diagonal A[{i},{i}] cero.")
                ops_iter['abs'] += 1;
                ops_iter['compare'] += 1
                ops_iter['add_sub'] += 2;
                ops_iter['div'] += 1
                x[i] = (b[i] - s1 - s2) / diag

            diff_norm = np.linalg.norm(x - x_old, ord=np.inf)
            ops_iter['add_sub'] += n;
            ops_iter['abs'] += n;
            ops_iter['compare'] += (n - 1) if n > 1 else 0
            ops_iter['norm'] += 1;
            ops_iter['compare'] += 1
            for key in ops_total: ops_total[key] += ops_iter[key]

            if callback: callback(k=k + 1, x=x)

            if diff_norm < tol:
                return x, k + 1, True, ops_total
        return x, max_iter, False, ops_total

    def sor_method_gui(self, A, b, omega, x0, tol, max_iter, callback=None):
        n = len(b)
        x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
        ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
        if callback: callback(k=0, x=x)

        for k in range(max_iter):
            ops_iter = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0}
            x_old = x.copy()
            for i in range(n):
                s1 = np.dot(A[i, :i], x[:i])
                if i > 0: ops_iter['mul'] += i; ops_iter['add_sub'] += (i - 1) if i > 1 else 0
                num_terms_s2 = n - 1 - i
                s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
                if num_terms_s2 > 0: ops_iter['mul'] += num_terms_s2; ops_iter['add_sub'] += (
                            num_terms_s2 - 1) if num_terms_s2 > 1 else 0
                diag = A[i, i]
                if abs(diag) < 1e-15: raise ValueError(f"Elemento diagonal A[{i},{i}] cero.")
                ops_iter['abs'] += 1;
                ops_iter['compare'] += 1
                ops_iter['add_sub'] += 2;
                ops_iter['div'] += 1
                x_gs = (b[i] - s1 - s2) / diag
                ops_iter['add_sub'] += 1;
                ops_iter['mul'] += 1;
                ops_iter['mul'] += 1;
                ops_iter['add_sub'] += 1
                x[i] = (1 - omega) * x_old[i] + omega * x_gs

            diff_norm = np.linalg.norm(x - x_old, ord=np.inf)
            ops_iter['add_sub'] += n;
            ops_iter['abs'] += n;
            ops_iter['compare'] += (n - 1) if n > 1 else 0
            ops_iter['norm'] += 1;
            ops_iter['compare'] += 1
            for key in ops_total: ops_total[key] += ops_iter[key]

            if callback: callback(k=k + 1, x=x)

            if diff_norm < tol:
                return x, k + 1, True, ops_total
        return x, max_iter, False, ops_total

    def conjugate_gradient_gui(self, A, b, x0, tol, max_iter, callback=None):
        n = len(b)
        if not methods.es_simetrica(A):
            print("Advertencia: La matriz A no es simétrica. CG podría no funcionar.")
        x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
        ops_total = {'add_sub': 0, 'mul': 0, 'div': 0, 'compare': 0, 'abs': 0, 'norm': 0, 'matmul': 0, 'dot': 0}

        Ax = A @ x
        ops_total['matmul'] += 1;
        ops_total['mul'] += n * n;
        ops_total['add_sub'] += n * (n - 1) if n > 1 else 0
        r = b - Ax
        ops_total['add_sub'] += n

        # Pasar alpha=None, beta=None, p=None en iteración 0
        if callback: callback(k=0, x=x, r=r, alpha=None, beta=None, p=None)

        norm_r = np.linalg.norm(r)
        ops_total['norm'] += 1;
        ops_total['mul'] += n;
        ops_total['add_sub'] += (n - 1) if n > 1 else 0
        ops_total['compare'] += 1
        if norm_r < tol:
            return x, 0, True, ops_total

        p = r.copy()
        rs_old = r @ r
        ops_total['dot'] += 1;
        ops_total['mul'] += n;
        ops_total['add_sub'] += (n - 1) if n > 1 else 0

        for k in range(max_iter):
            Ap = A @ p
            ops_total['matmul'] += 1;
            ops_total['mul'] += n * n;
            ops_total['add_sub'] += n * (n - 1) if n > 1 else 0

            pAp = p @ Ap
            ops_total['dot'] += 1;
            ops_total['mul'] += n;
            ops_total['add_sub'] += (n - 1) if n > 1 else 0

            if abs(pAp) < 1e-15:
                print(f"\nCG Error: Denominador pAp = {pAp:.2e} cercano a cero en iteración {k + 1}.")
                if pAp <= 0: print("  La matriz podría no ser definida positiva.")
                return x, k + 1, False, ops_total
            ops_total['abs'] += 1;
            ops_total['compare'] += 1
            alpha = rs_old / pAp  # Este es t_k
            ops_total['div'] += 1

            x_prev = x.copy()  # Guardar x_k
            x = x + alpha * p
            ops_total['mul'] += n;
            ops_total['add_sub'] += n

            r_new = r - alpha * Ap
            ops_total['mul'] += n;
            ops_total['add_sub'] += n

            norm_r_new = np.linalg.norm(r_new)
            ops_total['norm'] += 1;
            ops_total['mul'] += n;
            ops_total['add_sub'] += (n - 1) if n > 1 else 0
            ops_total['compare'] += 1

            r = r_new  # Actualizar r

            if norm_r_new < tol:
                # Llamar callback una última vez antes de retornar
                if callback: callback(k=k + 1, x=x, r=r, alpha=alpha, beta=None, p=p)
                return x, k + 1, True, ops_total

            rs_new = r @ r
            ops_total['dot'] += 1;
            ops_total['mul'] += n;
            ops_total['add_sub'] += (n - 1) if n > 1 else 0

            if abs(rs_old) < 1e-15:
                print(f"\nCG Advertencia: rs_old = {rs_old:.2e} cercano a cero en iteración {k + 1}.")
                # Llamar callback antes de salir
                if callback: callback(k=k + 1, x=x, r=r, alpha=alpha, beta=None, p=p)
                return x, k + 1, False, ops_total
            ops_total['abs'] += 1;
            ops_total['compare'] += 1
            beta = rs_new / rs_old  # Este es s_k
            ops_total['div'] += 1

            p_prev = p.copy()  # Guardar p_k
            p = r + beta * p
            ops_total['mul'] += n;
            ops_total['add_sub'] += n

            rs_old = rs_new

            # Llamar callback al final del bucle
            if callback: callback(k=k + 1, x=x, r=r, alpha=alpha, beta=beta, p=p)

        return x, max_iter, False, ops_total

    def run_specific_problem_gui(self):
        # ... (sin cambios) ...
        n_list = [14, 15, 16, 17]
        self.show_result(f"Ejecutando problema específico para n={n_list}...\n"
                         "Los resultados detallados aparecerán en la consola/terminal.")
        self.master.update_idletasks()

        proceed = messagebox.askyesno("Problema Específico",
                                      f"Se generarán y resolverán sistemas para n={n_list}.\n"
                                      "Esto puede tardar unos segundos o minutos, especialmente para n=17.\n"
                                      "La aplicación podría parecer no responder durante el cálculo.\n\n"
                                      "Los detalles se imprimirán en la consola.\n\n¿Desea continuar?")

        if not proceed:
            self.show_result("Ejecución del problema específico cancelada.")
            return

        try:
            print("\n--- INICIO EJECUCIÓN PROBLEMA ESPECÍFICO (Resultados en Consola) ---")
            start_total_time = time.time()
            results = methods.run_specific_problem(n_list)
            end_total_time = time.time()
            print(f"--- FIN EJECUCIÓN PROBLEMA ESPECÍFICO (Tiempo total: {end_total_time - start_total_time:.2f}s) ---")

            summary_lines = ["Resumen Problema Específico (Consola para detalles):\n"]
            for n_res, res in results.items():
                time_str = f"{res['solve_time']:.4f}s" if res['solve_time'] is not None else "N/A"
                gen_time_str = f"{res['gen_time']:.4f}s" if res['gen_time'] is not None else "N/A"
                summary_lines.append(f" n={n_res}: Estado='{res['status']}', T.Gen={gen_time_str}, T.Sol={time_str}")

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "\n".join(summary_lines), "info")  # Usar tag info
            self.result_text.config(state=tk.DISABLED)

            self.A, self.b, self.L, self.U, self.L_chol = None, None, None, None, None
            messagebox.showinfo("Problema Específico Completado",
                                f"Ejecución para n={n_list} finalizada.\nRevise la consola para los detalles y soluciones.\n"
                                "(Variables A, b y factorizaciones reseteadas en la GUI).")

        except Exception as e:
            self.show_error(f"Error al ejecutar problema específico: {e}")
            import traceback
            print("--- ERROR EN PROBLEMA ESPECÍFICO ---")
            traceback.print_exc()
            print("---------------------------------")

    # --- Función check_diagonal_dominance ---
    def check_diagonal_dominance(self, matrix):
        # ... (sin cambios) ...
        n = matrix.shape[0]
        strictly_dominant_row_count = 0
        weakly_dominant_row_count = 0
        non_dominant_row_count = 0

        details_list = []

        for i in range(n):
            diag_element_abs = abs(matrix[i, i])
            if n == 1:
                sum_off_diag_abs = 0.0
            else:
                sum_off_diag_abs = np.sum(np.abs(matrix[i, :])) - diag_element_abs

            row_desc = f"Fila {i + 1}: |{matrix[i, i]:.3g}| vs Σ|restantes| = {sum_off_diag_abs:.3g}. "
            if diag_element_abs > sum_off_diag_abs + 1e-12:
                details_list.append(row_desc + "Estrictamente dominante.")
                strictly_dominant_row_count += 1
            elif np.isclose(diag_element_abs, sum_off_diag_abs):
                details_list.append(row_desc + "Débilmente dominante (igualdad).")
                weakly_dominant_row_count += 1
            else:
                details_list.append(row_desc + "NO es dominante.")
                non_dominant_row_count += 1

        summary = ""
        if non_dominant_row_count > 0:
            summary = "La matriz NO es diagonalmente dominante (al menos una fila no cumple)."
        elif strictly_dominant_row_count == n:
            summary = "La matriz es ESTRICTAMENTE diagonalmente dominante (todas las filas cumplen estrictamente)."
        elif (strictly_dominant_row_count + weakly_dominant_row_count) == n and strictly_dominant_row_count > 0:
            summary = "La matriz es DÉBILMENTE diagonalmente dominante (cumple en todas las filas, con al menos una fila estricta)."
        elif (strictly_dominant_row_count + weakly_dominant_row_count) == n and strictly_dominant_row_count == 0:
            summary = "La matriz es DÉBILMENTE diagonalmente dominante (cumple en todas las filas, pero ninguna es estricta, solo igualdad)."
        else:
            summary = "La matriz NO es diagonalmente dominante (combinación no válida encontrada)."

        full_details_str = "\n".join(details_list)
        return summary, full_details_str


if __name__ == '__main__':
    root = tk.Tk()
    app = AlgebraLinealTkinterGUI(root)
    root.mainloop()
