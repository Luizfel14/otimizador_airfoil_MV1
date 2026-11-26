import subprocess
import os
import tempfile
import numpy as np
from math import comb
from scipy.optimize import minimize, least_squares, differential_evolution
from scipy.interpolate import interp1d

def bezier_curve(control_points, n_points=120):
    n = len(control_points) - 1
    beta = np.linspace(0, np.pi, n_points)
    t = 0.5 * (1 - np.cos(beta))
    curve = np.zeros((n_points, 2))
    for i in range(n + 1):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, control_points[i])
    return curve

def load_selig_from_lines(lines):
    coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    coords = np.array(coords)
    le_idx = np.argmin(coords[:, 0])
    upper = coords[:le_idx + 1][::-1]
    lower = coords[le_idx:]
    upper[:, 0] = np.clip(upper[:, 0], 0.0, 1.0)
    lower[:, 0] = np.clip(lower[:, 0], 0.0, 1.0)
    return upper, lower

def fit_initial_bezier(upper_orig, lower_orig, degree=8):
    beta_cp = np.linspace(0, np.pi, degree)
    x_fixed_interior = 0.5 * (1 - np.cos(beta_cp[1:-1]))
    x_cp = np.concatenate(([0.0, 0.0], x_fixed_interior, [1.0]))

    def fit_side(target_pts, is_upper):
        y_te = target_pts[-1, 1]
        interp = interp1d(target_pts[:, 0], target_pts[:, 1], kind='cubic', fill_value="extrapolate")
        y1_init = 0.08 if is_upper else -0.04
        y_init = np.concatenate(([y1_init], interp(x_fixed_interior)))

        def res_fn(y_p):
            cp = np.column_stack((x_cp, np.concatenate(([0.0], y_p, [y_te]))))
            c = bezier_curve(cp, 130)
            w = np.where(c[:, 0] < 0.05, 2.5, 1.0)
            return (c[:, 1] - interp(c[:, 0])) * w

        res = least_squares(res_fn, y_init, max_nfev=2000)
        return x_cp, res.x, y_te

    x_cp, yu_init, yu_te = fit_side(upper_orig, True)
    _, yl_init, yl_te = fit_side(lower_orig, False)
    return x_cp, yu_init, yu_te, yl_init, yl_te

def check_mdo_geometry(upper_curve, lower_curve, mdo_specs):
    x_grid = np.linspace(0.01, 0.99, 200)
    yu = np.interp(x_grid, upper_curve[:, 0], upper_curve[:, 1])
    yl = np.interp(x_grid, lower_curve[:, 0], lower_curve[:, 1])
    thickness = yu - yl

    if np.min(thickness) <= 0.0005:
        return False, 1e4

    max_tc = np.max(thickness)
    spar_tc = np.interp(mdo_specs["spar_x"], x_grid, thickness)
    te_tc = np.interp(0.95, x_grid, thickness)

    penalty = 0.0
    if spar_tc < mdo_specs["min_spar_tc"]:
        penalty += 500.0 * (mdo_specs["min_spar_tc"] - spar_tc) ** 2
    if max_tc < mdo_specs["min_max_tc"]:
        penalty += 500.0 * (mdo_specs["min_max_tc"] - max_tc) ** 2
    if max_tc > mdo_specs["max_max_tc"]:
        penalty += 500.0 * (max_tc - mdo_specs["max_max_tc"]) ** 2
    if te_tc < mdo_specs["min_te_tc"]:
        penalty += 500.0 * (mdo_specs["min_te_tc"] - te_tc) ** 2

    return (penalty == 0.0), penalty

def run_xfoil_polar(dat_path, reynolds, alpha_seq=(0.0, 18.0, 1.5), ncrit=7):
    polar_file = dat_path + ".pol"
    if os.path.exists(polar_file):
        os.remove(polar_file)

    commands = f"""PLOP
G F

LOAD {dat_path}
PPAR
N 220


PANE
OPER
VISC {reynolds}
M 0.04
VPAR
N {ncrit}

ITER 150
PACC
{polar_file}

ASEQ {alpha_seq[0]} {alpha_seq[1]} {alpha_seq[2]}
PACC

QUIT
"""
    try:
        subprocess.run(
            ["xfoil"],
            input=commands.encode("ascii"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8
        )
    except Exception:
        return None

    if not os.path.exists(polar_file):
        return None

    data = []
    with open(polar_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    a, cl, cd, _, cm = map(float, parts[:5])
                    if cd > 0 and not np.isnan(cl):
                        data.append([a, cl, cd, cm])
                except ValueError:
                    continue
    os.remove(polar_file)

    if len(data) < 6:
        return None
    return np.array(data)

def run_optimization(upper_orig, lower_orig, re_list, mdo_specs, weights, bezier_degree=8, max_fev=450, progress_callback=None):
    x_cp, yu_base, yu_te, yl_base, yl_te = fit_initial_bezier(upper_orig, lower_orig, degree=bezier_degree)
    n_vars_side = len(yu_base)
    x0 = np.concatenate((yu_base, yl_base))

    bounds = [(val - 0.06, val + 0.06) for val in x0]

    best_score = -1e9
    best_x = x0.copy()
    history = []
    eval_count = [0]

    def objective(vars_vec):
        eval_count[0] += 1
        yu = vars_vec[:n_vars_side]
        yl = vars_vec[n_vars_side:]

        cp_u = np.column_stack((x_cp, np.concatenate(([0.0], yu, [yu_te]))))
        cp_l = np.column_stack((x_cp, np.concatenate(([0.0], yl, [yl_te]))))

        upper_curve = bezier_curve(cp_u, n_points=105)
        lower_curve = bezier_curve(cp_l, n_points=105)

        is_valid, geo_penalty = check_mdo_geometry(upper_curve, lower_curve, mdo_specs)
        if not is_valid:
            return 100.0 + geo_penalty

        airfoil_coords = np.vstack((upper_curve[::-1], lower_curve[1:]))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as tmp:
            tmp.write("OPT-CANDIDATE\n")
            for x, y in airfoil_coords:
                tmp.write(f"  {x:.6f}  {y:.6f}\n")
            tmp_path = tmp.name

        total_fitness = 0.0
        for re in re_list:
            polar = run_xfoil_polar(tmp_path, re, alpha_seq=(0.0, 18.0, 1.5), ncrit=7)
            if polar is None:
                os.remove(tmp_path)
                return 100.0

            cl_vals = polar[:, 1]
            cd_vals = polar[:, 2]
            cm_vals = polar[:, 3]

            cl_max = np.max(cl_vals)

            mask_voo = (cl_vals >= 1.1) & (cl_vals <= 1.7)
            if np.any(mask_voo):
                ld_operacional = np.mean(cl_vals[mask_voo] / cd_vals[mask_voo])
            else:
                ld_operacional = np.max(cl_vals / cd_vals) * 0.7

            cm_min = np.min(cm_vals)
            cm_penalty = 0.0
            if cm_min < mdo_specs["min_cm"]:
                cm_penalty = 25.0 * (mdo_specs["min_cm"] - cm_min) ** 2

            re_score = (weights["cl_max"] * cl_max) + (weights["ld_max"] * (ld_operacional / 45.0)) - cm_penalty
            total_fitness += re_score / len(re_list)

        os.remove(tmp_path)

        nonlocal best_score, best_x
        if total_fitness > best_score:
            best_score = total_fitness
            best_x = vars_vec.copy()
            history.append((eval_count[0], best_score))
            if progress_callback:
                progress_callback(eval_count[0], max_fev, best_score)

        return -total_fitness

    base_score = -objective(x0)

    # Estágio 1: Exploração global (Differential Evolution)
    pop_size = 10
    n_dim = len(x0)
    init_pop = np.clip(
        x0 + np.random.uniform(-0.025, 0.025, size=(pop_size, n_dim)),
        [b[0] for b in bounds],
        [b[1] for b in bounds]
    )
    init_pop[0] = x0

    max_gen = max(5, int((max_fev * 0.65) // pop_size))
    differential_evolution(
        objective,
        bounds=bounds,
        init=init_pop,
        maxiter=max_gen,
        tol=1e-3,
        polish=False
    )

    # Estágio 2: Refinamento local (Powell)
    rem_fev = max(80, max_fev - eval_count[0])
    minimize(
        objective,
        best_x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 45, "maxfev": rem_fev, "ftol": 1e-4}
    )

    u_base = bezier_curve(np.column_stack((x_cp, np.concatenate(([0.0], yu_base, [yu_te])))), 120)
    l_base = bezier_curve(np.column_stack((x_cp, np.concatenate(([0.0], yl_base, [yl_te])))), 120)
    base_coords = np.vstack((u_base[::-1], l_base[1:]))

    yu_opt = best_x[:n_vars_side]
    yl_opt = best_x[n_vars_side:]
    u_opt = bezier_curve(np.column_stack((x_cp, np.concatenate(([0.0], yu_opt, [yu_te])))), 120)
    l_opt = bezier_curve(np.column_stack((x_cp, np.concatenate(([0.0], yl_opt, [yl_te])))), 120)
    opt_coords = np.vstack((u_opt[::-1], l_opt[1:]))

    cp_u_opt = np.column_stack((x_cp, np.concatenate(([0.0], yu_opt, [yu_te]))))
    cp_l_opt = np.column_stack((x_cp, np.concatenate(([0.0], yl_opt, [yl_te]))))

    return base_coords, opt_coords, base_score, best_score, history, (cp_u_opt, cp_l_opt)