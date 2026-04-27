import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


L = 10.0
Nx = 200
h = L / Nx
x = np.linspace(0, L, Nx + 1)

c = 1.0
T_max = 5.0
tau = 0.9 * h / c
Nt = int(T_max / tau) + 1

a_val = 2.0
b_val = 1.0
l1, l2, l12 = 2.0, 4.0, 3.0
LAMBDA = 10.0


def u0_A(x):
    return np.where(x <= 0.0, a_val, b_val)

def u0_B(x):
    cond = (x >= l1) & (x <= l2)
    val = np.zeros_like(x)
    val[cond] = 0.5 * (1 - np.cos(2 * np.pi * (x[cond] - l1) / (l2 - l1)))
    return val

def u0_C(x):
    cond1 = (x >= l1) & (x <= l12)
    cond2 = (x >= l12) & (x <= l2)
    val = np.zeros_like(x)
    if l12 != l1:
        val[cond1] = 1 - (2/3) * (x[cond1] - l1) / (l12 - l1)
    if l2 != l12:
        val[cond2] = (2/3) * (x[cond2] - l12) / (l2 - l12)
    return val


def exact_solution(x, t, u0_func):
    return u0_func(x - c * t)


def implicit_left_corner(u, tau, h, c, t=None, u0_func=None):
    
    N = len(u) - 1
    sigma = c * tau / h
    
    u_new = np.zeros_like(u)
    
    # Левое граничное условие (характеристика)
    if u0_func is not None and t is not None:
        u_new[0] = u0_func(np.array([-c * (t + tau)]))[0]
    else:
        u_new[0] = u[0]
    
    # Явная формула
    for i in range(1, N + 1):
        u_new[i] = (sigma * u_new[i-1] + u[i]) / (1 + sigma)
    
    return u_new


def box_scheme_thomas(u, tau, h, c, t=None, u0_func=None):
    
    N = len(u) - 1
    gamma = c * tau / (2 * h)
    
    u_new = np.zeros_like(u)
    
    # Левое граничное условие (характеристика)
    if u0_func is not None and t is not None:
        u_new[0] = u0_func(np.array([-c * (t + tau)]))[0]
    else:
        u_new[0] = u[0]
    
    # Явный расчёт остальных точек
    for i in range(1, N + 1):
        u_new[i] = (gamma * u_new[i-1] + (1 - gamma) * u[i] + gamma * u[i-1]) / (1 + gamma)
    
    return u_new


def fedorenko_scheme_vectorized(u, tau, h, c, t, u0_func, lam=10.0):
    N = len(u) - 1
    gamma = c * tau / h
    u_new = np.zeros_like(u)
    
    # Левое граничное условие (характеристическое)
    if u0_func is not None:
        u_new[0] = u0_func(np.array([-c * (t + tau)]))[0]
    else:
        u_new[0] = u[0]
    
    d1 = u[1:-1] - u[:-2]
    d2 = u[2:] - 2*u[1:-1] + u[:-2]
    
    # Избегаем деления на ноль
    with np.errstate(divide='ignore', invalid='ignore'):
        smooth = np.abs(d2) <= lam * np.abs(d1)
        smooth = np.nan_to_num(smooth, False)
    
    sigma = np.where(smooth, 0.0, 1.0)
    
    u_new[1:-1] = u[1:-1] - gamma * d1 - 0.5 * sigma * (gamma - gamma**2) * d2
    
    # Правое граничное условие (экстраполяция)
    u_new[N] = 2*u_new[N-1] - u_new[N-2]
    return u_new

def compute_error(u_cur, h, u_exact, name):
    diff = u_cur - u_exact

    if name == "L2":
        return np.sqrt(h * np.sum(diff**2))
    
    elif name == "Linf":
        return np.max(np.abs(diff))
    
    elif name == "L1":
        return h * np.sum(np.abs(diff))

# ============================================================
# Анимация
# ============================================================
def animate_scheme_with_error(scheme_func, u0_func, title, scheme_name, delay_ms=100):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.set_xlim(0, L)
    ax1.set_ylim(-0.5, max(a_val, b_val, 2.2))
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.grid(True, alpha=0.3)
    
    line_num, = ax1.plot([], [], 'b-', lw=2, label='Численное')
    line_exact, = ax1.plot([], [], 'r--', lw=2, label='Точное')
    ax1.legend(loc='upper right')
    
    ax2.set_xlim(0, T_max)
    ax2.set_yscale('log')
    ax2.set_xlabel("Время t")
    ax2.set_ylabel("Ошибка")
    ax2.grid(True, alpha=0.3)
    
    line_L2, = ax2.plot([], [], 'g-', lw=1.5, label='L₂')
    line_Linf, = ax2.plot([], [], 'm-', lw=1.5, label='L∞')
    line_L1, = ax2.plot([], [], 'c-', lw=1.5, label='L₁')
    ax2.legend(loc='upper left')
    
    t_history, L2_history, Linf_history, L1_history = [], [], [], []
    u_cur = u0_func(x)
    t_cur = 0.0
    
    def init():
        u_exact_0 = exact_solution(x, 0, u0_func)
        line_num.set_data(x, u_cur)
        line_exact.set_data(x, u_exact_0)
        
        t_history.clear()
        L2_history.clear()
        Linf_history.clear()
        L1_history.clear()
        
        t_history.append(0.0)
        L2_history.append(0.0)
        Linf_history.append(0.0)
        L1_history.append(0.0)
        
        line_L2.set_data(t_history, L2_history)
        line_Linf.set_data(t_history, Linf_history)
        line_L1.set_data(t_history, L1_history)
        
        ax1.set_title(f"{title} – {scheme_name}, t = 0.00")
        return line_num, line_exact, line_L2, line_Linf, line_L1
    
    def update(frame):
        nonlocal u_cur, t_cur
        
        # Передаём t_cur и u0_func для схем, которым это нужно
        u_cur = scheme_func(u_cur, tau, h, c, t_cur, u0_func)
        t_cur += tau
        
        u_exact = exact_solution(x, t_cur, u0_func)
        
        err_L2 = compute_error(u_cur, h, u_exact, 'L2')
        err_Linf = compute_error(u_cur, h, u_exact, 'Linf')
        err_L1 = compute_error(u_cur, h, u_exact, 'L1')
        
        line_num.set_data(x, u_cur)
        line_exact.set_data(x, u_exact)
        
        t_history.append(t_cur)
        L2_history.append(err_L2)
        Linf_history.append(err_Linf)
        L1_history.append(err_L1)
        
        line_L2.set_data(t_history, L2_history)
        line_Linf.set_data(t_history, Linf_history)
        line_L1.set_data(t_history, L1_history)
        
        ax2.relim()
        ax2.autoscale_view()
        
        ax1.set_title(f"{title} – {scheme_name}, t = {t_cur:.2f} | "
                     f"L₂={err_L2:.3e}, L∞={err_Linf:.3e}, L₁={err_L1:.3e}")
        
        return line_num, line_exact, line_L2, line_Linf, line_L1
    
    ani = FuncAnimation(fig, update, frames=Nt, init_func=init, 
                       interval=delay_ms, repeat=False, blit=True)
    plt.tight_layout()
    return ani



def run_convergence_experiment():
   
    # Гладкое начальное условие
    def u0_smooth(x):
        # return np.sin(2 * np.pi * x / L)
        return np.sin(np.pi * x / L)

    T_end = 2.0
    cfl = 0.5
    c_const = c

    schemes = [
        ("Неявный левый уголок", implicit_left_corner),
        ("Схема квадрат",        box_scheme_thomas),
        ("Схема Федоренко",      fedorenko_scheme_vectorized),
    ]

    base_Nx = 200
    levels = 4
    Nx_list = [base_Nx * (2**k) for k in range(levels)]

    print("\n" + "=" * 80)
    print("ЧИСЛЕННЫЙ ЭКСПЕРИМЕНТ: ПОРЯДОК СХОДИМОСТИ НА ГЛАДКОМ РЕШЕНИИ")
    print(f"u₀(x) = sin(2πx/L),  L = {L},  T_max = {T_end},  CFL = {cfl}")
    print(f"Сетки: Nx = {Nx_list}")
    print("Норма ошибки: L₂")
    print("=" * 80)

    for scheme_name, scheme_func in schemes:
        print(f"\n{'─' * 80}")
        print(f"Схема: {scheme_name}")
        print(f"{'Nx':<8} {'h':<14} {'τ':<14} {'L₂-ошибка':<16}")
        print("-" * 55)

        errors_L2 = []

        for Nx_cur in Nx_list:
            h_cur = L / Nx_cur
            x_cur = np.linspace(0, L, Nx_cur + 1)
            tau_cur = cfl * h_cur / c_const
            Nt_cur = int(T_end / tau_cur) + 1

            u = u0_smooth(x_cur).copy()
            t = 0.0

            for _ in range(Nt_cur):
                u = scheme_func(u, tau_cur, h_cur, c_const, t, u0_smooth)
                t += tau_cur

            u_exact = exact_solution(x_cur, t, u0_smooth)
            err_L2 = compute_error(u, h_cur, u_exact, 'L2')  
            errors_L2.append(err_L2)

            print(f"{Nx_cur:<8} {h_cur:<14.6f} {tau_cur:<14.6f} {err_L2:<16.8e}")

        print("\nПорядки сходимости:")
        if len(errors_L2) >= 2:
            print("-" * 28)
            # for k in range(1, len(errors_L2)):
            #     p = np.log2(errors_L2[k-1] / errors_L2[k])
            #     transition = f"{Nx_list[k-1]} → {Nx_list[k]}"
            #     print(f"{transition:<16} {p:<10.3f}")
            for k in range(1, len(errors_L2)):
                p = np.log2(errors_L2[k - 1] / errors_L2[k])
                print(f"Порядки сходимости: {p:.2f}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'experiment':
        run_convergence_experiment()
        sys.exit(0)

    print("=" * 60)
    print(f"Параметры: L={L}, Nx={Nx}, c={c}, T_max={T_max}")
    print(f"h={h:.4f}, τ={tau:.4f}, CFL={c*tau/h:.3f}")
    print("=" * 60)
    
    delay = int(input("Задержка (мс) [50]: ") or 50)
    
    ani_A = animate_scheme_with_error(implicit_left_corner, u0_A, 
                                       "Условие A", "Неявный левый уголок", delay)
    ani_B = animate_scheme_with_error(box_scheme_thomas, u0_B, 
                                       "Условие B", "Схема квадрат", delay)
    ani_C = animate_scheme_with_error(fedorenko_scheme_vectorized, u0_C, 
                                       "Условие C", "Схема Федоренко", delay)
    
    plt.show()