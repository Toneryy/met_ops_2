import numpy as np
import matplotlib.pyplot as plt
import time
from math import cos, sin, pi, sqrt, exp
from matplotlib.backends.backend_pdf import PdfPages

func_str = "x**2 - 10*cos(2*pi*x) + 10"
a, b = -5.12, 5.12
eps = 0.01

def f(x):
    return eval(func_str, {"x": x, "np": np, "cos": cos, "sin": sin, "pi": pi, "sqrt": sqrt, "exp": exp})

def strongin(f, a, b, eps, r=2.0, max_iter=10000):
    start = time.time()
    pts, vals = [a, b], [f(a), f(b)]
    it = 0

    while it < max_iter:
        it += 1
        idx = np.argsort(pts)
        pts, vals = [pts[i] for i in idx], [vals[i] for i in idx]
        n = len(pts)

        M = max(abs((vals[i] - vals[i - 1]) / (pts[i] - pts[i - 1])) for i in range(1, n))
        m = r * (M if M > 1e-9 else 1)

        R = [m * (pts[i] - pts[i - 1]) + (vals[i] - vals[i - 1]) ** 2 / (m * (pts[i] - pts[i - 1]))
             - 2 * (vals[i] + vals[i - 1]) for i in range(1, n)]
        i_max = np.argmax(R) + 1

        dx, dz = pts[i_max] - pts[i_max - 1], vals[i_max] - vals[i_max - 1]
        x_new = (pts[i_max] + pts[i_max - 1]) / 2 - dz / (2 * m)
        x_new = np.clip(x_new, pts[i_max - 1], pts[i_max])

        pts.append(x_new)
        vals.append(f(x_new))
        if dx < eps:
            break

    i_min = np.argmin(vals)
    return pts[i_min], vals[i_min], it, time.time() - start, pts, vals

def lower_piecewise(points, values, r=2.0):
    idx = np.argsort(points)
    points, values = [points[i] for i in idx], [values[i] for i in idx]
    M = max(abs((values[i] - values[i - 1]) / (points[i] - points[i - 1])) for i in range(1, len(points)))
    m = r * (M if M > 1e-9 else 1)

    x_line, y_line = [], []
    for i in range(1, len(points)):
        x1, x2, z1, z2 = points[i - 1], points[i], values[i - 1], values[i]
        x_mid = np.clip((x1 + x2) / 2 - (z2 - z1) / (2 * m), x1, x2)
        y_mid = min(z1 - m * abs(x_mid - x1), z2 - m * abs(x_mid - x2))
        x_line += [x1, x_mid, x2]
        y_line += [z1, y_mid, z2]
    return x_line, y_line

print("Поиск глобального минимума методом Стронгина")
print(f"f(x) = {func_str}\n[{a}, {b}], eps = {eps}\n")

x_min, f_min, iters, t, pts, vals = strongin(f, a, b, eps)
print(f"x_min = {x_min:.6f}\nf_min = {f_min:.6f}\nитераций = {iters}\nвремя = {t:.4f} с")

X = np.linspace(a, b, 1000)
Y = [f(x) for x in X]
x_aux, y_aux = lower_piecewise(pts, vals)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X, Y, 'b-', lw=2, label=f'f(x) = {func_str}')
ax.scatter(pts, vals, color='red', s=30, label=f'Точки ({len(pts)})', alpha=0.7)
ax.plot(x_aux, y_aux, 'g--', lw=1.5, label='Оценка снизу')
ax.scatter([x_min], [f_min], color='lime', s=200, marker='*',
           edgecolors='darkgreen', lw=2, label=f'Минимум: x={x_min:.4f}, f={f_min:.4f}')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('Поиск глобального минимума методом Стронгина', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()

pdf = "result_strongin_method.pdf"
with PdfPages(pdf) as out:
    out.savefig(fig, bbox_inches='tight')
    rep = plt.figure(figsize=(8.5, 11))
    text = f"""
Метод Стронгина (глобальный поиск с оценкой по Липшицу)

Функция: f(x) = {func_str}
Отрезок: [{a}, {b}]
Точность eps = {eps}
x_min = {x_min:.8f}
f_min = {f_min:.8f}
Итераций = {iters}
Время = {t:.4f} с

Алгоритм:
1. Инициализация граничных точек
2. Оценка константы Липшица M
3. Построение характеристик интервалов R(i)
4. Добавление новой точки в интервал с max(R)
5. Повтор до dx < eps

Преимущества:
- Гарантирует нахождение глобального минимума
- Адаптивен к форме функции
- Детерминированный, воспроизводимый результат
"""
    rep.text(0.1, 0.9, text, family='monospace', va='top', fontsize=10)
    plt.axis('off')
    out.savefig(rep, bbox_inches='tight')
    plt.close(rep)

print(f"\nРезультаты сохранены в {pdf}")
plt.show()
