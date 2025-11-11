import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt, exp
from matplotlib.backends.backend_pdf import PdfPages
from strongin import strongin_method, lower_piecewise


def rastrigin(x: float) -> float:
    return x**2 - 10*cos(2*pi*x) + 10


def main():
    func = rastrigin
    func_str = "x² - 10*cos(2πx) + 10"
    a, b = -5.12, 5.12
    eps = 0.01
    
    print("Поиск глобального минимума методом Стронгина")
    print(f"f(x) = {func_str}")
    print(f"Отрезок: [{a}, {b}], eps = {eps}\n")
    
    try:
        x_min, f_min, iters, elapsed_time, pts, vals = strongin_method(
            func, a, b, eps
        )
        
        print(f"x_min = {x_min:.6f}")
        print(f"f_min = {f_min:.6f}")
        print(f"Итераций = {iters}")
        print(f"Время = {elapsed_time:.4f} с\n")
        
        X = np.linspace(a, b, 1000)
        Y = [func(x) for x in X]
        x_aux, y_aux = lower_piecewise(pts, vals)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(X, Y, 'b-', lw=2, label=f'f(x) = {func_str}')
        ax.scatter(pts, vals, color='red', s=30, label=f'Точки ({len(pts)})', 
                   alpha=0.7, zorder=3)
        ax.plot(x_aux, y_aux, 'g--', lw=1.5, label='Оценка снизу', alpha=0.7)
        ax.scatter([x_min], [f_min], color='lime', s=200, marker='*',
                   edgecolors='darkgreen', lw=2, label=f'Минимум: x={x_min:.4f}, f={f_min:.4f}',
                   zorder=4)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title('Поиск глобального минимума методом Стронгина', 
                     fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        pdf_filename = "result_strongin_method.pdf"
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            
            report_fig = plt.figure(figsize=(8.5, 11))
            report_text = f"""
Метод Стронгина

Функция: f(x) = {func_str}
Отрезок: [{a}, {b}]
Точность eps = {eps}

РЕЗУЛЬТАТЫ:
-----------
x_min = {x_min:.8f}
f_min = {f_min:.8f}
Итераций = {iters}
Время = {elapsed_time:.4f} с
Точек испытаний = {len(pts)}

ТЕОРИЯ:
-------
Липшицевая функция: |f(x₁) - f(x₂)| ≤ L·|x₁ - x₂|
"""
            report_fig.text(0.1, 0.95, report_text, family='monospace', 
                          va='top', fontsize=9, wrap=True)
            plt.axis('off')
            pdf.savefig(report_fig, bbox_inches='tight')
            plt.close(report_fig)
        
        plt.close(fig)
        print(f"✓ Результаты сохранены в {pdf_filename}")
        
        plt.show()
        
    except ValueError as e:
        print(f"Ошибка входных данных: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
