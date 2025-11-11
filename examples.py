import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi
from matplotlib.backends.backend_pdf import PdfPages
from strongin import strongin_method, lower_piecewise


def rastrigin(x: float) -> float:
    return x**2 - 10*cos(2*pi*x) + 10


def griewank(x: float) -> float:
    return x**2/4000 - cos(x) + 1


def modified_shekel(x: float) -> float:
    return -(1 / ((x - 0.1)**2 + 0.01) + 1 / ((x - 0.9)**2 + 0.04))


def multimodal(x: float) -> float:
    return sin(x) + sin(10*x/3) + x**2/100


def noisy_quadratic(x: float) -> float:
    return (x - 2)**2 + 0.1*sin(30*x)


test_functions = [
    {
        "name": "Функция Растригина",
        "func": rastrigin,
        "a": -5.12,
        "b": 5.12,
        "eps": 0.01,
        "true_min": 0.0
    },
    {
        "name": "Функция Гриванка",
        "func": griewank,
        "a": -10,
        "b": 10,
        "eps": 0.01,
        "true_min": 0.0
    },
    {
        "name": "Модифицированная функция Шекеля",
        "func": modified_shekel,
        "a": 0,
        "b": 1,
        "eps": 0.001,
        "true_min": None
    },
    {
        "name": "Многоэкстремальная функция",
        "func": multimodal,
        "a": -10,
        "b": 10,
        "eps": 0.01,
        "true_min": None
    },
    {
        "name": "Функция с шумом",
        "func": noisy_quadratic,
        "a": -5,
        "b": 5,
        "eps": 0.01,
        "true_min": 2.0
    }
]


def run_tests():
    print("="*80)
    print("ТЕСТИРОВАНИЕ МЕТОДА СТРОНГИНА НА РАЗЛИЧНЫХ ФУНКЦИЯХ")
    print("="*80)
    
    results = []
    
    for i, test in enumerate(test_functions, 1):
        print(f"\n{i}. {test['name']}")
        print("-" * 80)
        
        try:
            f = test['func']
            a, b = test['a'], test['b']
            eps = test['eps']
            
            x_min, f_min, iterations, elapsed_time, points, values = strongin_method(
                f, a, b, eps, r=2.0
            )
            
            print(f"   x_min      = {x_min:.6f}")
            print(f"   f_min      = {f_min:.6f}")
            
            if test['true_min'] is not None:
                error = abs(f_min - test['true_min'])
                print(f"   Погрешность = {error:.6e}")
            
            print(f"   Итераций    = {iterations}")
            print(f"   Время       = {elapsed_time:.4f} с")
            
            results.append({
                'name': test['name'],
                'x_min': x_min,
                'f_min': f_min,
                'iterations': iterations,
                'time': elapsed_time,
                'points': points,
                'values': values,
                'func': f,
                'a': a,
                'b': b,
                'true_min': test['true_min']
            })
            
        except Exception as e:
            print(f"   ОШИБКА: {e}")
            continue
    
    if not results:
        print("\nНе удалось выполнить ни одного теста!")
        return
    
    print("\n" + "="*80)
    print("СОЗДАНИЕ СВОДНОГО ГРАФИКА...")
    print("="*80)
    
    n_tests = len(results)
    n_cols = 3
    n_rows = (n_tests + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    if n_tests == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        X = np.linspace(result['a'], result['b'], 1000)
        Y = [result['func'](x) for x in X]
        
        x_aux, y_aux = lower_piecewise(result['points'], result['values'])
        
        ax.plot(X, Y, 'b-', linewidth=1.5, label='f(x)')
        if x_aux and y_aux:
            ax.plot(x_aux, y_aux, 'g--', linewidth=1, alpha=0.5, label='Оценка снизу')
        ax.scatter(result['points'], result['values'], color='red', 
                   s=20, alpha=0.5, label='Точки испытаний', zorder=3)
        ax.scatter([result['x_min']], [result['f_min']], color='lime', 
                   s=100, marker='*', edgecolors='darkgreen', linewidths=1.5,
                   label=f"Min: {result['f_min']:.3f}", zorder=4)
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('f(x)', fontsize=10)
        title = f"{result['name']}\n{result['iterations']} итераций"
        if result['true_min'] is not None:
            error = abs(result['f_min'] - result['true_min'])
            title += f", погр.={error:.2e}"
        ax.set_title(title, fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_tests, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    pdf_filename = "examples_test_results.pdf"
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        
        fig_table = plt.figure(figsize=(11, 8.5))
        fig_table.text(0.5, 0.95, 'СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ', 
                       ha='center', fontsize=16, fontweight='bold')
        
        table_data = []
        table_data.append(['Функция', 'x_min', 'f_min', 'Итераций', 'Время (с)', 'Погрешность'])
        
        for r in results:
            error_str = "-"
            if r['true_min'] is not None:
                error = abs(r['f_min'] - r['true_min'])
                error_str = f"{error:.2e}"
            
            table_data.append([
                r['name'],
                f"{r['x_min']:.4f}",
                f"{r['f_min']:.4f}",
                f"{r['iterations']}",
                f"{r['time']:.4f}",
                error_str
            ])
        
        table = plt.table(cellText=table_data, cellLoc='left',
                         loc='center', bbox=[0.05, 0.2, 0.9, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.axis('off')
        pdf.savefig(fig_table, bbox_inches='tight')
        plt.close(fig_table)
    
    plt.close(fig)
    print(f"\n✓ Результаты сохранены в файл: {pdf_filename}")
    print("\nТестирование завершено!")
    plt.show()


if __name__ == "__main__":
    run_tests()
