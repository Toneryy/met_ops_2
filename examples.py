import numpy as np
import matplotlib.pyplot as plt
import time
from math import cos, sin, pi, sqrt, exp
from matplotlib.backends.backend_pdf import PdfPages

def strongin_method(f, a, b, eps, r=2.0, max_iter=10000):
    start_time = time.time()
    points = [a, b]
    values = [f(a), f(b)]
    iterations = 0
    
    while iterations < max_iter:
        iterations += 1
        sorted_indices = np.argsort(points)
        points = [points[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        n = len(points)
        
        M = 0
        for i in range(1, n):
            dx = points[i] - points[i-1]
            if dx > 0:
                lipschitz_estimate = abs(values[i] - values[i-1]) / dx
                M = max(M, lipschitz_estimate)
        
        if M == 0:
            M = 1.0
        
        m = r * M
        
        R = []
        for i in range(1, n):
            dx = points[i] - points[i-1]
            R_i = m * dx + (values[i] - values[i-1])**2 / (m * dx) - 2 * (values[i] + values[i-1])
            R.append((R_i, i))
        
        R.sort(reverse=True)
        max_R, i_max = R[0]
        
        dx = points[i_max] - points[i_max-1]
        dz = values[i_max] - values[i_max-1]
        x_new = (points[i_max] + points[i_max-1]) / 2 - dz / (2 * m)
        x_new = max(points[i_max-1], min(x_new, points[i_max]))
        
        points.append(x_new)
        values.append(f(x_new))
        
        if dx < eps:
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    min_idx = np.argmin(values)
    x_min = points[min_idx]
    f_min = values[min_idx]
    
    return x_min, f_min, iterations, elapsed_time, points, values

test_functions = [
    {
        "name": "Функция Растригина",
        "func": lambda x: x**2 - 10*cos(2*pi*x) + 10,
        "a": -5.12,
        "b": 5.12,
        "eps": 0.01,
        "true_min": 0.0
    },
    {
        "name": "Функция Гриванка",
        "func": lambda x: x**2/4000 - cos(x) + 1,
        "a": -10,
        "b": 10,
        "eps": 0.01,
        "true_min": 0.0
    },
    {
        "name": "Модифицированная функция Шекеля",
        "func": lambda x: -(1 / ((x - 0.1)**2 + 0.01) + 1 / ((x - 0.9)**2 + 0.04)),
        "a": 0,
        "b": 1,
        "eps": 0.001,
        "true_min": None
    },
    {
        "name": "Многоэкстремальная функция",
        "func": lambda x: sin(x) + sin(10*x/3) + x**2/100,
        "a": -10,
        "b": 10,
        "eps": 0.01,
        "true_min": None
    },
    {
        "name": "Функция с шумом",
        "func": lambda x: (x - 2)**2 + 0.1*sin(30*x),
        "a": -5,
        "b": 5,
        "eps": 0.01,
        "true_min": 2.0
    }
]

print("="*80)
print("ТЕСТИРОВАНИЕ МЕТОДА СТРОНГИНА НА РАЗЛИЧНЫХ ФУНКЦИЯХ")
print("="*80)

results = []

for i, test in enumerate(test_functions, 1):
    print(f"\n{i}. {test['name']}")
    print("-" * 80)
    
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
        'b': b
    })

print("\n" + "="*80)
print("СОЗДАНИЕ СВОДНОГО ГРАФИКА...")
print("="*80)

n_tests = len(results)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, result in enumerate(results):
    ax = axes[idx]
    
    X = np.linspace(result['a'], result['b'], 1000)
    Y = [result['func'](x) for x in X]
    
    ax.plot(X, Y, 'b-', linewidth=1.5, label='f(x)')
    ax.scatter(result['points'], result['values'], color='red', 
               s=20, alpha=0.5, label='Точки испытаний')
    ax.scatter([result['x_min']], [result['f_min']], color='lime', 
               s=100, marker='*', edgecolors='darkgreen', linewidths=1.5,
               label=f"Min: {result['f_min']:.3f}")
    
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.set_title(f"{result['name']}\n{result['iterations']} итераций", fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

if n_tests < 6:
    fig.delaxes(axes[-1])

plt.tight_layout()

pdf_filename = "examples_test_results.pdf"
with PdfPages(pdf_filename) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    
    fig_table = plt.figure(figsize=(11, 8.5))
    fig_table.text(0.5, 0.95, 'СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ', 
                   ha='center', fontsize=16, fontweight='bold')
    
    table_data = []
    table_data.append(['Функция', 'x_min', 'f_min', 'Итераций', 'Время (с)'])
    
    for r in results:
        table_data.append([
            r['name'],
            f"{r['x_min']:.4f}",
            f"{r['f_min']:.4f}",
            f"{r['iterations']}",
            f"{r['time']:.4f}"
        ])
    
    table = plt.table(cellText=table_data, cellLoc='left',
                     loc='center', bbox=[0.1, 0.3, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.axis('off')
    pdf.savefig(fig_table, bbox_inches='tight')
    plt.close(fig_table)

print(f"\n✓ Результаты сохранены в файл: {pdf_filename}")
print("\nТестирование завершено!")
plt.show()

