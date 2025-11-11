import numpy as np
import time
from typing import Callable, Tuple, List, Optional


def strongin_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 0.01,
    r: float = 2.0,
    max_iter: int = 10000
) -> Tuple[float, float, int, float, List[float], List[float]]:
    if a >= b:
        raise ValueError(f"Левая граница a={a} должна быть меньше правой границы b={b}")
    if eps <= 0:
        raise ValueError(f"Точность eps={eps} должна быть положительной")
    if r <= 0:
        raise ValueError(f"Параметр надежности r={r} должен быть положительным")
    
    start_time = time.time()
    points = [a, b]
    values = [f(a), f(b)]
    iterations = 0
    
    EPS_MIN = 1e-12
    
    while iterations < max_iter:
        iterations += 1
        
        sorted_indices = np.argsort(points)
        points = [points[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        n = len(points)
        
        M = 0.0
        for i in range(1, n):
            dx = points[i] - points[i - 1]
            if dx > EPS_MIN:
                lipschitz_estimate = abs(values[i] - values[i - 1]) / dx
                M = max(M, lipschitz_estimate)
        
        if M < EPS_MIN:
            M = 1.0
        
        m = r * M
        
        R = []
        for i in range(1, n):
            dx = points[i] - points[i - 1]
            if dx > EPS_MIN:
                R_i = (m * dx + 
                       (values[i] - values[i - 1]) ** 2 / (m * dx) - 
                       2 * (values[i] + values[i - 1]))
                R.append((R_i, i))
        
        if not R:
            break
        
        R.sort(reverse=True, key=lambda x: x[0])
        _, i_max = R[0]
        
        dx = points[i_max] - points[i_max - 1]
        dz = values[i_max] - values[i_max - 1]
        x_new = (points[i_max] + points[i_max - 1]) / 2 - dz / (2 * m)
        x_new = max(points[i_max - 1], min(x_new, points[i_max]))
        
        points.append(x_new)
        values.append(f(x_new))
        
        if dx < eps:
            break
    
    elapsed_time = time.time() - start_time
    
    min_idx = np.argmin(values)
    x_min = points[min_idx]
    f_min = values[min_idx]
    
    return x_min, f_min, iterations, elapsed_time, points, values


def lower_piecewise(
    points: List[float],
    values: List[float],
    r: float = 2.0
) -> Tuple[List[float], List[float]]:
    if len(points) != len(values) or len(points) < 2:
        return [], []
    
    sorted_indices = np.argsort(points)
    points_sorted = [points[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]
    
    M = 0.0
    EPS_MIN = 1e-12
    
    for i in range(1, len(points_sorted)):
        dx = points_sorted[i] - points_sorted[i - 1]
        if dx > EPS_MIN:
            lipschitz_estimate = abs(values_sorted[i] - values_sorted[i - 1]) / dx
            M = max(M, lipschitz_estimate)
    
    if M < EPS_MIN:
        M = 1.0
    
    m = r * M
    
    x_line, y_line = [], []
    
    for i in range(1, len(points_sorted)):
        x1, x2 = points_sorted[i - 1], points_sorted[i]
        z1, z2 = values_sorted[i - 1], values_sorted[i]
        
        x_mid = np.clip((x1 + x2) / 2 - (z2 - z1) / (2 * m), x1, x2)
        y_mid = min(z1 - m * abs(x_mid - x1), z2 - m * abs(x_mid - x2))
        
        x_line.extend([x1, x_mid, x2])
        y_line.extend([z1, y_mid, z2])
    
    return x_line, y_line
