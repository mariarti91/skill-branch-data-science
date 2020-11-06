import math
from itertools import accumulate

# посчитать значений производной функции $\cos(x) + 0.05x^3 + \log_2{x^2}$ в точке $x = 10$. 
# Ответ округлить до 2-го знака. 
# Пожалуйста назовите функцию `derivation`, функция должна принимать точку, в которой нужно вычислить значение производной, 
# и функцию, производную которой мы хотим вычислить.
def derivation(x, function, d_x = 10**-5):
    return round((float(function(x + d_x) - function(x))/d_x), 2)

# посчитать значение градиента функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$ в точке $(10, 1)$.
# Пожалуйста назовите функцию `gradient`, функция должна принимать список с координатами точки, в которой нужно вычислить значение производной, 
# и функцию, производную которой мы хотим вычислить. Ответ округлить до 2-го знака.
def gradient(x, function, d_x = 10**-5):
    res = list()
    for i in range(len(x)):
        x_ = x.copy()
        x_[i] += d_x
        d_f = float(function(x_) - function(x))
        res.append(round(d_f/d_x, 2))
    return res

# найти точку минимуму для функции $\cos(x) + 0.05x^3 + \log_2{x^2}$. Зафиксировать параметр $\epsilon = 0.001$, 
# начальное значение принять равным 10. Выполнить 50 итераций градиентного спуска. 
# Ответ округлить до второго знака; Пожалуйста назовите функцию `gradient_optimization_one_dim`. Функция должна принимать на вход функцию, которую требуется оптимизировать.
def gradient_optimization_one_dim(function, x = 10, max_iterations = 50, epsilone = 10**-3):
    while max_iterations:
        max_iterations -= 1
        next_x = x - epsilone * derivation(x, function)
        x = next_x
    return round(x, 2)

# найти точку минимуму для функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$. Зафиксировать параметр $\epsilon = 0.001$, 
# начальные значения весов принять равным [4, 10]. Выполнить 50 итераций градиентного спуска. 
# Ответ округлить до второго знака; Пожалуйста назовите функцию `gradient_optimization_multi_dim`.
def gradient_optimization_multi_dim(function, x = [4, 10], max_iterations = 50, epsilone = 10**-3):
    # двигаем x на величину -alpha*F'(x), alpha = |F'(x)| * epsilone
    def calc_next_point(x, grad):
        for i in range(len(x)):
            x[i] -= round(epsilone * grad[i], 2)
        return x

    while max_iterations:
        max_iterations -= 1
        grad_f = gradient(x, function)
        next_x = calc_next_point(x, grad_f)
        x = next_x
    return [round(i, 2) for i in x]

if __name__ == "__main__":
    fun_1 = lambda x : math.cos(x) + 0.05 * x**3 + math.log2(x**2)
    x_1 = 10
    d_fun_1 = lambda x : -math.sin(x) + 0.15 * x**2 + 2 / x / math.log(2)

    print('{}'.format(derivation(x_1, fun_1)))
    # print('{:.2f}'.format(d_fun_1(x_1)))

    fun_2 = lambda x: x[0]**2 * math.cos(x[1]) + 0.05 * x[1]**3 + 3 * x[0]**3 * math.log2(x[1]**2)
    x_2 = [10, 1]
    d_fun_2 = lambda x: [ round(2*x[0] * math.cos(x[1]) + 9 * x[0]**2 * math.log2(x[1]**2), 2), 
        round(- x[0]**2 * math.sin(x[1]) + 0.15 * x[1]**2 + 6 * x[0]**3 / x[1] / math.log(2), 2)]
    print(gradient(x_2, fun_2))
    # print('{}'.format(d_fun_2(x_2)))

    print(gradient_optimization_one_dim(fun_1))
    print(gradient_optimization_multi_dim(fun_2))