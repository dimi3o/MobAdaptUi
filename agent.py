from numpy import np

class Agent:
    pass

# Learning Intrinsic Symbolic Rewards in RL (2020)
# https://arxiv.org/pdf/2010.03694.pdf

def add(left, right):
    return left + right

def subtract(left, right):
    return left - right

def multiply(left, right):
    return left * right

def cos(left):
    return np.cos(left)

def sin(left):
    return np.sin(left)

def tan(left):
    return np.tan(left)

def max(nums):
    return np.maxmimum(nums)

def min(nums):
    return np.minimum(nums)

def pass_greater(left, right):
    if left > right: return left
    return right

def pass_smaller(left, right):
    if left < right: return left
    return right

def equal_to(left, right):
    return float(left == right)

def gate(left, right, condtion):
    if condtion <= 0: return left
    else: return right

def square(left):
    return left * left

def is_negative(left):
    if left < 0: return 1.0
    return 0.0

def div_by_100(left):
    return left / 100.0

def div_by_10(left):
    return left / 10.0

def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

# An example of a discovered symbolic reward on PixelCopter. We unroll the correspond-
# ing symbolic tree into Python-like code that can be parsed and debugged.
# {si} represent state observations.
def get_intrinsic_reward(s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7):
    p_1 = tan(cos(s_4)); p_2 = cos(s_3); p_3 = pass_smaller(p_1, p_2)
    x_1 = multiply(-1, abs(subtract(s_7, p_3)))
    q_1 = multiply(-1, abs(subtract(1, s_4)))
    q_2 = max([s_2, 1, s_7, q_1, 0])
    q_3 = max([q_2, s_7, cos(0), multiply(s_0, s_6), multiply(s_5, subtract(s_6, 1))])
    y_1 = div_by_10(q_3)
    y_2 = square(s_7)
    y_3 = protected_div(1, div_by_100(s_0))
    x_2 = gate(y_1, y_2, y_3)
    z = equal_to(x_2, x_1)
    reward = add(0, pass_smaller(div_by_10(s_7), z))
    return reward