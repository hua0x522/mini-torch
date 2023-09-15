import math 

def mul(x: float, y: float) -> float: 
    "f(x, y) = x * y"
    return x * y

def id(x: float) -> float: 
    "f(x) = x"
    return x 

def add(x: float, y: float) -> float:
    "f(x, y) = x + y"
    return x + y

def neg(x: float) -> float:
    "f(x) = -x"
    return -x 

def lt(x: float, y: float) -> float:
    "f(x) = 1.0 if x is less than y else 0.0"
    if x < y:
        return 1.0
    else: 
        return 0.0 
    
def eq(x: float, y: float) -> float: 
    "f(x) = 1.0 if x is equal to y else 0.0"
    if x == y:
        return 1.0
    else: 
        return 0.0
    
def max(x: float, y: float) -> float:
    "f(x) = x if x is greater than y else y"
    if x > y:
        return x
    else: 
        return y 

def is_close(x: float, y: float) -> float: 
    "f(x) = |x - y| < 1e-2"
    if (abs(x - y) < 1e-2):
        return 1.0
    else: 
        return 0.0

def sigmoid(x: float) -> float:
    "f(x) = 1 / (1 + e^(-x)) if x >= 0 else e^x / (1 + e^x)"
    if x >= 0:
        return 1 / (1 + math.e ** (-x))
    else: 
        return (math.e ** x) / (1 + math.e ** x)
    
def relu(x: float) -> float:
    "f(x) = x if x is greater than 0, else 0"
    if x > 0:
        return x 
    else: 
        return 0 
    
EPS = 1e-6

def log(x: float) -> float:
    "f(x) = log(x)"
    return math.log(x + EPS)

def exp(x: float) -> float:
    "f(x) = e^{x}"
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    "If f = log as above, compute $d x f'(x)$"
    pass 

def inv(x: float) -> float:
    "f(x) = 1/x"
    return 1 / x


def inv_back(x: float, d: float) -> float:
    "If f(x) = 1/x compute d x f'(x)"
    return -d / (x ** 2)


def relu_back(x: float, d: float) -> float:
    "If f = relu compute d x f'(x)"
    if x > 0:
        return d 
    else: 
        return 0
