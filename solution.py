import pandas as pd
import numpy as np
from hyppo.ksample import MMD


chat_id = 1019285902 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    _, pvalue = MMD(compute_kernel='rbf', gamma = 0.1).test(x, y)
    ans = True if pvalue < 0.05 else False
    return ans # Ваш ответ, True или False
