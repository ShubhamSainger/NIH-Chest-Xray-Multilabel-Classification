from tensorflow import reduce_sum, cast, float16, int32
from configs import configs

def accuracy( y_true, y_pred):
    
    y_pred = cast(y_pred > configs.DECISION_THRESHOLD, int32)
    
    t_p = reduce_sum(cast((y_true == 1) & (y_pred == 1), float16))
    t_n = reduce_sum(cast((y_true == 0) & (y_pred == 0), float16))
    f_p = reduce_sum(cast((y_true == 0) & (y_pred == 1), float16))
    f_n = reduce_sum(cast((y_true == 1) & (y_pred == 0), float16))

    accuracy = (t_p + t_n) / (t_p + t_n + f_p + f_n)
    return accuracy

def recall(y_true, y_pred):

    y_pred = cast(y_pred > configs.DECISION_THRESHOLD, int32)
    
    t_p = reduce_sum(cast((y_true == 1) & (y_pred == 1), float16))
    f_n = reduce_sum(cast((y_true == 1) & (y_pred == 0), float16))

    recall = t_p / (t_p + f_n + 10**(-5))

    return recall

def precision (y_true, y_pred):

    y_pred = cast(y_pred > configs.DECISION_THRESHOLD, int32)

    t_p = reduce_sum(cast((y_true == 1) & (y_pred == 1), float16))
    f_p = reduce_sum(cast((y_true == 0) & (y_pred == 1), float16))

    precision = t_p / (t_p + f_p  + 10**(-5))

    return precision


def f1_score(y_true, y_pred):

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    f_1 = (2 * precision * recall) / (precision + recall)

    return f_1

