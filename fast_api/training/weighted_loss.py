from tensorflow.math import reduce_mean, reduce_sum, log

def Weighted_Binary_CLE (y_true, y_pred):
    
    n_possitive = reduce_sum(y_true, axis = 1)
    n_negetive = reduce_sum(1-y_true, axis = 1)

    beta_p = (n_possitive + n_negetive)/(n_possitive + 10**(-7))
    beta_n = (n_possitive + n_negetive)/(n_negetive + 10**(-7))

    first_term =  beta_p * reduce_sum(y_true * (-1) * log(y_pred + 10**(-7)), axis = 1)
    second_term =  beta_n * reduce_sum((1 - y_true) * (-1) * log((1-y_pred) + 10**(-7)), axis = 1)

    loss = reduce_mean(first_term + second_term)

    return loss