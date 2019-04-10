from functools import reduce

# Compute the accuracy
def accuracy(a, b):
    common_keys = set(a).intersection(b)
    all_keys = set(a).union(b)
    score = len(common_keys) / len(all_keys)  # key score
    if score == 0:
        return score
    else:  # value score
        pred = {}
        for k in common_keys:
            pred[k] = b[k]
        # true_values_sum = reduce(lambda x,y:int(x)+int(y),a.values())
        all_keys = dict.fromkeys(all_keys, 0)
        for k in a.keys():
            all_keys.update({k: a[k]})
        for k in b.keys():
            all_keys.update({k: b[k]})
        true_values_sum = reduce(lambda x, y: int(x) + int(y), all_keys.values())
        pred_values_sum = reduce(lambda x, y: int(x) + int(y), pred.values())
        val_score = int(pred_values_sum) / int(true_values_sum)
        return (score + val_score) / 2  # avg


# Compute the precision
def precision(a, b):
    # return len(set(a).intersection(b))/len(a)
    common_keys = set(a).intersection(b)
    score = len(common_keys) / len(a)
    if score == 0:
        return score
    else:
        pred = {}
        for k in common_keys:
            pred[k] = b[k]
        true_values_sum = reduce(lambda x, y: int(x) + int(y), a.values())
        pred_values_sum = reduce(lambda x, y: int(x) + int(y), pred.values())
        val_score = int(pred_values_sum) / int(true_values_sum)
        return (score + val_score) / 2


# Compute the recall
def recall(a, b):
    common_keys = set(a).intersection(b)
    score = len(common_keys) / len(b)
    if score == 0:
        return score
    else:
        pred = {}
        for k in common_keys:
            pred[k] = b[k]
        true_values_sum = reduce(lambda x, y: int(x) + int(y), b.values())
        pred_values_sum = reduce(lambda x, y: int(x) + int(y), pred.values())
        val_score = int(pred_values_sum) / int(true_values_sum)
        return (score + val_score) / 2


# Compute the total time taken

# Compute the time between each scene, do this only after completing everything
