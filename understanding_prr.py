import numpy as np

def normalize(target):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target
def understand_ue_metric(estimator, target
                    ):
    target = normalize(target)  # min-max normalization
    risk = 1 - np.array(target)
    cr_pair = list(zip(estimator, risk))
    cr_pair.sort(key=lambda x: x[0])
    cumulative_risk = np.cumsum([x[1] for x in cr_pair])
    if True:
        cumulative_risk = cumulative_risk / np.arange(1, len(estimator) + 1)
    return cumulative_risk.mean()

tar = [0.1, 0.8, 0.7, 0.5, 0.2, 0.4]

est = [7, 6, 5, 4, 3, 2]

tar = np.array(tar)

oracle_score = understand_ue_metric(-tar, tar)
est_score = understand_ue_metric(est, tar)

print(oracle_score)
print(est_score)
