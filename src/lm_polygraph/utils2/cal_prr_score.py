import copy

import numpy as np
from collections import defaultdict
import json

from src.lm_polygraph.ue_metrics.ue_metric import (
    # UEMetric,
    get_random_scores,
    normalize_metric,
)

def json_file_to_list(file_name):
    res = []
    with open(file_name, 'r') as ini_f:
        ini_str = ini_f.readlines()
        for ele in ini_str:
            mid_res = json.loads(ele) # res['pred'], res['label']
            res.append(mid_res)
    return res

def Cal_Prr_Score(
estimations_file,
gen_metrics_file,
ass_gt_file_name=None,
gpt_gt_file_name=None,
# ignore_keywords=None,
ue_metrics=None,
rerank_est=False,
size_per_batch = 1889,
):
    # transfer dataformat for PRR calculation
    estimations_ini = json_file_to_list(estimations_file)
    gen_metrics_ini = json_file_to_list(gen_metrics_file)
    try:
        est_keys = list(estimations_ini[0].keys())
        gen_keys = list(gen_metrics_ini[0].keys())
    except:
        estimations_ini = estimations_ini[0]
        gen_metrics_ini = gen_metrics_ini[0]
        est_keys = list(estimations_ini[0].keys())
        gen_keys = list(gen_metrics_ini[0].keys())

    print("len(estimations_ini) and len(gen_metrics_ini)")
    print(len(estimations_ini), len(gen_metrics_ini))

    if rerank_est==True:

        if len(gen_metrics_ini) > size_per_batch: # it means that use concatenate
            mini_bs_l = size_per_batch
            prev_list = []
            back_list = []
            batch_num = int(len(gen_metrics_ini)/size_per_batch)
            for i in range(0, batch_num*2, 2):
                prev_list.extend(estimations_ini[i*mini_bs_l:(i+1)*mini_bs_l])
                back_list.extend(estimations_ini[(i+1)*mini_bs_l:(i+2)*mini_bs_l])
            estimations_ini = []
            estimations_ini.extend(prev_list)
            estimations_ini.extend(back_list)



    if len(estimations_ini) == (2 * len(gen_metrics_ini)):
        bs_l = len(gen_metrics_ini)
        estimations_ini_new = []
        for i in range(bs_l):
            # if i == 1888:
                # print('start debug')
            estimations_ini_new.append(estimations_ini[i])
            for ky in estimations_ini[i+bs_l].keys():
                if ky not in estimations_ini[i].keys():
                    estimations_ini_new[i][ky] = estimations_ini[i+bs_l][ky]
        estimations_ini = copy.deepcopy(estimations_ini_new)
        est_keys = list(estimations_ini[0].keys())


    assert len(estimations_ini)==len(gen_metrics_ini)
    assert len(estimations_ini) != 0


    estimations = defaultdict(list)
    gen_metrics = defaultdict(list)

    # jsq = 0
    for ele in estimations_ini:
        # print(f"current jsg={jsq}")
        for k in est_keys:
            estimations[k].append(ele[k])
        # jsq +=1

    for ele in gen_metrics_ini:
        for k in gen_keys:
            gen_metrics[k].append(ele[k])

    metrics = {}

    for ek, estimator_values in estimations.items():
        if '#' not in ek:
            continue
        print(f"ek is {ek}")
        for eg, generation_metric in gen_metrics.items():
            if "#" not in eg:
                continue
            ek2 = ek.split('#')
            eg2 = eg.split('#')
            e_level, e_name = ek2[0], ek2[1]
            gen_level, gen_name = eg2[0], eg2[1]
            for ue_metric in ue_metrics:
                if gen_level != e_level:
                    continue

                if len(estimator_values) != len(generation_metric):
                    raise Exception(
                        f"Got different number of metrics for {e_name} and {gen_name}: "
                        f"{len(estimator_values)} and {len(generation_metric)}"
                    )

                ue, metric, selected_ids = _delete_nans(  # selected_ids is the samples without nan cases
                    estimator_values, generation_metric
                )
                if len(ue) == 0:
                    metrics[e_level, e_name, gen_name, str(ue_metric)] = np.nan
                else:

                    inputs_no_nans = np.array(estimations["input_texts"])[
                        selected_ids
                    ]
                    rec_ue, rec_metric = _recombine_data(ue, metric, inputs_no_nans)  # ue_metrics & gen_metrics


                    rec_metric = np.array(rec_metric)
                    oracle_score = ue_metric(-rec_metric, rec_metric)
                    random_score = get_random_scores(ue_metric,
                                                     rec_metric)  # random is the mean of 1000-times randomly assigning scores
                    ue_metric_val = ue_metric(rec_ue, rec_metric)
                    metrics[
                        e_level, e_name, gen_name, str(ue_metric)
                    ] = ue_metric_val
                    metrics[
                        e_level, e_name, gen_name, str(ue_metric) + "_normalized"
                    ] = normalize_metric(ue_metric_val, oracle_score, random_score)  # here is the formula in paper


    return metrics


def _delete_nans(ue, metric):
    new_ue, new_metric, selected_ids = [], [], []
    for i in range(len(metric)):
        if not np.isnan(metric[i]) and not np.isnan(ue[i]):
            if not isinstance(ue[i], complex):
                new_ue.append(ue[i])
            else:
                new_ue.append(ue[i].real)
            new_metric.append(metric[i])
            selected_ids.append(i)
    return new_ue, new_metric, selected_ids

def _recombine_data(ue, gen_metric, inputs):
    ue = np.array(ue)
    gen_metric = np.array(gen_metric)

    # np.unique() with return_counts=True?
    recombined_inputs = defaultdict(list)
    len_inputs = len(inputs)
    for i, input_text in enumerate(inputs):

        # case for xsum divide operation with random seed 42
        if len_inputs > 9000:
            if (i >= 3778 and i < 5667 ):
                recombined_inputs[input_text].append(i)
            else:
                recombined_inputs[input_text + f"{i}"].append(i)
        else:
            recombined_inputs[input_text].append(i)

    # check
    for k,v in recombined_inputs.items():
        if len(v) >= 7:
            print("cal_prr", v)
    print('finished one round')

    recombined_ue, recombined_gen_metric = [], []
    for input_text, ids in recombined_inputs.items():
        recombined_ue.append(ue[ids].mean())
        # Assumes that metric is bigger for better generations! # comment: need to pay attention to
        recombined_gen_metric.append(gen_metric[ids].max())

    print(f"recombined_ue has len {len(recombined_ue)}")
    print(f"recombined_gen_metric has len {len(recombined_gen_metric)}")
    return recombined_ue, recombined_gen_metric