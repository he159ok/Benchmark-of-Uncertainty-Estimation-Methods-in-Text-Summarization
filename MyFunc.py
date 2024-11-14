import pandas as pd
from src.lm_polygraph.generation_metrics import RougeMetric, BartScoreSeqMetric, ModelScoreSeqMetric, ModelScoreTokenwiseMetric, SUMMACMetric,CTCFactConsistency, P_K_Correlation, chatGPT_Metric, \
    UniEval, UniEval_Coherence, UniEval_Consistency, UniEval_Relevance, UniEval_Fluency, claude_Metric, chatGPT_Metric_35_Def, chatGPT_Metric_40_Def, chatGPT_Metric_35_Def_0, chatGPT_Metric_35_Def_Both
import json

import os

from collections import defaultdict

import numpy as np

import copy

from src.lm_polygraph.ue_metrics.ue_metric import (
    get_random_scores,
    normalize_metric,
)

def get_nlg_metric(nlg_metric_list, assist_file_name, args):
    res = []
    if args.model_type == 'w':
        depend=["greedy_texts"]
    else:
        depend = ["blackbox_greedy_texts"]
    for ele in nlg_metric_list:
        if ele == 'rouge':
            res.append(RougeMetric(rouge_name='rougeL', depend=depend))
        elif ele == 'bart':
            res.append(BartScoreSeqMetric('rh'))
        elif ele == 'summac':
            res.append(SUMMACMetric(model_card='microsoft/deberta-base-mnli', depend=depend))
        elif ele == 'ctc':
            res.append(CTCFactConsistency(depend=depend, set_align='E-roberta'))
        elif ele == 'spearmanr':
            res.append(P_K_Correlation(depend=depend, cor_type='spearmanr'))
        elif ele == 'kendalltau':
            res.append(P_K_Correlation(depend=depend, cor_type='kendalltau'))
        elif ele == 'chatGPT':
            chat_metric = chatGPT_Metric(
                depend=depend,
                model="gpt-3.5-turbo",
                             logprobs=True,
                             top_logprobs=1,
                             task="text summarization",
                             view="overall",
                             file_name=args.chatgpt_file_name,
                             temperature=args.eval_temperature,
                             open_ai_token=args.api_token
            )
            res.append(chat_metric)
        elif ele == 'chatGPT_def_35':
            chat_metric = chatGPT_Metric_35_Def(
                depend=depend,
                model="gpt-3.5-turbo",
                             logprobs=True,
                             top_logprobs=1,
                             task="text summarization",
                             view="overall",
                             file_name=args.chatgpt_file_name[:-5] + '_def_35.json',
                             temperature=args.eval_temperature,
                             open_ai_token=args.api_token
            )
            res.append(chat_metric)

        elif ele == 'chatGPT_def_0_35':
            chat_metric = chatGPT_Metric_35_Def_0(
                depend=depend,
                model="gpt-3.5-turbo",
                             logprobs=True,
                             top_logprobs=1,
                             task="text summarization",
                             view="overall",
                             file_name=args.chatgpt_file_name[:-5] + '_def_0_35.json',
                             temperature=args.eval_temperature,
                             open_ai_token=args.api_token
            )
            res.append(chat_metric)



        elif ele == 'chatGPT_def_both_35':
            chat_metric = chatGPT_Metric_35_Def_Both(
                depend=depend,
                model="gpt-3.5-turbo",
                             logprobs=True,
                             top_logprobs=1,
                             task="text summarization",
                             view="overall",
                             file_name=args.chatgpt_file_name[:-5] + '_def_both_35.json',
                             temperature=args.eval_temperature,
                             open_ai_token=args.api_token
            )
            res.append(chat_metric)

        elif ele == 'chatGPT_def_40':
            chat_metric = chatGPT_Metric_40_Def(
                depend=depend,
                model="gpt-4",
                             logprobs=True,
                             top_logprobs=1,
                             task="text summarization",
                             view="overall",
                             file_name=args.chatgpt_file_name[:-5] + '_def_40.json',
                             temperature=args.eval_temperature,
                             open_ai_token=args.api_token
            )
            res.append(chat_metric)


        elif ele == 'claude3':
            chat_metric = claude_Metric(
                depend=depend,
                model="claude-3-opus-20240229",
                             logprobs=True,
                             top_logprobs=1,
                             task="text summarization",
                             view="overall",
                             file_name=args.chatgpt_file_name,
                             temperature=args.eval_temperature
            )
            res.append(chat_metric)
        elif ele == 'unieval':
            res.append(UniEval(task='summarization', selected_key='overall', file_name=assist_file_name, depend=depend))
            # [coherence, consistency, fluency, relevance, overall]
        elif ele == 'unieval_coherence':
            res.append(UniEval_Coherence(task='summarization', selected_key='coherence', file_name=assist_file_name, depend=depend))
        elif ele == 'unieval_consistence':
            res.append(UniEval_Consistency(task='summarization', selected_key='consistency', file_name=assist_file_name, depend=depend))
        elif ele == 'unieval_fluency':
            res.append(UniEval_Fluency(task='summarization', selected_key='fluency', file_name=assist_file_name, depend=depend))
        elif ele == 'unieval_relevance':
            res.append(UniEval_Relevance(task='summarization', selected_key='relevance', file_name=assist_file_name, depend=depend))
        else:
            raise ValueError(f'metric={ele} is wrongly set!')
    return res

def get_nlg_metric_mullingual(nlg_metric_list):
    res = []
    for ele in nlg_metric_list:
        if ele == 'rouge':
            res.append(RougeMetric('rougeL'))
        elif ele == 'bart':
            res.append(BartScoreSeqMetric('rh'))
        elif ele == 'summac':
            res.append(SUMMACMetric('microsoft/deberta-base-mnli'))
        elif ele == 'ctc':
            res.append(CTCFactConsistency('E-roberta-large'))
        elif ele == 'spearmanr':
            res.append(P_K_Correlation('spearmanr'))
        elif ele == 'kendalltau':
            res.append(P_K_Correlation('kendalltau'))
        elif ele == 'chatGPT':
            chat_metric = chatGPT_Metric(model="gpt-3.5-turbo",
                             logprobs=True,
                             top_logprobs=1,
                             task="machine translation",
                             view="overall")
            res.append(chat_metric)
        elif ele == 'unieval':
            res.append(UniEval(task='summarization', selected_key='overall'))
            # [coherence, consistency, fluency, relevance, overall]
        elif ele == 'unieval_coherence':
            res.append(UniEval_Coherence(task='summarization', selected_key='coherence'))
        elif ele == 'unieval_consistence':
            res.append(UniEval_Consistency(task='summarization', selected_key='consistency'))
        elif ele == 'unieval_fluency':
            res.append(UniEval_Fluency(task='summarization', selected_key='fluency'))
        elif ele == 'unieval_relevance':
            res.append(UniEval_Relevance(task='summarization', selected_key='relevance'))
        else:
            raise ValueError(f'metric={ele} is wrongly set!')
    return res

def transfer_general_res_dict(results):
    final_res = {}
    for key in results.keys():
        l_key = list(key)
        str_key = '#'.join(l_key)
        final_res[str_key] = results[key]
    return final_res

def save_list_dict_to_json(d_dict, file_name):


    with open(file_name, 'a+') as f:
        json_str = json.dumps(
            d_dict
        )
        f.write(json_str)
        f.write('\n')

def obtain_nlg_metric_list(args):
    res = []
    if args.use_rouge:
        res.append('rouge')
    if args.use_bart:
        res.append('bart')
    if args.use_summac:
        res.append('summac')
    if args.use_ctc:
        res.append('ctc')
    if args.use_spearmanr:
        res.append('spearmanr')
    if args.use_kendalltau:
        res.append('kendalltau')
    if args.use_chatgpt:
        res.append('chatGPT')
    if args.use_chatgpt_35_def:
        res.append('chatGPT_def_35')
    if args.use_chatgpt_35_def_0:
        res.append('chatGPT_def_0_35')
    if args.use_chatgpt_35_def_both:
        res.append('chatGPT_def_both_35')
    if args.use_chatgpt_40_def:
        res.append('chatGPT_def_40')
    if args.use_claude:
        res.append('claude3')

    if args.use_unieval_overall:
        res.append('unieval')


    return res



def transfer_dict_list_format(dict_list, input_text_key='input_texts'):
    batch_size = len(dict_list[input_text_key])
    res_list = []
    for i in range(batch_size):
        mid_dict = {}
        for k, v in dict_list.items():
            try:
                if len(v) == batch_size:
                    mid_dict[k] = v[i]
            except:
                continue
        res_list.append(mid_dict)
    return res_list

def save_list_dict_to_json_inbat(list_dict, file_name):
    batch_size = len(list_dict)
    with open(file_name, 'a+') as f:
        for i in range(batch_size):
            json_str = json.dumps(
                list_dict[i]
            )
            f.write(json_str)
            f.write('\n')


def list_to_json(pred_list, label_list, file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)

    res = {}
    res['pred'] = pred_list
    res['label'] = label_list


    with open(file_name, 'w') as f:
        json_str = json.dumps(res)
        f.write(json_str)
    print(f"{file_name} has saved all sampled predictions ['pred'] and labels ['label'] .")

def json_to_list(file_name):
    with open(file_name, 'r') as ini_f:
        ini_str = ini_f.readline()
        res = json.loads(ini_str) # res['pred'], res['label']
    return res

def json_file_to_list(file_name):
    res = []
    with open(file_name, 'r') as ini_f:
        ini_str = ini_f.readlines()
        for ele in ini_str:
            mid_res = json.loads(ele) # res['pred'], res['label']
            res.append(mid_res)
    return res

def transfer_dict_list_format_complex(dict_list, input_text_key='input_texts', np_ele=[],
                                      list_np=["greedy_log_probs"], tensor_list=['embeddings_encoder', 'embeddings_decoder'], remove_list=[]
                                      ):
    batch_size = len(dict_list[input_text_key])
    res_list = []
    for i in range(batch_size):
        mid_dict = {}
        for k, v in dict_list.items():
            if k in remove_list:
                continue
            try:
                if len(v) == batch_size:
                    if k in list_np:
                        mid_dict[k] = v[i].tolist()
                    elif k in tensor_list:
                        mid_dict[k] = v[i].tolist()
                    elif k in np_ele:
                        mid_dict[k] = [float(ele.real) for ele in v[i]]
                    else:
                        mid_dict[k] = v[i]
            except:
                continue
        res_list.append(mid_dict)
    return res_list


def obtain_subscores_chatgpt(sample_overall_dict_file, sample_subscore_dict_file, ass_gt_file_name, ue_metric, inputs_no_nans=None, dataset_name=None):
    # read two dict file
    sample_overall_dict = json_file_to_list(sample_overall_dict_file)
    sample_subscore_dict = json_file_to_list(sample_subscore_dict_file)

    flag = 0
    if len(sample_overall_dict) == 1:
        sample_overall_dict = sample_overall_dict[0]
        assert len(sample_subscore_dict) == 1
        sample_subscore_dict = sample_subscore_dict[0]
        flag = 1

    sub_keywords = ['overall', 'coherence', 'consistency', 'fluency', 'relevance']
    ini_res = defaultdict(list)
    for i in range(len(sample_overall_dict)):
        cur_sample_overall = sample_overall_dict[i]
        # cur_input_text = cur_sample_overall['input_texts'] # different from obtain_subscores_unieval due to no usage of src text
        cur_input_text = list(sample_subscore_dict[i].keys())[0]
        # assert cur_input_text == list(sample_subscore_dict[i].keys())[0]

        cur_subscore_item = sample_subscore_dict[i][cur_input_text]
        for keyword in sub_keywords:
            ini_res[keyword].append(cur_subscore_item[keyword])

    # print(res)

    # added for the recombine
    for keyword in sub_keywords:
        ini_res[keyword] = np.array(ini_res[keyword])

    recombined_inputs = defaultdict(list)
    for i, input_text in enumerate(inputs_no_nans):

        if (i >=3778 and i<5667 and dataset_name=='xsum') or dataset_name=='aeslc':
            recombined_inputs[input_text].append(i)
        else:
            recombined_inputs[input_text+f"{i}"].append(i)

    res = defaultdict(list)
    for input_text, ids in recombined_inputs.items():
        if len(ids) >= 2:
            print("chatgpt", ids)
        for keyword in sub_keywords:
            res[keyword].append(ini_res[keyword][ids].max())
    # ended for the recombine


    # conduct random sampling calculation
    ass_ue_gt_all = json_file_to_list(ass_gt_file_name)

    if flag == 1:
        assert len(ass_ue_gt_all) == 1
        ass_ue_gt_all = ass_ue_gt_all[0]

    metric_res = {}

    for k in range(len(ass_ue_gt_all)):
        ass_ue_gt = ass_ue_gt_all[k]

        rec_ue = ass_ue_gt[0]
        rec_metric = ass_ue_gt[1]
        # print('below is the size of two values')
        # print(rec_metric, res['overall'])
        print(f"len(rec_ue) is {len(rec_ue)}")
        print(f"len(rec_metric) is {len(rec_metric)}")
        # assert rec_metric == res['overall']
        # assert rec_metric == res['overall']
        # assert len(rec_metric) == len(res['overall'])
        e_level, e_name, gen_name = ass_ue_gt[2], ass_ue_gt[3], ass_ue_gt[4]



        for keyword in sub_keywords:
            rec_metric = res[keyword]

            rec_metric = np.array(rec_metric)
            oracle_score = ue_metric(-rec_metric, rec_metric)
            random_score = get_random_scores(ue_metric,
                                             rec_metric)  # random is the mean of 1000-times randomly assigning scores
            ue_metric_val = ue_metric(rec_ue, rec_metric)

            metric_res[
                e_level, e_name, gen_name+'_'+keyword, str(ue_metric)
            ] = ue_metric_val
            metric_res[
                e_level, e_name, gen_name+'_'+keyword, str(ue_metric) + "_normalized"
            ] = normalize_metric(ue_metric_val, oracle_score, random_score)  # here is the formula in

    print(metric_res)
    # save dict file


    return metric_res

def obtain_subscores_chatgpt_cp(sample_overall_dict_file, sample_subscore_dict_file, ass_gt_file_name, ue_metric):
    # read two dict file
    sample_overall_dict = json_file_to_list(sample_overall_dict_file)
    sample_subscore_dict = json_file_to_list(sample_subscore_dict_file)

    sub_keywords = ['overall', 'coherence', 'consistency', 'fluency', 'relevance']
    res = defaultdict(list)
    for i in range(len(sample_overall_dict)):
        cur_sample_overall = sample_overall_dict[i]
        # cur_input_text = cur_sample_overall['input_texts'] # different from obtain_subscores_unieval due to no usage of src text
        cur_input_text = list(sample_subscore_dict[i].keys())[0]
        # assert cur_input_text == list(sample_subscore_dict[i].keys())[0]

        cur_subscore_item = sample_subscore_dict[i][cur_input_text]
        for keyword in sub_keywords:
            res[keyword].append(cur_subscore_item[keyword])



    # conduct random sampling calculation
    ass_ue_gt = json_to_list(ass_gt_file_name)
    rec_ue = ass_ue_gt[0]
    rec_metric = ass_ue_gt[1]
    print('below is the size of two values')
    print(rec_metric, res['overall'])
    assert rec_metric == res['overall']
    e_level, e_name, gen_name = ass_ue_gt[2], ass_ue_gt[3], ass_ue_gt[4]

    metric_res = {}

    for keyword in sub_keywords:
        rec_metric = res[keyword]

        rec_metric = np.array(rec_metric)
        oracle_score = ue_metric(-rec_metric, rec_metric)
        random_score = get_random_scores(ue_metric,
                                         rec_metric)  # random is the mean of 1000-times randomly assigning scores
        ue_metric_val = ue_metric(rec_ue, rec_metric)

        metric_res[
            e_level, e_name, gen_name+'_'+keyword, str(ue_metric)
        ] = ue_metric_val
        metric_res[
            e_level, e_name, gen_name+'_'+keyword, str(ue_metric) + "_normalized"
        ] = normalize_metric(ue_metric_val, oracle_score, random_score)  # here is the formula in

    print(metric_res)
    # save dict file


    return metric_res


from sklearn.neighbors import NearestNeighbors


def find_nearest_points(list_a, list_b):
    """
    Find the nearest point in list_b for each point in list_a.

    Args:
        list_a (list): A list of float values.
        list_b (list): A list of float values.

    Returns:
        list: A list of tuples, where each tuple contains a point from list_a
              and its nearest point from list_b.
    """
    # Convert lists to numpy arrays
    a = np.array(list_a).reshape(-1, 1)
    b = np.array(list_b).reshape(-1, 1)

    # Initialize NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(b)

    # Find nearest neighbors
    distances, indices = neigh.kneighbors(a)

    # Create a list of tuples with the matching points
    matches = [(a[i][0], b[indices[i]][0]) for i in range(len(a))]

    return matches, indices

def obtain_subscores_unieval(sample_overall_dict_file, sample_subscore_dict_file, ass_gt_file_name, ue_metric, inputs_no_nans=None, dataset_name=None):
    # read two dict file
    sample_overall_dict = json_file_to_list(sample_overall_dict_file)
    sample_subscore_dict = json_file_to_list(sample_subscore_dict_file)

    flag = 0
    if len(sample_overall_dict) == 1:
        sample_overall_dict = sample_overall_dict[0]
        assert len(sample_subscore_dict) == 1
        sample_subscore_dict = sample_subscore_dict[0]
        flag = 1



    sub_keywords = ['overall', 'coherence', 'consistency', 'fluency', 'relevance']
    ini_res = defaultdict(list)
    for i in range(len(sample_overall_dict)):
        cur_sample_overall = sample_overall_dict[i]
        cur_input_text = cur_sample_overall['input_texts']
        # assert cur_input_text == list(sample_subscore_dict[i].keys())[0]
        cur_subscore_item = sample_subscore_dict[i][cur_input_text]
        for keyword in sub_keywords:
            ini_res[keyword].append(cur_subscore_item[keyword])

    # print(res)

    # added for the recombine
    for keyword in sub_keywords:
        ini_res[keyword] = np.array(ini_res[keyword])

    recombined_inputs = defaultdict(list)
    for i, input_text in enumerate(inputs_no_nans):
        # (3778,5667) g3 has overlapping and is concatenate scenario.
        if (i >=3778 and i<5667 and dataset_name=='xsum') or dataset_name=='aeslc':
            recombined_inputs[input_text].append(i)
        else:
            recombined_inputs[input_text+f"{i}"].append(i)

    res = defaultdict(list)
    for input_text, ids in recombined_inputs.items():
        if len(ids) >= 2:
            print("unieval", ids)
        for keyword in sub_keywords:
            res[keyword].append(ini_res[keyword][ids].max())
    # ended for the recombine


    # conduct random sampling calculation
    ass_ue_gt_all = json_file_to_list(ass_gt_file_name)
    if flag == 1:
        assert len(ass_ue_gt_all) == 1
        ass_ue_gt_all = ass_ue_gt_all[0]
    metric_res = {}
    # ass_ue_gt_all = ass_ue_gt_all[0]
    for k in range(len(ass_ue_gt_all)):
        ass_ue_gt = ass_ue_gt_all[k]
        rec_ue = ass_ue_gt[0]
        rec_metric = ass_ue_gt[1]
        print(f"len(rec_ue) is {len(rec_ue)}")
        print(f"len(rec_metric) is {len(rec_metric)}")
        print(f"len(res['overall']) is {len(res['overall'])}")
        # assert rec_metric == res['overall']
        assert len(rec_metric) == len(res['overall']) # the length could be different due to nan values in the manager.py calculation

        e_level, e_name, gen_name = ass_ue_gt[2], ass_ue_gt[3], ass_ue_gt[4]



        for keyword in sub_keywords:
            rec_metric = res[keyword]
            print(f"len(res[keyword]) in {keyword} is {len(res[keyword])}")
            rec_metric = np.array(rec_metric)
            oracle_score = ue_metric(-rec_metric, rec_metric)
            random_score = get_random_scores(ue_metric,
                                             rec_metric)  # random is the mean of 1000-times randomly assigning scores
            ue_metric_val = ue_metric(rec_ue, rec_metric)

            metric_res[
                e_level, e_name, gen_name+'_'+keyword, str(ue_metric)
            ] = ue_metric_val
            metric_res[
                e_level, e_name, gen_name+'_'+keyword, str(ue_metric) + "_normalized"
            ] = normalize_metric(ue_metric_val, oracle_score, random_score)  # here is the formula in

    print(metric_res)
    # save dict file


    return metric_res

def obtain_subscores_unieval_cp(sample_overall_dict_file, sample_subscore_dict_file, ass_gt_file_name, ue_metric):
    # read two dict file
    sample_overall_dict = json_file_to_list(sample_overall_dict_file)
    sample_subscore_dict = json_file_to_list(sample_subscore_dict_file)

    sub_keywords = ['overall', 'coherence', 'consistency', 'fluency', 'relevance']
    res = defaultdict(list)
    for i in range(len(sample_overall_dict)):
        cur_sample_overall = sample_overall_dict[i]
        cur_input_text = cur_sample_overall['input_texts']
        # assert cur_input_text == list(sample_subscore_dict[i].keys())[0]
        cur_subscore_item = sample_subscore_dict[i][cur_input_text]
        for keyword in sub_keywords:
            res[keyword].append(cur_subscore_item[keyword])

    # print(res)


    # conduct random sampling calculation
    ass_ue_gt = json_to_list(ass_gt_file_name)
    rec_ue = ass_ue_gt[0]
    rec_metric = ass_ue_gt[1]
    print(f"len(rec_ue) is {len(rec_ue)}")
    print(f"len(rec_metric) is {len(res['overall'])}")
    if rec_metric != res['overall']:
        print("warning, nan happends")
        matches, indice = find_nearest_points(rec_metric, res['overall'])
        for i in range(len(indice-1)):
            assert indice[i] < indice[i+1]
        assert 1==0
    e_level, e_name, gen_name = ass_ue_gt[2], ass_ue_gt[3], ass_ue_gt[4]

    metric_res = {}

    for keyword in sub_keywords:
        rec_metric = res[keyword]

        rec_metric = np.array(rec_metric)
        oracle_score = ue_metric(-rec_metric, rec_metric)
        random_score = get_random_scores(ue_metric,
                                         rec_metric)  # random is the mean of 1000-times randomly assigning scores
        ue_metric_val = ue_metric(rec_ue, rec_metric)

        metric_res[
            e_level, e_name, gen_name+'_'+keyword, str(ue_metric)
        ] = ue_metric_val
        metric_res[
            e_level, e_name, gen_name+'_'+keyword, str(ue_metric) + "_normalized"
        ] = normalize_metric(ue_metric_val, oracle_score, random_score)  # here is the formula in

    print(metric_res)
    # save dict file


    return metric_res

def json_to_list_multi_row(file_name, select_ids_file):
    res = []
    select_ids = None
    if select_ids_file is not None:
        select_ids = json_to_list(select_ids_file)
    with open(file_name, 'r') as ini_f:
        ini_str = ini_f.readlines()
        jsq = 0
        for ele in ini_str:
            if select_ids is None or jsq in select_ids:
                mid_res = json.loads(ele) # res['pred'], res['label']
                res.append(mid_res)
            jsq += 1
    return res


def merge_json(merge_file_name_list, target_file_name, select_ids_file_list=None):
    merge_res = []
    for i in range(len(merge_file_name_list)):
        merge_file_name = merge_file_name_list[i]
        if select_ids_file_list==None:
            select_ids_file = None
        else:
            select_ids_file=select_ids_file_list[i]
        mid_res = json_to_list_multi_row(merge_file_name, select_ids_file)
        print(len(mid_res))
        merge_res.extend(mid_res)
    with open(target_file_name, 'w') as f:
        json_str = json.dumps(merge_res)
        f.write(json_str)
    print(f'save {len(merge_res)}-row merged json file: {target_file_name}')

def merge_list(merge_file_name_list, target_file_name):
    merge_res = []
    for merge_file_name in merge_file_name_list:
        mid_res = json_to_list_multi_row(merge_file_name, select_ids_file=None)
        print(len(mid_res))
        if len(merge_res) == 0:
            merge_res = copy.deepcopy(mid_res)
        else:
            for i in range(len(mid_res)):
                # sub_res = mid_res[i]
                merge_res[i][0].extend(mid_res[i][0])
                merge_res[i][1].extend(mid_res[i][1])
                assert merge_res[i][2] == mid_res[i][2]
                assert merge_res[i][3] == mid_res[i][3]
                assert merge_res[i][4] == mid_res[i][4]

    with open(target_file_name, 'w') as f:
        json_str = json.dumps(merge_res)
        f.write(json_str)
    print(f'save {len(merge_res)}-row merged json file: {target_file_name}')

def merge_instructed_json(
        dataset_name,
        abs_path,
        ue_cal_name_key,
        ue_cal_name_end,
        total_folder_path,
        end_gen,
        gen_seed,
        llm,
        situ,
        select_ids_file_list=None, # the kept ids after remove nan

                          ):
    print(situ)
    ue_cal_name_list = [f"{ue_cal_name_key}{cur_ue_cal_name_end}" for cur_ue_cal_name_end in ue_cal_name_end]

    ue_cal_name_total = f"{ue_cal_name_key}total"
    if situ=='est':
        ini_estimations_file_name_list = [f"sample_res/{dataset_name}_{llm}_{gen_seed}/est_{gen_seed}-{ue_cal_name}-{end_gen}.json" for ue_cal_name in ue_cal_name_list]
    elif situ == 'sample':
        ini_estimations_file_name_list = [
            f"sample_res/{dataset_name}_{llm}_{gen_seed}/sample_{gen_seed}-{ue_cal_name}-{end_gen}.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "ass_gt":
        ini_estimations_file_name_list = [
            f'assist_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-unieval_gt.json' for ue_cal_name in ue_cal_name_list]
    elif situ == "ass_res":
        ini_estimations_file_name_list = [
            f'assist_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-unieval_res.json' for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_gt":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_gt.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_res":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_res.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_gt_def_35":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_gt_def_35.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_res_def_35":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_res_def_35.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_gt_def_0_35":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_gt_def_0_35.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_res_def_0_35":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_res_def_0_35.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_gt_def_both_35":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_gt_def_both_35.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_res_def_both_35":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_res_def_both_35.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_gt_def_40":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_gt_def_40.json" for ue_cal_name in ue_cal_name_list]
    elif situ == "gpt_res_def_40":
        ini_estimations_file_name_list = [
            f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name}-chatgpt_res_def_40.json" for ue_cal_name in ue_cal_name_list]
    else:
        raise ValueError(f"situ={situ} is wrong")

    ini_estimations_file_list = [os.path.join(total_folder_path, ini_estimations_file_name) for
                                 ini_estimations_file_name in ini_estimations_file_name_list]
    if situ == 'est':
        estimations_file_name = f"total_result/sample_res/{dataset_name}_{llm}_{gen_seed}/est_{gen_seed}-{ue_cal_name_total}-{end_gen}.json"
    elif situ == 'sample':
        estimations_file_name = f"total_result/sample_res/{dataset_name}_{llm}_{gen_seed}/sample_{gen_seed}-{ue_cal_name_total}-{end_gen}.json"
    elif situ == "ass_gt":
        estimations_file_name = f'total_result/assist_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-unieval_gt.json'
    elif situ == "ass_res":
        estimations_file_name = f'total_result/assist_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-unieval_res.json'
    elif situ == "gpt_gt":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt.json"
    elif situ == "gpt_res":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res.json"

    ### used for A100
    elif situ == "gpt_gt_def_35":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_35.json"
    elif situ == "gpt_res_def_35":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_35.json"
    elif situ == "gpt_gt_def_0_35":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_0_35.json"
    elif situ == "gpt_res_def_0_35":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_0_35.json"
    elif situ == "gpt_gt_def_both_35":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_both_35.json"
    elif situ == "gpt_res_def_both_35":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_both_35.json"
    elif situ == "gpt_gt_def_40":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_40.json"
    elif situ == "gpt_res_def_40":
        estimations_file_name = f"total_result/chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_40.json"

    ### used for 3090
    # elif situ == "gpt_gt_def_35":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_35.json"
    # elif situ == "gpt_res_def_35":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_35.json"
    # elif situ == "gpt_gt_def_0_35":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_0_35.json"
    # elif situ == "gpt_res_def_0_35":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_0_35.json"
    # elif situ == "gpt_gt_def_both_35":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_both_35.json"
    # elif situ == "gpt_res_def_both_35":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_both_35.json"
    # elif situ == "gpt_gt_def_40":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_gt_def_40.json"
    # elif situ == "gpt_res_def_40":
    #     estimations_file_name = f"chatgpt_res/{dataset_name}_{llm}_{gen_seed}/{ue_cal_name_total}-chatgpt_res_def_40.json"

    else:
        raise ValueError(f"situ={situ} is wrong")

    # merge_json(merge_file_name_list=ini_estimations_file_list, target_file_name=estimations_file_name)
    if situ not in ["gpt_gt", "ass_gt"]:
        merge_json(merge_file_name_list=ini_estimations_file_list, target_file_name=estimations_file_name, select_ids_file_list=select_ids_file_list)
    else:
        merge_list(merge_file_name_list=ini_estimations_file_list, target_file_name=estimations_file_name)
    estimations_file = os.path.join(abs_path, estimations_file_name)

    return estimations_file


def get_input_text(estimations_file, use_select_ids=False):
    estimations_ini = json_file_to_list(estimations_file)
    try:
        est_keys = list(estimations_ini[0].keys())
    except:
        estimations_ini = estimations_ini[0]
        est_keys = list(estimations_ini[0].keys())

    assert len(estimations_ini) != 0
    estimations = defaultdict(list)

    for ele in estimations_ini:
        for k in est_keys:
            estimations[k].append(ele[k])

    if not use_select_ids:
        selected_ids = list(range(len(estimations["input_texts"])))

    inputs_no_nans = np.array(estimations["input_texts"])[selected_ids]

    return  inputs_no_nans

def obtain_w_b_gen_list(metric_cat, my_logit):
    if metric_cat == 'Relevance':
           white_gen_metrics_key_list = [
                   "Rouge_rougeL",
                   "correlation_spearmanr",
                   "correlation_kendalltau",

                   "UniEval_summarization_overall_relevance",
                   f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",
                   f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",
                   f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",
                   f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",

           ]

           white_gen_metrics_key_list_bart = [
                   "Rouge_rougeL",
                   "correlation_spearmanr",
                   "correlation_kendalltau",

                   "UniEval_summarization_overall_relevance",
                   "chatGPT_gpt-3.5-turbo_True_2_text summarization_overall_relevance",

           ]

           black_gen_metrics_key_list = [
                   "Rouge_rougeL",
                   "correlation_spearmanr",
                   "correlation_kendalltau",

                   "UniEval_summarization_overall_relevance",
                   f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",
                   f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",
                   f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",
                   f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_relevance",

           ]
    elif metric_cat == 'Consistency':
            white_gen_metrics_key_list = [
                    "BARTScoreSeq-rh",
                    "summac_cardname_microsoft/deberta-base-mnli",
                    "ctc_fact_consistency",

                    "UniEval_summarization_overall_consistency",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",


            ]

            white_gen_metrics_key_list_bart = [
                    "BARTScoreSeq-rh",
                    "summac_cardname_microsoft/deberta-base-mnli",
                    "ctc_fact_consistency",

                    "UniEval_summarization_overall_consistency",
                    "chatGPT_gpt-3.5-turbo_True_2_text summarization_overall_consistency",


            ]

            black_gen_metrics_key_list = [
                    # "BARTScoreSeq-rh",
                    "summac_cardname_microsoft/deberta-base-mnli",
                    "ctc_fact_consistency",

                    "UniEval_summarization_overall_consistency",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_consistency",

            ]


    elif metric_cat == 'Coherence':
            white_gen_metrics_key_list = [
                    "UniEval_summarization_overall_coherence",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",

            ]

            white_gen_metrics_key_list_bart = [
                    "UniEval_summarization_overall_coherence",
                    "chatGPT_gpt-3.5-turbo_True_2_text summarization_overall_coherence",

            ]

            black_gen_metrics_key_list = [
                    "UniEval_summarization_overall_coherence",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_coherence",

            ]
    elif metric_cat == 'Fluency':
            white_gen_metrics_key_list = [
                    "UniEval_summarization_overall_fluency",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
            ]

            white_gen_metrics_key_list_bart = [
                    "UniEval_summarization_overall_fluency",
                    "chatGPT_gpt-3.5-turbo_True_2_text summarization_overall_fluency",
            ]

            black_gen_metrics_key_list = [
                    "UniEval_summarization_overall_fluency",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_fluency",
            ]
    elif metric_cat == 'Overall':
            white_gen_metrics_key_list = [
                    "UniEval_summarization_overall_overall",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
            ]

            white_gen_metrics_key_list_bart = [
                    "UniEval_summarization_overall_overall",
                    "chatGPT_gpt-3.5-turbo_True_2_text summarization_overall_overall",
            ]

            black_gen_metrics_key_list = [
                    "UniEval_summarization_overall_overall",
                    f"chatGPT_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
                    f"chatGPT_def_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
                    f"chatGPT_def_0_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
                    f"chatGPT_def_both_gpt-3.5-turbo_True_{my_logit}_text summarization_overall_overall",
            ]
    else:
            raise ValueError(f"metric_cat = {metric_cat} is wrong!")

    return white_gen_metrics_key_list, white_gen_metrics_key_list_bart, black_gen_metrics_key_list



def transfer_tofu_dict(csv_file):
    pick_csv = pd.read_csv(csv_file)
    res = defaultdict(dict)
    row_n, col_n = pick_csv.shape
    for i in range(row_n):
        mid_res = pick_csv.iloc[i]
        # cur_key = mid_res[]

    pass





def save_prr_results(results, general_save_file_name):
    tranfer_general_results = transfer_general_res_dict(results)
    print(tranfer_general_results)
    if os.path.exists(general_save_file_name):
        os.remove(general_save_file_name)
    save_list_dict_to_json(d_dict=tranfer_general_results, file_name=general_save_file_name)