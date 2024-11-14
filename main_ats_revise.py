from datasets import load_dataset


from src.lm_polygraph.estimators import *
from src.lm_polygraph.utils2.model import WhiteboxModel, BlackboxModel
from src.lm_polygraph.utils2.dataset import Dataset
from src.lm_polygraph.utils2.processor import Logger


from src.lm_polygraph.utils2.manager import UEManager
from src.lm_polygraph.utils2.manager import estimate_uncertainty

from src.lm_polygraph.ue_metrics import ReversedPairsProportion, PredictionRejectionArea, RiskCoverageCurveAUC

import time
import MyFunc
import os
import argparse
import src.lm_polygraph.utils2 as mytils

import nltk
nltk.download('punkt')


parser = argparse.ArgumentParser(description='UE Benchmark on NLG Metric')
# framework setting
parser.add_argument('-model_type', type=str, default="w", help='[w(hite), b(lack)]')
parser.add_argument('-api_token', type=str, default=None, help='the api key for the black model')
parser.add_argument('-model_name_or_path', type=str, default="facebook/bart-large-cnn", help='the model used to generate response')
parser.add_argument('-device', type=str, default="cuda:0", help='the device')
parser.add_argument('-dataset_name', type=str, default="xsum", help='name of used dataset')
parser.add_argument('-batch_size',  type=int, default=4,   help='')
parser.add_argument('-split_seed', type=int, default=42, help='seed used to obtain the train/eval/test splits')
parser.add_argument('-seed', type=int, default=42, help='seed used to obtain the generation sampling')
parser.add_argument('-use_small', type=int, default=1, help='whether use small sampling for debug')
parser.add_argument('-use_part', type=str, default=None, help='whether use small sampling for debug')
# xsum
# (0,1889) g1
# (1889,3778) g2
# (3778,5667) g3
# (5667,7556) g4
# (7556,9445) g5
# (9445,11334) g6
parser.add_argument('-use_b_methods', type=str, default="[EigValLaplacian(verbose=True), LexicalSimilarity(metric=\"rougeL\"), Eccentricity(), NumSemSets()]", help='whether use small sampling for debug')

### original
parser.add_argument('-use_w_methods', type=str, default="[MaximumSequenceProbability(), MeanTokenEntropy(), MonteCarloSequenceEntropy(), MahalanobisDistanceSeq(\"decoder\"), RDESeq(\"decoder\"), PTrue()]", help='whether use small sampling for debug')
parser.add_argument('-use_e_methods', type=str, default="[EPTtu(),EPTrmi(),PETtu(),PETrmi(),EPStu(),EPSrmi(),PEStu(),PESrmi(),EPSrmiabs(),PESrmiabs()]", help='whether ensembling case')
# parser.add_argument('-use_e_methods', type=str, default=None, help='whether ensembling case')


### debug usage
# parser.add_argument('-use_w_methods', type=str, default="[PTrue(), PTrueSampling()]", help='whether use small sampling for debug')
# parser.add_argument('-use_e_methods', type=str, default="[EPTtu()]", help='whether ensembling case')

# parser.add_argument('-temperature', type=float, default=0.0, help='the temperature used in gpt generation for generated summaries')


# nlg_metrics setting
parser.add_argument('-use_rouge', type=int, default=1, help='whether use rouge')
parser.add_argument('-use_bart', type=int, default=1, help='whether use_bart')
parser.add_argument('-use_summac', type=int, default=0, help='whether use_summac')
parser.add_argument('-use_ctc', type=int, default=0, help='whether use_ctc')
parser.add_argument('-use_spearmanr', type=int, default=0, help='whether use_spearmanr')
parser.add_argument('-use_kendalltau', type=int, default=0, help='whether use kendalltau')

parser.add_argument('-use_chatgpt', type=int, default=0, help='whether use chatGPT for evaluation')
parser.add_argument('-use_chatgpt_35_def', type=int, default=0, help='whether use chatGPT for evaluation')
parser.add_argument('-use_chatgpt_35_def_0', type=int, default=0, help='whether use chatGPT for evaluation')
parser.add_argument('-use_chatgpt_35_def_both', type=int, default=0, help='whether use chatGPT for evaluation')
parser.add_argument('-use_chatgpt_40_def', type=int, default=0, help='whether use chatGPT for evaluation')
parser.add_argument('-use_claude', type=int, default=0, help='whether use claude3 for evaluation')
parser.add_argument('-eval_temperature', type=float, default=0.0, help='the temperature used in gpt generation for uncertainty evaluation')


parser.add_argument('-use_unieval_overall', type=int, default=0, help='whether unieval_overall')

parser.add_argument('-ue_cal_name', type=str, default="LexSim", help='the name used to save the respective generation model')

parser.add_argument('-use_ensemble_ue', type=int, default=0, help='whether include the ensemble-based ue methods')




args = parser.parse_args()

T1 = time.time()

# Specify HyperParameters
model_name_or_path = args.model_name_or_path
device = args.device
dataset_name = args.dataset_name
batch_size = args.batch_size
split_seed = args.split_seed
seed = args.seed
use_small = args.use_small




nlg_metric_keyword = MyFunc.obtain_nlg_metric_list(args)



nlg_metric_keyword_name = '-'.join(nlg_metric_keyword)
# dset_name_str = args.dset_name_str
# model_name_str = args.model_name_str

if args.model_name_or_path == 'gpt-3.5-turbo':
    model_name_str = 'gpt35'
elif args.model_name_or_path == 'gpt-4-turbo-preview': #'gpt-4':
    model_name_str = 'gpt40_turbo'
elif args.model_name_or_path == 'facebook/bart-large-cnn':
    model_name_str = 'bart_large'
elif args.model_name_or_path == 'meta-llama/Llama-2-7b-chat-hf':
    model_name_str = 'llama2_7b_chat'
else:
    # this should be correct
    raise ValueError(f"args.model_name_or_path={args.model_name_or_path} is not set!")

save_folder = f'./sample_res/{args.dataset_name}_{model_name_str}_{seed}/'
assist_folder = f'./assist_res/{args.dataset_name}_{model_name_str}_{seed}/'
chatgpt_folder = f'./chatgpt_res/{args.dataset_name}_{model_name_str}_{seed}/'
# result_dict_folder = f'./sample_res/{dset_name_str}_{model_name_str}_{seed}_result_dict/'

if not os.path.exists(save_folder):
    os.mkdir(save_folder)
if not os.path.exists(assist_folder):
    os.mkdir(assist_folder)
if not os.path.exists(chatgpt_folder):
    os.mkdir(chatgpt_folder)
# if not os.path.exists(result_dict_folder):
#     os.mkdir(result_dict_folder)


assist_file_name = os.path.join(assist_folder, f'{args.ue_cal_name}-unieval_res.json')
ass_gt_file_name = os.path.join(assist_folder, f'{args.ue_cal_name}-unieval_gt.json')
chatgpt_file_name = os.path.join(chatgpt_folder, f'{args.ue_cal_name}-chatgpt_res.json')
gpt_gt_file_name = os.path.join(chatgpt_folder, f'{args.ue_cal_name}-chatgpt_gt.json')

args.chatgpt_file_name = chatgpt_file_name


use_method_end = f"use_rouge={args.use_rouge}_bart={args.use_bart}_summac={args.use_summac}_" \
                 f"ctc={args.use_ctc}_spear={args.use_spearmanr}_kendal={args.use_kendalltau}" \
                 f"_chatgpt={args.use_chatgpt}_def35={args.use_chatgpt_35_def}_35_def_0={args.use_chatgpt_35_def_0}_35_both={args.use_chatgpt_35_def_both}_def40={args.use_chatgpt_40_def}_unieval={args.use_unieval_overall}"


sample_save_file_name = os.path.join(save_folder, f'sample_{seed}-{args.ue_cal_name}-small_{use_small}-{use_method_end}.json')  # sample level generation metrics
general_save_file_name = os.path.join(save_folder, f'general_{seed}-{args.ue_cal_name}-small_{use_small}-{use_method_end}.json') # final results
est_save_file_name = os.path.join(save_folder, f'est_{seed}-{args.ue_cal_name}-small_{use_small}-{use_method_end}.json') # estimation metrics self.estimations



if use_small:
    if os.path.exists(sample_save_file_name):
        os.remove(sample_save_file_name)
    if os.path.exists(est_save_file_name):
        os.remove(est_save_file_name)
    if os.path.exists(general_save_file_name):
        os.remove(general_save_file_name)
    if os.path.exists(chatgpt_file_name):
        os.remove(chatgpt_file_name)
    if os.path.exists(chatgpt_file_name[:-5] + '_def_35.json'):
        os.remove(chatgpt_file_name[:-5] + '_def_35.json')
    if os.path.exists(chatgpt_file_name[:-5] + '_def_0_35.json'):
        os.remove(chatgpt_file_name[:-5] + '_def_0_35.json')
    if os.path.exists(chatgpt_file_name[:-5] + '_def_both_35.json'):
        os.remove(chatgpt_file_name[:-5] + '_def_both_35.json')
    if os.path.exists(chatgpt_file_name[:-5] + '_def_40.json'):
        os.remove(chatgpt_file_name[:-5] + '_def_40.json')

    if os.path.exists(gpt_gt_file_name):
        os.remove(gpt_gt_file_name)
    if os.path.exists(gpt_gt_file_name[:-5] + '_def_35.json'):
        os.remove(gpt_gt_file_name[:-5] + '_def_35.json')
    if os.path.exists(gpt_gt_file_name[:-5] + '_def_0_35.json'):
        os.remove(gpt_gt_file_name[:-5] + '_def_0_35.json')
    if os.path.exists(gpt_gt_file_name[:-5] + '_def_both_35.json'):
        os.remove(gpt_gt_file_name[:-5] + '_def_both_35.json')
    if os.path.exists(gpt_gt_file_name[:-5] + '_def_40.json'):
        os.remove(gpt_gt_file_name[:-5] + '_def_40.json')

    if os.path.exists(assist_file_name):
        os.remove(assist_file_name)
    if os.path.exists(ass_gt_file_name):
        os.remove(ass_gt_file_name)


if not use_small and os.path.exists(sample_save_file_name):
    raise ValueError(f'{sample_save_file_name} already exists!')
if not use_small and os.path.exists(general_save_file_name):
    raise ValueError(f'{general_save_file_name} already exists!')



if args.use_part == 'g1':
    args.api_token = "xx"
elif args.use_part == 'g2':
    args.api_token = "xx"
elif args.use_part == 'g3':
    args.api_token = "xx"
elif args.use_part == 'g4':
    args.api_token = "xx"
elif args.use_part == 'g5':
    args.api_token = "xx"
elif args.use_part == 'g6':
    args.api_token = "xx"



# Initialize Model
if args.model_type == 'w':
    model = WhiteboxModel.from_pretrained(
        model_name_or_path,
        device=device,
    )
elif args.model_type == 'b':
    model = BlackboxModel.from_openai(
        args.api_token,
        model_name_or_path,
        seed=seed
    )
else:
    raise ValueError(f'model_type={args.model_type} is invalid!')


if dataset_name == 'xsum':
    doc_col_name, summ_col_name = 'document', 'summary'
elif dataset_name == 'aeslc':
    doc_col_name, summ_col_name = 'email_body', 'subject_line'
else:
    raise ValueError(f'dataset_name={dataset_name} is wrongly set!')

# Train and Eval Datasets
dataset = Dataset.load(
    dataset_name,
    doc_col_name, summ_col_name,
    batch_size=batch_size,
    split="test",
    use_small=use_small,
)

train_dataset = Dataset.load(
    dataset_name,
    summ_col_name, summ_col_name,
    batch_size=batch_size,
    split="train",
    use_small=use_small,
)

if (not args.use_small) and (args.use_part is not None):
    # use_part_range = eval(args.use_part)
    if args.use_part == 'g1':
        use_part_range = (0,1889)
    elif args.use_part == 'g2':
        use_part_range = (1889,3778)
    elif args.use_part == 'g3':
        use_part_range = (3778,5667)
    elif args.use_part == 'g4':
        use_part_range = (5667,7556)
    elif args.use_part == 'g5':
        use_part_range = (7556,9445)
    elif args.use_part == 'g6':
        use_part_range = (9445,11334)
    elif args.use_part == 'debug_aslec':
        use_part_range = (90,100)


    dataset.select(list(range(use_part_range[0], use_part_range[1])))





if use_small:
    # 20
    dataset.subsample(4, seed=split_seed)
    train_dataset.subsample(8, seed=split_seed)
else:
    train_dataset.subsample(1000, seed=split_seed)

# dataset.subsample(16, seed=)
# train_dataset.subsample(16, seed=)

# Metric, UE Metric, and UE Methods
if args.use_ensemble_ue:
    ue_methods = eval(args.use_e_methods) + eval(args.use_w_methods)
    print(f"ue_methods is {ue_methods}")

elif args.model_type == 'w':

    ue_methods = eval(args.use_w_methods)

    print(f"ue_methods is {ue_methods}")
elif args.model_type == 'b':

    ue_methods = eval(args.use_b_methods)


ue_metrics = [RiskCoverageCurveAUC()]


metrics = MyFunc.get_nlg_metric(nlg_metric_keyword, assist_file_name, args)

loggers = [Logger()]


### used for ensemble
ensemble_model = None
if args.use_ensemble_ue:
    ensemble_model = mytils.model.create_ensemble(
        model_paths=[model_name_or_path],
        mc=True,
        seed=1,
        mc_seeds=[2],
        ensembling_mode="pe",
        device="cuda:0",
        dropout_rate=0.1,
    )
    # assert 1==0

### print(ensemble_model)


# Initialize UE Manager
man = UEManager(
    dataset,
    model,
    ue_methods,
    metrics,
    ue_metrics,
    loggers,
    train_data=train_dataset,
    sample_save_file_name=sample_save_file_name,
    est_save_file_name=est_save_file_name,
    ensemble_model=ensemble_model,
    ass_gt_file_name=ass_gt_file_name,
    gpt_gt_file_name=gpt_gt_file_name,
    # used for saving predictions
    # cal_save_embeddings=args.cal_save_embeddings,
    # result_dict_save_file_name=result_dict_save_file_name,
)

# Compute Results
results = man()




for key in results.keys():
    print(f"UE Method: {key[1]}, NLG Metric: {key[2]}, UE Metric: {key[3]}, Final Score: {results[key]:.3f}")

tranfer_general_results = MyFunc.transfer_general_res_dict(results)
print(tranfer_general_results)
if os.path.exists(general_save_file_name):
    os.remove(general_save_file_name)
MyFunc.save_list_dict_to_json(d_dict=tranfer_general_results, file_name=general_save_file_name)

print(f'{os.path.join(os.getcwd(), est_save_file_name)}')
print(f'{os.path.join(os.getcwd(), sample_save_file_name)}')
print(f'{os.path.join(os.getcwd(), assist_file_name)}')
print(f'{os.path.join(os.getcwd(), gpt_gt_file_name)}')
print(f'{os.path.join(os.getcwd(), general_save_file_name)}')



T2 = time.time()

print((T2-T1) * 1000) # ms after * 1000

