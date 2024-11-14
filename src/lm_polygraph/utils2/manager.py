import copy

import numpy as np
import torch
import sys
import gc
import json

from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.lm_polygraph.utils2.dataset import Dataset
from src.lm_polygraph.utils2.model import WhiteboxModel, BlackboxModel, Model
from src.lm_polygraph.utils2.processor import Processor
from src.lm_polygraph.utils2.normalize import normalize_ue, can_normalize_ue
from src.lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from src.lm_polygraph.ue_metrics.ue_metric import (
    UEMetric,
    get_random_scores,
    normalize_metric,
)
from src.lm_polygraph.estimators.estimator import Estimator
# from src.lm_polygraph.estimators.common import DEBERTA
from src.lm_polygraph.stat_calculators.stat_calculator import (
    StatCalculator,
    STAT_CALCULATORS,
    STAT_DEPENDENCIES,
)

from .utils_myfunc import transfer_dict_list_format, save_list_dict_to_json

from src.lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator


from lm_polygraph.estimators.common import DEBERTA



def _order_calculators(stats: List[str]) -> Tuple[List[str], Set[str]]:
    ordered: List[str] = []
    have_stats: Set[str] = set()
    while len(stats) > 0:
        stat = stats[0]
        if stat in have_stats:
            stats = stats[1:]
            continue
        dependent = False
        if stat not in STAT_DEPENDENCIES.keys():
            raise Exception(
                f"Cant find stat calculator for: {stat}. Maybe you forgot to register it by calling register()?"
            )
        for d in STAT_DEPENDENCIES[stat]:
            if d not in have_stats:
                stats = [d] + stats
                if stats.count(d) > 40:
                    raise Exception(f"Found possibly cyclic dependencies: {d}")
                dependent = True
        if not dependent:
            stats = stats[1:]
            ordered.append(stat)
            for new_stat in STAT_CALCULATORS[stat].stats:
                have_stats.add(new_stat)
    return ordered, have_stats


def _check_unique_names(xs):
    names = set()
    for x in xs:
        if str(x) in names:
            raise Exception(f"Got multiple __str__ values for {x}")
        names.add(str(x))


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
    for i, input_text in enumerate(inputs):
        recombined_inputs[input_text].append(i)

    recombined_ue, recombined_gen_metric = [], []
    for input_text, ids in recombined_inputs.items():
        recombined_ue.append(ue[ids].mean())
        # Assumes that metric is bigger for better generations! # comment: need to pay attention to
        recombined_gen_metric.append(gen_metric[ids].max())

    return recombined_ue, recombined_gen_metric


@dataclass
class UncertaintyOutput:
    """
    Uncertainty estimator output.

    Parameters:
        uncertainty (float): uncertainty estimation.
        input_text (str): text used as model input.
        generation_text (str): text generated by the model.
        model_path (str): path to the model used in generation.
    """

    uncertainty: float
    input_text: str
    generation_text: str
    model_path: str


def estimate_uncertainty(
    model: Model, estimator: Estimator, input_text: str
) -> UncertaintyOutput:
    """
    Estimated uncertainty of the model generation using the provided esitmator.

    Parameters:
        model (Model): model to estimate uncertainty of. Either lm_polygraph.WhiteboxModel or
            lm_polygraph.BlackboxModel model can be used.
        estimator (Estimator): uncertainty estimation method to use. Can be any of the methods at
            lm_polygraph.estimators.
        input_text (str): text to estimate uncertainty of.
    Returns:
        UncertaintyOutput: uncertainty estimation float along with supporting info.

    Examples:

    ```python
    >>> from lm_polygraph import WhiteboxModel
    >>> from lm_polygraph.estimators import LexicalSimilarity
    >>> model = WhiteboxModel.from_pretrained(
    ...     'bigscience/bloomz-560m',
    ...     device='cpu',
    ... )
    >>> estimator = LexicalSimilarity('rougeL')
    >>> estimate_uncertainty(model, estimator, input_text='Who is George Bush?')
    UncertaintyOutput(uncertainty=-0.9176470588235295, input_text='Who is George Bush?', generation_text=' President of the United States', model_path='bigscience/bloomz-560m')
    ```

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> from lm_polygraph.estimators import EigValLaplacian
    >>> model = BlackboxModel.from_openai(
    ...     'YOUR_OPENAI_TOKEN',
    ...     'gpt-3.5-turbo'
    ... )
    >>> estimator = EigValLaplacian()
    >>> estimate_uncertainty(model, estimator, input_text='When did Albert Einstein die?')
    UncertaintyOutput(uncertainty=1.0022274826855433, input_text='When did Albert Einstein die?', generation_text='Albert Einstein died on April 18, 1955.', model_path='gpt-3.5-turbo')
    ```
    """
    man = UEManager(
        Dataset([input_text], [""], batch_size=1),
        model,
        [estimator],
        [],
        [],
        [],
        ignore_exceptions=False,
        verbose=False,
    )
    man()
    ue = man.estimations[estimator.level, str(estimator)]
    #if can_normalize_ue(estimator, model.model_path):
    #    if estimator.level == "sequence":
    #        ue = normalize_ue(estimator, model.model_path, ue[0])
    #    else:
    #        ue = [normalize_ue(estimator, model.model_path, i) for i in ue]
    texts = man.stats.get("greedy_texts", man.stats.get("blackbox_greedy_texts", None))
    return UncertaintyOutput(ue[0], input_text, texts[0], model.model_path)


class UEManager:
    """
    Manager to conduct uncertainty estimation experiments by using several uncertainty methods, ground-truth
    uncertainty values and correlation metrics at once. Used for running benchmarks.

    Examples:

    ```python
    >>> from lm_polygraph import WhiteboxModel
    >>> from lm_polygraph.utils.dataset import Dataset
    >>> from lm_polygraph.estimators import *
    >>> from lm_polygraph.ue_metrics import *
    >>> from lm_polygraph.generation_metrics import *
    >>> model = WhiteboxModel.from_pretrained(
    ...     'bigscience/bloomz-560m',
    ...     device='cuda:0',
    ... )
    >>> dataset = Dataset.load(
    ...     '../workdir/data/triviaqa.csv',
    ...     'question', 'answer',
    ...     batch_size=4,
    ... )
    >>> ue_methods = [MaximumSequenceProbability(), SemanticEntropy()]
    >>> ue_metrics = [RiskCoverageCurveAUC()]
    >>> ground_truth = [RougeMetric('rougeL'), BartScoreSeqMetric('rh')]
    >>> man = UEManager(dataset, model, ue_methods, ground_truth, ue_metrics, processors=[])
    >>> results = man()
    >>> results.save("./manager.man")
    ```
    """

    def __init__(
        self,
        data: Dataset,
        model: Model,
        estimators: List[Estimator],
        generation_metrics: List[GenerationMetric],
        ue_metrics: List[UEMetric],
        processors: List[Processor],
        train_data: Dataset = None,
        background_train_data: Dataset = None,
        ignore_exceptions: bool = True,
        ensemble_model: Optional[WhiteboxModel] = None,
        deberta_batch_size: int = 10,
        verbose: bool = True,
        max_new_tokens: int = 100,
        background_train_dataset_max_new_tokens: int = 100,
        sample_save_file_name=None,
        est_save_file_name=None,
        # cal_save_embeddings=0,
        # result_dict_save_file_name=None,
        ass_gt_file_name=None,
        gpt_gt_file_name=None,
    ):
        """
        Parameters:
            data (Dataset): Dataset to run benchmark on.
            model (Model): Model to run benchmark on. Can be either lm_polygraph.WhiteboxModel or
                lm_polygraph.BlackboxModel
            estimators (List[Estimator]): List of estimators to evaluate at benchmark.
            generation_metrics (List[GenerationMetrics]): List of methods to use to calculate ground-truth uncertainty.
            ue_metrics (List[UEMetric]): List of methods to measure correlation between ground-truth uncertainties from
                `generation_metrics` and uncertainty estimators in `estimators`.
            processors (List[Processor]): List of processors to apply after each batch.
            train_data (Optional[Dataset]): Dataset to train density-based estimators on. Can be set to None, if
                no density-based method is used. Default: None.
            ignore_exceptions (bool): If true, exceptions on a new batch will be printed to stderr and
                the batch will be skipped. Useful to skip CUDA OOM errors on large datasets. Default: True.
            deberta_batch_size (int): Batch size for DeBERTa model used in some estimators. Default: 10.
            verbose (bool): If set, will print useful info during batch processing. Default: True.
            max_new_tokens (int): Maximum new tokens to use in generation. Default: 100.

            sample_save_file_name (str): the path and file name to save the sample-level input text, target text, gready text (prediction) and each metric scores
        """
        self.model: WhiteboxModel = model
        self.train_data: Dataset = train_data
        self.background_train_data: Dataset = background_train_data
        self.ensemble_model = ensemble_model
        self.deberta_batch_size = deberta_batch_size
        self.data: Dataset = data
        self.estimators: List[Estimator] = estimators
        self.generation_metrics: List[GenerationMetric] = generation_metrics
        self.ue_metrics: List[UEMetric] = ue_metrics

        self.sample_save_file_name = sample_save_file_name
        self.est_save_file_name = est_save_file_name

        _check_unique_names(generation_metrics)
        _check_unique_names(estimators)
        _check_unique_names(ue_metrics)

        if isinstance(model, BlackboxModel):
            greedy = ["blackbox_greedy_texts"]
        else:
            greedy = ["greedy_tokens", "greedy_texts"]

        stats = (
            [s for e in self.estimators for s in e.stats_dependencies]
            + [s for m in generation_metrics for s in m.stats_dependencies]
            + greedy
        )
        stats, have_stats = _order_calculators(stats)
        ### original
        # stats = [
        #     s
        #     for s in stats
        #     if not (str(s).startswith("ensemble_"))
        #     or (
        #         (str(s).startswith("blackbox_") and s[len("blackbox_") :] in have_stats)
        #     )
        # ]
        ### revised for ensembling
        stats = [
            s
            for s in stats
            if not (str(s).startswith("ensemble_"))
            or (str(s).startswith("ensemble_")) # revised but some redundant
            or (
                (str(s).startswith("blackbox_") and s[len("blackbox_") :] in have_stats)
            )

        ]

        self.stat_calculators: List[StatCalculator] = [
            STAT_CALCULATORS[c] for c in stats
        ]
        if verbose:
            print("Stat calculators:", self.stat_calculators)

        self.ensemble_estimators = []
        single_estimators = []
        for e in estimators:
            for s in e.stats_dependencies:
                if s.startswith("ensemble"):
                    self.ensemble_estimators.append(e)
                    break
            if e not in self.ensemble_estimators:
                single_estimators.append(e)
        self.estimators = single_estimators

        train_stats = [
            s
            for e in self.estimators
            for s in e.stats_dependencies
            if s.startswith("train")
        ]
        train_stats += (
            ["greedy_tokens", "greedy_texts"]
            if "train_greedy_log_likelihoods" in train_stats
            else []
        )
        train_stats, _ = _order_calculators(train_stats)
        self.train_stat_calculators: List[StatCalculator] = [
            STAT_CALCULATORS[c] for c in train_stats
        ]
        background_train_stats = [
            s
            for e in self.estimators
            for s in e.stats_dependencies
            if s.startswith("background_train")
        ]
        background_train_stats, _ = _order_calculators(background_train_stats)
        self.background_train_stat_calculators: List[StatCalculator] = [
            STAT_CALCULATORS[c] for c in background_train_stats
        ]

        ensemble_stats = [
            s
            for e in self.ensemble_estimators
            for s in e.stats_dependencies
            if s.startswith("ensemble")
        ]
        ensemble_stats, _ = _order_calculators(ensemble_stats)
        self.ensemble_stat_calculators: List[StatCalculator] = [
            STAT_CALCULATORS[c] for c in ensemble_stats
        ]

        self.gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.metrics: Dict[Tuple[str, str, str, str], float] = {}
        self.stats: Dict[str, List] = defaultdict(list)

        self.processors = processors
        self.ignore_exceptions = ignore_exceptions
        self.verbose = verbose
        self.max_new_tokens = max_new_tokens
        self.background_train_dataset_max_new_tokens = (
            background_train_dataset_max_new_tokens
        )

        # # realted to save pretrained model prediction
        # self.cal_save_embeddings = cal_save_embeddings
        # self.result_dict_save_file_name = result_dict_save_file_name
        self.ass_gt_file_name = ass_gt_file_name
        self.gpt_gt_file_name = gpt_gt_file_name

    def __call__(self) -> Dict[Tuple[str, str, str, str], float]:
        """
        Runs benchmark and reports metrics results. Saves all useful calculated statistics for further usage.
        The run includes:
        * Calculating uncertainty estimations for each `estimator` for all input texts in the dataset
        * Calculating ground-truth uncertainties for each `generation_metrics` for all input texts in the dataset.
        * Calculating correlation measure for each `ue_metrics`, between each pair of
          (uncertainty estimation, ground-truth uncertainty) which comes from the same level
          (both 'sequence' or both 'token').
        * Saving uncertainty estimations, ground-truth uncertainties and ue_metrics values for further usage.

        Returns:
            Dict[Tuple[str, str, str, str], float]: dictionary with metrics results. Dictionary keys consist of
                - uncertainty estimation level: 'sequence' or 'token',
                - estimator name,
                - generation metrics name,
                - `ue_metrics` name which was used to calculate quality.
        """

        # load DEBERTA to correct device
        if hasattr(self.model, "device"):
            DEBERTA.to(self.model.device())
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            DEBERTA.to(device)

        train_stats = self._extract_train_embeddings()
        background_train_stats = self._extract_train_embeddings(background=True)

        iterable_data = tqdm(self.data) if self.verbose else self.data
        for inp_texts, target_texts in iterable_data:

            # used for llama
            if ("llama" in self.model.model_path) or ('gpt' in self.model.model_path):
                inp_texts = [f"Only give the summary of below text. \n {ele}" for ele in inp_texts]


            batch_stats: Dict[str, np.ndarray] = {}
            for key, val in [
                ("input_texts", inp_texts),
                ("target_texts", target_texts),
            ]:
                self.stats[key] += val
                batch_stats[key] = val

            if isinstance(self.model, WhiteboxModel):
                target_tokens = [
                    self.model.tokenizer([text])["input_ids"][0]
                    + [self.model.tokenizer.eos_token_id]
                    for text in target_texts
                ]
                self.stats["target_tokens"] += target_tokens
                batch_stats["target_tokens"] = target_tokens

            batch_stats["deberta_batch_size"] = self.deberta_batch_size
            train_stats_keys = list(train_stats.keys())
            for stat in train_stats_keys:
                batch_stats[stat] = train_stats.pop(stat) # extract the 'train_embeddings_encoder', 'train_embeddings_decoder' info batch_stats

            background_train_stats_keys = list(background_train_stats.keys())
            for stat in background_train_stats_keys:
                batch_stats[stat] = background_train_stats.pop(stat)

            # revised: ensemble_model usage
            if self.ensemble_model is not None:
                batch_stats["ensemble_model"] = self.ensemble_model
                if "ensemble_generation_params" not in batch_stats.keys():
                    batch_stats["ensemble_generation_params"] = {}

            # obtain the decoder output as well
            batch_stats = self.calculate(batch_stats, self.stat_calculators, inp_texts) # calcualte using each sub-methods and merge the attribute
            # batch_stats = self.new_calculate(batch_stats, self.stat_calculators, inp_texts, self.cal_save_embeddings, self.result_dict_save_file_name)  # calcualte using each sub-methods and merge the attribute
            # assert 1==0

            batch_estimations, bad_estimators = self.estimate(
                batch_stats, self.estimators
            )

            for bad_estimator in bad_estimators:
                key = (bad_estimator.level, str(bad_estimator))
                self.estimations.pop(key, None)
                self.estimators.remove(bad_estimator)

            batch_gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            save_stats= {}
            save_stats['input_texts'] = batch_stats['input_texts']
            save_stats['target_texts'] = batch_stats['target_texts']

            for generation_metric in self.generation_metrics:
                # # original
                # m = generation_metric(
                #     batch_stats, target_texts=target_texts, target_tokens=target_tokens
                # ).tolist()
                if isinstance(self.model, WhiteboxModel):
                    m = generation_metric(
                        batch_stats, target_texts=target_texts, target_tokens=target_tokens, white=True
                    ).tolist()
                else:
                    m = generation_metric(
                        batch_stats, target_texts=target_texts, target_tokens=None, white=False
                    ).tolist()

                self.gen_metrics[generation_metric.level, str(generation_metric)] += m
                batch_gen_metrics[generation_metric.level, str(generation_metric)] += m
                save_stats[generation_metric.level + '#' + str(generation_metric)] = m

            for key in ["blackbox_greedy_texts", "greedy_texts", "greedy_tokens"]:
                if key in batch_stats.keys():
                    self.stats[key] += batch_stats[key]
                    save_stats[key] = batch_stats[key]

            # preprare to save into the json file
            if self.sample_save_file_name:
                trans_save_stats = transfer_dict_list_format(save_stats)
                save_list_dict_to_json(trans_save_stats,file_name=self.sample_save_file_name)
            # assert 1==0


            for processor in self.processors:
                processor.on_batch(batch_stats, batch_gen_metrics, batch_estimations) # print status and results

        if self.ensemble_model is not None:
            # Now do the same for ensemble calculators
            device = self.model.model.device
            self.model.model.to("cpu")
            self.ensemble_model.model.to(device)

            iterable_data = tqdm(self.data) if self.verbose else self.data
            for inp_texts, target_texts in iterable_data:
                batch_stats: Dict[str, np.ndarray] = {}
                for key, val in [
                    ("input_texts", inp_texts),
                    ("target_texts", target_texts),
                ]:
                    batch_stats[key] = val

                target_tokens = [
                    self.model.tokenizer([text])["input_ids"][0]
                    + [self.model.tokenizer.eos_token_id]
                    for text in target_texts
                ]
                batch_stats["target_tokens"] = target_tokens

                batch_stats["ensemble_generation_params"] = {}
                batch_stats["ensemble_model"] = self.ensemble_model

                batch_stats = self.calculate(
                    batch_stats, self.ensemble_stat_calculators, inp_texts
                )

                batch_estimations, bad_estimators = self.estimate(
                    batch_stats, self.ensemble_estimators
                )

                for bad_estimator in bad_estimators:
                    key = (bad_estimator.level, str(bad_estimator))
                    self.estimations.pop(key, None)
                    self.ensemble_estimators.remove(bad_estimator)

            torch.cuda.empty_cache()
            gc.collect()

        for (e_level, e_name), estimator_values in self.estimations.items():
            for (gen_level, gen_name), generation_metric in self.gen_metrics.items():
                for ue_metric in self.ue_metrics:
                    if gen_level != e_level:
                        continue
                    print(f"estimator_values are {estimator_values}")
                    print(f"generation_metric are {generation_metric}")
                    # assert 1==0
                    if len(estimator_values) != len(generation_metric):
                        raise Exception(
                            f"Got different number of metrics for {e_name} and {gen_name}: "
                            f"{len(estimator_values)} and {len(generation_metric)}"
                        )
                    # TODO: Report how many nans!
                    # This is important to know for a user
                    ue, metric, selected_ids = _delete_nans( # selected_ids is the samples without nan cases
                        estimator_values, generation_metric
                    )
                    print(f"selected_ids is {selected_ids}")
                    if len(ue) == 0:
                        self.metrics[e_level, e_name, gen_name, str(ue_metric)] = np.nan
                    else:
                        inputs_no_nans = np.array(self.stats["input_texts"])[
                            selected_ids
                        ]
                        print(f'metric={metric} before the recombine_data')
                        rec_ue, rec_metric = _recombine_data(ue, metric, inputs_no_nans) # ue_metrics & gen_metrics

                        # debug usage
                        print(f"for calculation {[rec_ue, rec_metric, e_level, e_name, gen_name]}")

                        print(f"gen_name is {gen_name}")
                        # add by me: if unievaloverall: save the rec_metric
                        if gen_name == 'UniEval_summarization_overall':
                            with open(self.ass_gt_file_name, 'a') as gt_f:
                                json_str = json.dumps([rec_ue, rec_metric, e_level, e_name, gen_name])
                                gt_f.write(json_str)
                                gt_f.write('\n')
                            print(f'{self.ass_gt_file_name} has been saved!')
                            if len(selected_ids) > 0:
                                with open(self.ass_gt_file_name[:-5]+"_seid.json", 'a') as gt_f:
                                    json_str = json.dumps(selected_ids)
                                    gt_f.write(json_str)
                                    gt_f.write('\n')
                                print(f'{self.ass_gt_file_name[:-5]+"_seid.json"} has been saved!')
                        # print(f'generation_metric={generation_metric}')

                        # if "gpt" in gen_name and "text summarization_overall" in gen_name:  #  if gen_name == 'chatGPT_gpt-3.5-turbo_True_2_machine translation_overall':
                        if gen_name == "chatGPT_gpt-3.5-turbo_True_1_text summarization_overall":
                            print(gen_name)
                            with open(self.gpt_gt_file_name, 'a') as gt_f:
                                json_str = json.dumps([rec_ue, rec_metric, e_level, e_name, gen_name])
                                gt_f.write(json_str)
                                gt_f.write('\n')
                            print(f'{self.gpt_gt_file_name} has been saved!')
                            if len(selected_ids) > 0:
                                with open(self.gpt_gt_file_name[:-5]+"_seid.json", 'a') as gt_f:
                                    json_str = json.dumps(selected_ids)
                                    gt_f.write(json_str)
                                    gt_f.write('\n')
                                print(f'{self.gpt_gt_file_name[:-5]+"_seid.json"} has been saved!')

                        if gen_name == "chatGPT_def_gpt-3.5-turbo_True_1_text summarization_overall":
                            # print(gen_name)
                            cur_gpt_gt_file_name_35 = self.gpt_gt_file_name[:-5] + "_def_35.json"
                            with open(cur_gpt_gt_file_name_35, 'a') as gt_f:
                                json_str = json.dumps([rec_ue, rec_metric, e_level, e_name, gen_name])
                                gt_f.write(json_str)
                                gt_f.write('\n')
                            print(f'{cur_gpt_gt_file_name_35} has been saved!')

                        if gen_name == "chatGPT_def_0_gpt-3.5-turbo_True_1_text summarization_overall":
                            # print(gen_name)
                            cur_gpt_gt_file_name_35_0 = self.gpt_gt_file_name[:-5] + "_def_0_35.json"
                            with open(cur_gpt_gt_file_name_35_0, 'a') as gt_f:
                                json_str = json.dumps([rec_ue, rec_metric, e_level, e_name, gen_name])
                                gt_f.write(json_str)
                                gt_f.write('\n')
                            print(f'{cur_gpt_gt_file_name_35_0} has been saved!')

                        if gen_name == "chatGPT_def_both_gpt-3.5-turbo_True_1_text summarization_overall":
                            # print(gen_name)
                            cur_gpt_gt_file_name_35_both = self.gpt_gt_file_name[:-5] + "_def_both_35.json"
                            with open(cur_gpt_gt_file_name_35_both, 'a') as gt_f:
                                json_str = json.dumps([rec_ue, rec_metric, e_level, e_name, gen_name])
                                gt_f.write(json_str)
                                gt_f.write('\n')
                            print(f'{cur_gpt_gt_file_name_35_both} has been saved!')

                        if gen_name == "chatGPT_def_gpt-4_True_1_text summarization_overall":
                            # print(gen_name)
                            cur_gpt_gt_file_name_40 = self.gpt_gt_file_name[:-5] + "_def_40.json"
                            with open(cur_gpt_gt_file_name_40, 'a') as gt_f:
                                json_str = json.dumps([rec_ue, rec_metric, e_level, e_name, gen_name])
                                gt_f.write(json_str)
                                gt_f.write('\n')
                            print(f'{cur_gpt_gt_file_name_40} has been saved!')

                        print(f'generation_metric={generation_metric}')





                        rec_metric = np.array(rec_metric)
                        oracle_score = ue_metric(-rec_metric, rec_metric)
                        random_score = get_random_scores(ue_metric, rec_metric) # random is the mean of 1000-times randomly assigning scores
                        ue_metric_val = ue_metric(rec_ue, rec_metric)
                        self.metrics[
                            e_level, e_name, gen_name, str(ue_metric)
                        ] = ue_metric_val
                        self.metrics[
                            e_level, e_name, gen_name, str(ue_metric) + "_normalized"
                        ] = normalize_metric(ue_metric_val, oracle_score, random_score) # here is the formula in paper

        for processor in self.processors:
            processor.on_eval(self.metrics)

        return self.metrics

    def calculate(self, batch_stats: dict, calculators: list, inp_texts: list) -> dict:
        """
        Runs stat calculators and handles errors if any occur. Returns updated batch stats

        Parameters:
            batch_stats (dict): contains current batch statistics to be updated
            calculators (list): list of stat calculators to run
            inp_texts (list): list of inputs to the model in the batch
        """
        for stat_calculator in calculators:
            try:
                new_stats = stat_calculator(
                    batch_stats, inp_texts, self.model, self.max_new_tokens
                )
                for stat, stat_value in new_stats.items():
                    if stat in batch_stats.keys(): # might not be able to update
                        continue
                    batch_stats[stat] = stat_value
            except Exception as e:
                # if self.ignore_exceptions:
                if False:
                    log_msg = f"Caught exception while calculating stats: {e} in Stat Calculator {stat_calculator}"
                    sys.stderr.write(log_msg)
                    print(log_msg)
                    continue
                else:
                    raise e

        return batch_stats


    def new_calculate(self, batch_stats: dict, calculators: list, inp_texts: list, cal_save_embeddings, result_dict_save_file_name) -> dict:
        """
        Runs stat calculators and handles errors if any occur. Returns updated batch stats

        Parameters:
            batch_stats (dict): contains current batch statistics to be updated
            calculators (list): list of stat calculators to run
            inp_texts (list): list of inputs to the model in the batch
        """
        for stat_calculator in calculators:
            try:
                if isinstance(stat_calculator, GreedyProbsCalculator):
                    new_stats = stat_calculator(
                        batch_stats, inp_texts, self.model, self.max_new_tokens, cal_save_embeddings, result_dict_save_file_name
                    )
                else:
                    new_stats = stat_calculator(
                        batch_stats, inp_texts, self.model, self.max_new_tokens
                    )
                for stat, stat_value in new_stats.items():
                    if stat in batch_stats.keys(): # might not be able to update
                        continue
                    batch_stats[stat] = stat_value
            except Exception as e:
                if self.ignore_exceptions:
                    log_msg = f"Caught exception while calculating stats: {e} in Stat Calculator {stat_calculator}"
                    sys.stderr.write(log_msg)
                    print(log_msg)
                    continue
                else:
                    raise e

        return batch_stats

    def estimate(
        self, batch_stats: dict, estimators: list
    ) -> Dict[Tuple[str, str], List[float]]:
        """
        Runs stat calculators and handles errors if any occur. Returns updated batch stats

        Parameters:
            batch_stats (dict): contains current batch statistics to be updated
            estimators (list): list of estimators to run
        """
        batch_estimations = defaultdict(list)
        bad_estimators = []

        # added by me
        save_stats_est = {}
        save_stats_est['input_texts'] = batch_stats['input_texts']
        save_stats_est['target_texts'] = batch_stats['target_texts']

        for estimator in estimators:

            # e = estimator(batch_stats).tolist()  # e is a list with length of batch size
            # self.estimations[estimator.level, str(estimator)] += e
            # batch_estimations[estimator.level, str(estimator)] += e

            try:
                e = estimator(batch_stats).tolist()   # e is a list with length of batch size
                self.estimations[estimator.level, str(estimator)] += e
                batch_estimations[estimator.level, str(estimator)] += e

                # added by me
                save_stats_est[estimator.level + '#' + str(estimator)] = e

            except Exception as e:
                if self.ignore_exceptions:
                    bad_estimators.append(estimator)
                    log_msg = f"Caught exception while estimating uncertainty: {e} in estimator {estimator}. Estimator will be removed."
                    sys.stderr.write(log_msg)
                    print(log_msg)
                    continue
                else:
                    raise e




        if self.est_save_file_name:
            trans_save_stats_est = transfer_dict_list_format(save_stats_est)
            save_list_dict_to_json(trans_save_stats_est, file_name=self.est_save_file_name)

        return batch_estimations, bad_estimators

    def _extract_train_embeddings(
        self, background: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        train_stats = {}
        result_train_stat = {}
        if background:
            data = self.background_train_data
            stat_calculators = self.background_train_stat_calculators
            max_new_tokens = self.background_train_dataset_max_new_tokens
        else:
            data = self.train_data
            stat_calculators = self.train_stat_calculators
            max_new_tokens = self.max_new_tokens
        if len(stat_calculators) and (data is not None):
            for inp_texts, target_texts in tqdm(data):
                target_tokens = [
                    self.model.tokenizer([text])["input_ids"][0]
                    + [self.model.tokenizer.eos_token_id]
                    for text in target_texts
                ]

                batch_stats: Dict[str, np.ndarray] = {}
                for key, val in [
                    ("input_texts", inp_texts),
                    ("target_texts", target_texts),
                    ("target_tokens", target_tokens),
                ]:
                    batch_stats[key] = val

                for stat_calculator in stat_calculators:
                    new_stats = stat_calculator(
                        batch_stats, inp_texts, self.model, max_new_tokens
                    )
                    for stat, stat_value in new_stats.items():
                        if stat in batch_stats.keys(): # my comment: avoild overwriting, but may miss some update
                            continue
                        batch_stats[stat] = stat_value

                for stat in batch_stats.keys():
                    if stat in [
                        "input_tokens",
                        "input_texts",
                        "target_texts",
                        "target_tokens",
                    ]:
                        continue
                    if stat in train_stats.keys():
                        train_stats[stat].append(batch_stats[stat])
                    else:
                        train_stats[stat] = [batch_stats[stat]]

                torch.cuda.empty_cache()
                gc.collect()

            key_prefix = "background_train_" if background else "train_"
            for stat in train_stats.keys():
                if any(s is None for s in train_stats[stat]):
                    continue
                if isinstance(train_stats[stat][0], list):
                    result_train_stat[key_prefix + stat] = [
                        item for sublist in train_stats[stat] for item in sublist
                    ]
                else:
                    result_train_stat[key_prefix + stat] = np.concatenate(
                        train_stats[stat]
                    )

        return result_train_stat

    def save(self, save_path: str):
        """
        Saves the run results in the provided path. Will raise exception, if no results are calculated yet.
        To load the saved manager, see UEManager.load().

        Parameters:
            save_path (str): Path to file to save benchmark results to.
        """
        if len(self.metrics) == 0:
            raise Exception("Nothing to save. Consider calling manager() first.")
        torch.save(
            {
                "metrics": self.metrics,
                "gen_metrics": self.gen_metrics,
                "estimations": self.estimations,
                "stats": self.stats,
            },
            save_path,
        )

    @staticmethod
    def load(load_path: str) -> "UEManager":
        """
        Loads UEManager from the specified path. To save the calculated manager results, see UEManager.save().

        Parameters:
            load_path (str): Path to file with saved benchmark results to load.
        """
        res_dict = torch.load(load_path)
        man = UEManager(None, None, [], [], [], [])
        man.metrics = res_dict.get("metrics", None)
        man.gen_metrics = res_dict.get("gen_metrics", None)
        man.estimations = res_dict.get("estimations", None)
        man.stats = res_dict.get("stats", None)
        return man
