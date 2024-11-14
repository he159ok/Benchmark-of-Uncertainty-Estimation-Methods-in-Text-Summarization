import numpy as np
# from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from typing import List, Dict
from .generation_metric import GenerationMetric
import re
import json
import time

from openai import OpenAI


class chatGPT_Metric(GenerationMetric):
    """
    Calculates ChatGPT 3.5 based NLG metric scores.
    """

    def __init__(self,
                 depend=["greedy_texts"],
                 model="gpt-3.5-turbo",
                 logprobs=True,
                 top_logprobs=1,
                 task="text summarization",
                 view="overall",
                 file_name=None,
                 temperature=0.0,
                 open_ai_token=None,
                 ):
        """
        Parameters:
            rouge_name (str): rouge metric type. Possible values:
                * rouge1
                * rouge2
                * rougeL

            model_card (str): the NLI model used for hallucination evaluation

        """
        super().__init__(depend, "sequence")
        # self.rouge_name = rouge_name
        # self.scorer = rouge_scorer.RougeScorer([rouge_name], use_stemmer=True)


        # model_card = 'tals/albert-xlarge-vitaminc-mnli'
        self.model = model
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.task = task
        self.view = view
        self.keyword_list = ['overall', 'coherence', 'consistency', 'fluency', 'relevance']
        self.file_name = file_name
        self.temperature=temperature
        self.client = OpenAI(api_key=open_ai_token)

    def __str__(self):
        return f"chatGPT_{self.model}_{self.logprobs}_{self.top_logprobs}_{self.task}_{self.view}"

    def extract_floats(self, text):
        res = None
        try:
            try:
                floats = re.findall(r'\d+', text)
                res = float(floats[0])
            except:
                floats = re.findall(r'\d+\.\d+', text)
                res = float(floats[0])
            return res
        except:
            return res

    def _get_chatGPT_multiple_socre(self, hyp_list, ref_list):

        view_list = []
        with open(self.file_name, 'a+') as f:
            for hyp, ref in zip(hyp_list, ref_list):
                mid_dict = {}
                for keyword in self.keyword_list:
                    try:
                        mid_res = self._get_chatGPT_single_socre(hyp, ref, keyword)
                    except:
                        time.sleep(60)
                        mid_res = self._get_chatGPT_single_socre(hyp, ref, keyword)
                    mid_dict[keyword] = mid_res


                view_list.append(mid_dict[self.view])

                # save file
                json_str = json.dumps(
                    {ref: mid_dict}
                )
                print({ref: mid_dict})
                f.write(json_str)
                f.write('\n')

        return view_list


    def _get_chatGPT_single_socre(self, hyp, ref, aspect):

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[

                {"role": 'system',
                 "content": 'You are an analysis agent to response the uncertainty of each generation via an given aspect.',
                    "role": "user",
                 "content": f"The generated text: {hyp}. The original text: {ref}. Now, tell me the {aspect} score of {self.task} task. The {aspect} score is ranged from [1, 2, 3, 4, 5] and a higher score means better in terms of {aspect}. Please only directly output this score."
                            }

            ],
            logprobs=True,
            top_logprobs=1,
            seed=42,
            temperature=self.temperature,
        )
        # in case of the output content is not applicable to float()
        mid_res = self.extract_floats(completion.choices[0].message.content)
        if mid_res is None:
            mid_res = -1

        return mid_res


    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
        white=None,
    ) -> np.ndarray:
        """
        Calculates ChatGPT 3.5 based NLG scores.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of Rouge Scores for each sample in input.
        """
        if white:
            greedy_text_key = "greedy_texts"
        else:
            greedy_text_key = "blackbox_greedy_texts"


        res = self._get_chatGPT_multiple_socre(stats[greedy_text_key], target_texts)

        return np.array(res)
