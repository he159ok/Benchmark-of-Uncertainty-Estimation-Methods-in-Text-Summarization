import numpy as np
# from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from typing import List, Dict
from .generation_metric import GenerationMetric
import re
import json

# from openai import OpenAI
import anthropic

client = anthropic.Anthropic(
    api_key="xxxx",
)


class claude_Metric(GenerationMetric):
    """
    Calculates Claude based NLG scores.
    """

    def __init__(self,
                 depend=["greedy_texts"],
                 model="claude-3-opus-20240229",
                 logprobs=True,
                 top_logprobs=1,
                 task="text summarization",
                 view="overall",
                 file_name=None,
                 temperature=0.0,
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
        


        # model_card = 'tals/albert-xlarge-vitaminc-mnli'
        self.model = model
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.task = task
        self.view = view
        self.keyword_list = ['overall', 'coherence', 'consistency', 'fluency', 'relevance']
       
        self.file_name = file_name
        self.temperature=temperature


    def __str__(self):
        return f"chatGPT_{self.model}_{self.logprobs}_{self.top_logprobs}_{self.task}_{self.view}"

    

    def extract_floats(self, text):
        # 使用正则表达式匹配浮点数
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

    def _get_cluade_multiple_socre(self, hyp_list, ref_list):

        view_list = []
        # all_list = []
        with open(self.file_name, 'a+') as f:
            for hyp, ref in zip(hyp_list, ref_list):
                mid_dict = {}
                for keyword in self.keyword_list:
                    mid_res = self._get_claude_single_socre(hyp, ref, keyword)
                    mid_dict[keyword] = mid_res


                view_list.append(mid_dict[self.view])

                # save file
                json_str = json.dumps(
                    # {(sr_sent[i] + gt_sent[i]): eval_scores[i]}
                    {ref: mid_dict}
                )
                print({ref: mid_dict})
                f.write(json_str)
                f.write('\n')

        return view_list


    def _get_claude_single_socre(self, hyp, ref, aspect):

        completion = client.messages.create(
            model=self.model,
            system="You are an analysis agent to response the uncertainty of each generation via an given aspect.",
            messages=[
                {"role": "user",
                 "content":
                 [
                     {
                         "type": "text",
                         "text": f"The generated text: {hyp}. The original text: {ref}. Now, tell me the {aspect} score of {self.task} task. The {aspect} score is ranged from [1, 2, 3, 4, 5] and a higher score means better in terms of {aspect}. Please only directly output this score."
                     }
                 ]
                }
            ],
            max_tokens=1000,
            temperature=self.temperature,
        )

        mid_res = self.extract_floats(completion.content[0].text)
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
        Calculates Claude based NLG scores.

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



        res = self._get_cluade_multiple_socre(stats[greedy_text_key], target_texts)

        return np.array(res)
