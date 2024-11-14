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


class chatGPT_Metric_35_Def_0(GenerationMetric):
    """
    Calculates ChatGPT 3.5 based NLG scores.
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
        


        self.model = model
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.task = task
        self.view = view
        self.keyword_list = ['overall', 'coherence', 'consistency', 'fluency', 'relevance']

        self.file_name = file_name
        self.temperature=temperature
        
        self.client = OpenAI(api_key=open_ai_token)
        self.aspect_dict = {

'coherence': "coherence refers to whether all the sentences in the given generated text form a coherent body",
'consistency': "consistency is the factual alignment between the generated text and the original text",
'fluency': "fluency represents the quality of individual sentences in the generated text",
'relevance': "relevance means whether the summary contains only the important information of the source document",
'overall': "overall includes all below four aspects: 1. coherence refers to whether all the sentences in the given generated text form a coherent body; 2. consistency is the factual alignment between the generated text and the original text; 3. fluency represents the quality of individual sentences in the generated text; 4. relevance means whether the summary contains only the important information of the source document"

}

    def __str__(self):
        return f"chatGPT_def_0_{self.model}_{self.logprobs}_{self.top_logprobs}_{self.task}_{self.view}"

    

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

    def _get_chatGPT_multiple_socre(self, gen_list, ref_list, input_list):

        view_list = []
        # all_list = []
        with open(self.file_name, 'a+') as f:
            for gen, ref, inp in zip(gen_list, ref_list, input_list):
                mid_dict = {}
                for keyword in self.keyword_list:
                    try:
                        mid_res = self._get_chatGPT_single_socre(gen, ref, inp, keyword)
                    except:
                        time.sleep(60)
                        mid_res = self._get_chatGPT_single_socre(gen, ref, inp, keyword)
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


    def _get_chatGPT_single_socre(self, gen, ref, inp, aspect):
        inp = inp[39:]

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[

                # with definition, with input, without ground truth summary
                {
                    "role": "user",
                    "content": f"I want to know the quality of the generated summary that is {gen} for the orignal text that is {inp}. Please tell me the {aspect} score of {self.task} task. The {aspect} score is ranged from [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and a higher score means better in terms of {aspect} with a definition as {self.aspect_dict[aspect]}. Please only directly output this {aspect} score.",
                },

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
        Calculates Rouge score between stats['greedy_texts'] and target_texts.

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


        
        input_text = stats['input_texts']


        res = self._get_chatGPT_multiple_socre(stats[greedy_text_key], target_texts, input_text)

        return np.array(res)
