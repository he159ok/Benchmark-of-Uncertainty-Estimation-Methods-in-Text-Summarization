
## Can We Trust the Performance Evaluation of Uncertainty Estimation Methods in Text Summarization?

This is the code and data for EMNLP 2024 paper [Can We Trust the Performance Evaluation of Uncertainty Estimation Methods in Text Summarization?](https://aclanthology.org/2024.emnlp-main.923/)

### Introduction

Text summarization, a key natural language generation (NLG) task, is vital in various domains. However, the high cost of inaccurate summaries in risk-critical applications, particularly those involving human-in-the-loop decision-making, raises concerns about the reliability of uncertainty estimation on text summarization (UE-TS) evaluation methods. This concern stems from the dependency of uncertainty model metrics on diverse and potentially conflicting NLG metrics. To address this issue, we introduce a comprehensive UE-TS benchmark incorporating 31 NLG metrics across four dimensions. The benchmark evaluates the uncertainty estimation capabilities of two large language models and one pre-trained language model on three datasets, with human-annotation analysis incorporated where applicable. We also assess the performance of 14 common uncertainty estimation methods within this benchmark. Our findings emphasize the importance of considering multiple uncorrelated NLG metrics and diverse uncertainty estimation methods to ensure reliable and efficient evaluation of UE-TS techniques.

### Env

This work is done based on the code base of [LM-Polygraph: Uncertainty Estimation for Language Models](https://aclanthology.org/2023.emnlp-demo.41.pdf).

```
Below lists the steps to configure the envs used in our work.

conda install python=3.10

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

#### install lm-polygraph 
# (The lm-polygraph has been updated, our uploaded code includes our previous used lm-polygraph)
cd lm-polygraph
pip install -r requirements.txt
pip install .

#### install other related env
pip install cleantext
pip install openai

pip install prettytable


#### if Llama 2 is expected
#### log into huggingface for Llama2 authorization
huggingface-cli login

```

### Run

The representative running scripts have been uploaded into `./mysript/` folder. 

For example,

```angular2html
cd ./mysript

bash ./a100_ale_gpt35_r4_def35_both.sh
```

To better understand PRR score, please run `./understanding_prr.py`.

### Intermediate Data

Our intermediate data has been uploaded into a drive at [here](https://drive.google.com/file/d/1T061ZbbNee-Cpj5UxLYaHLaj4FUpqwjX/view?usp=sharing). More introduction about the intermediate data will come later.