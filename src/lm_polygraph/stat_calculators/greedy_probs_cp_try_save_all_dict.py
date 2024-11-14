import torch
import numpy as np

from typing import Dict, List

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator

# from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from src.lm_polygraph.utils2.model import WhiteboxModel, BlackboxModel
from MyFunc import save_list_dict_to_json_inbat, transfer_dict_list_format_complex

class BlackboxGreedyTextsCalculator(StatCalculator):
    """
    Calculates generation texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    def __init__(self):
        super().__init__(["blackbox_greedy_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates generation texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] generation texts at 'blackbox_greedy_texts' key.
        """
        with torch.no_grad():
            sequences = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                temperature=model.parameters.temperature,
                top_p=model.parameters.topp,
                top_k=model.parameters.topk,
                presence_penalty=model.parameters.presence_penalty,
                repetition_penalty=model.parameters.repetition_penalty,
                n=1,
            )

        return {"blackbox_greedy_texts": sequences}


class GreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    def __init__(self):
        super().__init__(
            [
                "input_texts",
                "input_tokens",
                "greedy_log_probs",
                "greedy_tokens",
                "greedy_texts",
                "greedy_log_likelihoods",
                "train_greedy_log_likelihoods",
                "embeddings",
            ],
            [],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        cal_save_embeddings=0,
        result_dict_save_file_name=None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'input_texts' (List[str]): input texts batch,
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """
        if cal_save_embeddings:
            batch: Dict[str, torch.Tensor] = model.tokenize(texts)
            batch = {k: v.to(model.device()) for k, v in batch.items()}
            with torch.no_grad():
                out = model.generate(
                    **batch,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    min_length=2,
                    output_attentions=False,
                    output_hidden_states=True,
                    temperature=model.parameters.temperature,
                    top_k=model.parameters.topk,
                    top_p=model.parameters.topp,
                    do_sample=model.parameters.do_sample,
                    num_beams=model.parameters.num_beams,
                    presence_penalty=model.parameters.presence_penalty,
                    repetition_penalty=model.parameters.repetition_penalty,
                    suppress_tokens=(
                        []
                        if model.parameters.allow_newlines
                        else [
                            t
                            for t in range(len(model.tokenizer))
                            if "\n" in model.tokenizer.decode([t])
                        ]
                    ),
                    num_return_sequences=1,
                )
                logits = torch.stack(out.scores, dim=1) # out.score is the score for each token in each sample [batch size, token size, total word dict dimen]

                sequences = out.sequences # out.sequences is the token id for each sentence [batch size, token size]
                embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                    out, batch, model.model_type
                )

            cut_logits = []
            cut_sequences = []
            cut_texts = []
            for i in range(len(texts)):
                if model.model_type == "CausalLM":
                    seq = sequences[i, batch["input_ids"].shape[1] :].cpu()
                else:
                    seq = sequences[i, 1:].cpu() # [max token size] for only one sample
                length, text_length = len(seq), len(seq)
                for j in range(len(seq)):
                    if seq[j] == model.tokenizer.eos_token_id:
                        length = j + 1
                        text_length = j # adjust the max token size to the real token size
                        break
                cut_sequences.append(seq[:length].tolist()) # extract single samples' token id
                cut_texts.append(model.tokenizer.decode(seq[:text_length])) # extract single samples' texts
                cut_logits.append(logits[i, :length, :].cpu().numpy())  # # extract single samples' token score (need to check whether not softmax)

            ll = []
            for i in range(len(texts)):
                log_probs = cut_logits[i] # (42, 42024)
                tokens = cut_sequences[i]
                assert len(tokens) == len(log_probs)
                ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))]) # each chosen token's score

            if model.model_type == "CausalLM":
                embeddings_dict = {
                    "embeddings_decoder": embeddings_decoder,
                }
            elif model.model_type == "Seq2SeqLM":
                embeddings_dict = {
                    "embeddings_encoder": embeddings_encoder,
                    "embeddings_decoder": embeddings_decoder,
                }
            else:
                raise NotImplementedError

            result_dict = {
                "input_texts": texts,
                "input_tokens": batch["input_ids"].to("cpu").tolist(),
                "greedy_log_probs": cut_logits, # [array(), array()]
                "greedy_tokens": cut_sequences,
                "greedy_texts": cut_texts,
                "greedy_log_likelihoods": ll,
            }
            result_dict.update(embeddings_dict)

            # save the results
            saved_result_dict = transfer_dict_list_format_complex(result_dict, input_text_key='input_texts', list_np=["greedy_log_probs"], tensor_list=['embeddings_encoder', 'embeddings_decoder'], np_ele=["greedy_log_likelihoods"], remove_list=["greedy_log_likelihoods"])
            save_list_dict_to_json_inbat(saved_result_dict, result_dict_save_file_name)
            print(f'generation results are saved to {result_dict_save_file_name}')
            return result_dict
        else:
            # read the data
            loaded_result_dict = None

            # recal the removed eles
            cut_logits = [np.array(ele) for ele in loaded_result_dict["greedy_log_probs"]]
            loaded_result_dict["greedy_log_probs"] = cut_logits
            ll = []
            for i in range(len(texts)):
                log_probs = cut_logits[i] # (42, 42024)
                tokens = cut_sequences[i]
                assert len(tokens) == len(log_probs)
                ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))]) # each chosen token's score

            result_dict=None
            return result_dict
