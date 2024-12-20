from abc import ABC, abstractmethod

import torch
import transformers.pipelines
from overrides import overrides

import os


class BaseGuardrail(ABC):
    @abstractmethod
    def __call__(self, text: str) -> bool:
        """
        detects if input is harmful/inappropriate.
        """
        return False


class NoGuardrail(BaseGuardrail):

    @overrides
    def __call__(self, text: str) -> bool:
        return True


class SelfIEGuardrail(BaseGuardrail):
    # interpretation_pre_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|> Summarize the following text.\n"
    # interpretation_prompt = "\nSummarize the previous text. <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    interpretation_pre_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|> Determine whether the following text contains harmful information or not. Answer only yes or no.\n"
    interpretation_prompt = "\nDetermine whether the previous text contains harmful information or not. Answer only yes or no. <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def __init__(self, answer_llm: transformers.pipelines.Pipeline):
        self.answer_llm = answer_llm
        self.tokenizer = self.answer_llm.tokenizer
        self.model = self.answer_llm.model

    @overrides
    def __call__(self, input_text: str) -> bool:
        tokenized_input = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)['input_ids']
        model_output = self.model(tokenized_input, return_dict=True, output_attentions=False, output_hidden_states=True)
        # num_hidden_layers = self.model.config.num_hidden_layers
        hidden_embeds_list = model_output['hidden_states']  # shape: (batch_size, seq_len, hidden_dim)
        concatenated_hidden_embeds = torch.concat(hidden_embeds_list, dim=1)

        interpretation_pre_prompt_token_ids = self.tokenizer(SelfIEGuardrail.interpretation_pre_prompt,
                                                             return_tensors='pt').to(self.model.device)['input_ids']
        interpretation_pre_prompt_embeds = self.model.model.embed_tokens(interpretation_pre_prompt_token_ids)

        interpretation_prompt_token_ids = self.tokenizer(SelfIEGuardrail.interpretation_prompt,
                                                         return_tensors='pt').to(self.model.device)['input_ids']
        interpretation_prompt_embeds = self.model.model.embed_tokens(interpretation_prompt_token_ids)
        concatenated_all_embeds = torch.concat((interpretation_pre_prompt_embeds, concatenated_hidden_embeds,
                                                interpretation_prompt_embeds), dim=1)

        selfie_result = self.model.generate(inputs=None, inputs_embeds=concatenated_all_embeds, max_new_tokens=100)
        return "Yes" in selfie_result or "yes" in selfie_result

    def try_selfie(self, input_text: str) -> str:
        tokenized_input = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)['input_ids']
        model_output = self.model(tokenized_input, return_dict=True, output_attentions=False, output_hidden_states=True)
        # num_hidden_layers = self.model.config.num_hidden_layers
        hidden_embeds_list = model_output['hidden_states']  # shape: (batch_size, seq_len, hidden_dim)
        concatenated_hidden_embeds = torch.concat(hidden_embeds_list, dim=1)

        interpretation_pre_prompt_token_ids = self.tokenizer(SelfIEGuardrail.interpretation_pre_prompt,
                                                             return_tensors='pt').to(self.model.device)['input_ids']
        interpretation_pre_prompt_embeds = self.model.model.embed_tokens(interpretation_pre_prompt_token_ids)

        interpretation_prompt_token_ids = self.tokenizer(SelfIEGuardrail.interpretation_prompt,
                                                         return_tensors='pt').to(self.model.device)['input_ids']
        interpretation_prompt_embeds = self.model.model.embed_tokens(interpretation_prompt_token_ids)
        concatenated_all_embeds = torch.concat((interpretation_pre_prompt_embeds, concatenated_hidden_embeds,
                                                interpretation_prompt_embeds), dim=1)

        selfie_result = self.model.generate(inputs=None, inputs_embeds=concatenated_all_embeds, max_new_tokens=100)
        selfie_output_text = self.tokenizer.batch_decode(selfie_result, skip_special_tokens=True)
        return selfie_output_text


class KeywordFilterGuardrail(BaseGuardrail):
    def __init__(self):
        # Load harmful keyword set from a text file
        with open(os.getcwd() + "/agent_project/verified_harmful_keywords.txt", "r") as f:
            self.harmful_keywords_llm = set(line.strip() for line in f)

    @overrides
    def __call__(self, text: str) -> bool:
        # Return True if any harmful keyword is found in the input text
        return any(phrase.lower() in text.lower() for phrase in self.harmful_keywords_llm)


