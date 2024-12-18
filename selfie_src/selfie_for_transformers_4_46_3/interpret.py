from selfie_src.selfie_for_transformers_4_46_3.generate_wrappers import model_forward_interpret, \
    my_generate_interpret
from tqdm import tqdm
import torch
import re


class InterpretationPrompt:
    def __init__(self, tokenizer, interpretation_prompt_sequence):
        self.tokenizer = tokenizer
        self.interpretation_prompt = ""
        self.insert_locations = []

        for part in interpretation_prompt_sequence:
            if isinstance(part, str):
                self.interpretation_prompt += part
            else:
                insert_start = len(self.tokenizer.encode(self.interpretation_prompt))
                self.insert_locations.append(insert_start)
                self.interpretation_prompt += "_ "

        self.tokenized_interpretation_prompt = self.tokenizer(self.interpretation_prompt, return_tensors="pt")

    @staticmethod
    def string_to_tuple(s: str, delimiters: list[str]) -> tuple[str | int]:
        """
        Converts a string into a tuple where substrings separated by delimiters
        are kept as strings, and each delimiter is replaced by an integer zero.

        Parameters:
        - s (str): The input string to be converted.
        - delimiters (List[str]): A list of delimiter substrings.

        Returns:
        - Tuple[Union[str, int], ...]: The resulting tuple with strings and zeros.
        """
        if not delimiters:
            # If no delimiters are provided, return the entire string as a single-element tuple
            return (s,)

        # Escape delimiters to handle any special regex characters
        escaped_delims = [re.escape(d) for d in delimiters]
        # Create a regex pattern that matches any of the delimiters
        pattern = '|'.join(escaped_delims)

        # Use re.split with capturing parentheses to include the delimiters in the result
        split_list = re.split(f'({pattern})', s)

        # Process the split list: replace delimiters with 0, keep other substrings as strings
        result = []
        for item in split_list:
            if item in delimiters:
                result.append(0)
            elif item:  # Exclude empty strings resulting from consecutive delimiters
                result.append(item)

        return tuple(result)


def interpret(original_prompt=None,
              tokenizer=None,
              interpretation_prompt=None,
              model=None,
              tokens_to_interpret=None,
              bs=8,
              max_new_tokens=30,
              k=1) -> dict[str, list]:
    print(f"Interpreting '{original_prompt}' with '{interpretation_prompt.interpretation_prompt}'")
    tokenized_interpretation_prompt = interpretation_prompt.tokenized_interpretation_prompt
    tokenized_interpretation_prompt = tokenized_interpretation_prompt.to(model.device)
    # insert_locations = interpretation_prompt.insert_locations
    original_prompt_inputs = tokenizer(original_prompt, return_tensors="pt").to(model.device)

    interpretation_dict = {
        'prompt': [],
        'interpretation': [],
        'layer': [],
        'token': [],
        'token_decoded': [],
    }

    # prompt_len = original_prompt_inputs['input_ids'].shape[-1]
    outputs = model_forward_interpret(model,
                                      **original_prompt_inputs,
                                      return_dict=True,
                                      output_attentions=False,
                                      output_hidden_states=True,
                                      )

    all_insert_infos = []
    for retrieve_layer, retrieve_token in tokens_to_interpret:
        insert_info = {}
        insert_info['replacing_mode'] = 'normalized'
        insert_info['overlay_strength'] = 1
        insert_info['retrieve_layer'] = retrieve_layer
        insert_info['retrieve_token'] = retrieve_token
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx == k:
                insert_locations = interpretation_prompt.insert_locations
                insert_info[layer_idx] = (insert_locations,
                                          outputs['hidden_states'][retrieve_layer][0][retrieve_token].repeat(1,
                                                                                                             len(insert_locations),
                                                                                                             1))
        all_insert_infos.append(insert_info)

    for batch_start_idx in tqdm(range(0, len(all_insert_infos), bs)):
        with torch.no_grad():
            batch_insert_infos = all_insert_infos[batch_start_idx: min(batch_start_idx + bs, len(all_insert_infos))]

            repeat_prompt_n_tokens = tokenized_interpretation_prompt['input_ids'].shape[-1]

            batched_interpretation_prompt_model_inputs = tokenizer(
                [interpretation_prompt.interpretation_prompt] * len(batch_insert_infos), return_tensors="pt")

            # Move tensors to the same device as the model
            for key in batched_interpretation_prompt_model_inputs:
                batched_interpretation_prompt_model_inputs[key] = batched_interpretation_prompt_model_inputs[key].to(
                    model.device)

            output = my_generate_interpret(**batched_interpretation_prompt_model_inputs, model=model,
                                        max_new_tokens=max_new_tokens, insert_info=batch_insert_infos,
                                        pad_token_id=tokenizer.eos_token_id, output_attentions=False)

            generated_output = output[:, repeat_prompt_n_tokens:]
            cropped_interpretation = tokenizer.batch_decode(generated_output, skip_special_tokens=True)

            for i in range(len(batch_insert_infos)):
                interpretation_dict['prompt'].append(original_prompt)
                interpretation_dict['interpretation'].append(cropped_interpretation[i])
                interpretation_dict['layer'].append(batch_insert_infos[i]['retrieve_layer'])
                interpretation_dict['token'].append(batch_insert_infos[i]['retrieve_token'])
                interpretation_dict['token_decoded'].append(
                    tokenizer.decode(original_prompt_inputs.input_ids[0, batch_insert_infos[i]['retrieve_token']]))
    return interpretation_dict


def my_interpret(original_prompt=None,
              tokenizer=None,
              interpretation_prompt=None,
              model=None,
              tokens_to_interpret=None,
              bs=8,
              max_new_tokens=30,
              k=1) -> dict[str, list]:
    print(f"Interpreting '{original_prompt}' with '{interpretation_prompt.interpretation_prompt}'")
    tokenized_interpretation_prompt = interpretation_prompt.tokenized_interpretation_prompt
    tokenized_interpretation_prompt = tokenized_interpretation_prompt.to(model.device)
    # insert_locations = interpretation_prompt.insert_locations
    original_prompt_inputs = tokenizer(original_prompt, return_tensors="pt").to(model.device)

    interpretation_dict = {
        'prompt': [],
        'interpretation': [],
        'layer': [],
        'token': [],
        'token_decoded': [],
    }

    # prompt_len = original_prompt_inputs['input_ids'].shape[-1]
    outputs = model.forward(model,
                            **original_prompt_inputs,
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=True,
                            )

    all_insert_infos = []
    for retrieve_layer, retrieve_token in tokens_to_interpret:
        insert_info = {}
        insert_info['replacing_mode'] = 'normalized'
        insert_info['overlay_strength'] = 1
        insert_info['retrieve_layer'] = retrieve_layer
        insert_info['retrieve_token'] = retrieve_token
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx == k:
                insert_locations = interpretation_prompt.insert_locations
                insert_info[k] = (insert_locations,
                                  outputs['hidden_states'][retrieve_layer][0][retrieve_token].repeat(1,
                                                                                                     len(insert_locations),
                                                                                                     1))
        all_insert_infos.append(insert_info)

    for batch_start_idx in tqdm(range(0, len(all_insert_infos), bs)):
        with torch.no_grad():
            batch_insert_infos = all_insert_infos[batch_start_idx: min(batch_start_idx + bs, len(all_insert_infos))]

            repeat_prompt_n_tokens = tokenized_interpretation_prompt['input_ids'].shape[-1]

            batched_interpretation_prompt_model_inputs = tokenizer(
                [interpretation_prompt.interpretation_prompt] * len(batch_insert_infos), return_tensors="pt")

            # Move tensors to the same device as the model
            for key in batched_interpretation_prompt_model_inputs:
                batched_interpretation_prompt_model_inputs[key] = batched_interpretation_prompt_model_inputs[key].to(
                    model.device)

            output = my_generate_interpret(**batched_interpretation_prompt_model_inputs, model=model,
                                        max_new_tokens=max_new_tokens, insert_info=batch_insert_infos,
                                        pad_token_id=tokenizer.eos_token_id, output_attentions=False)

            generated_output = output[:, repeat_prompt_n_tokens:]
            cropped_interpretation = tokenizer.batch_decode(generated_output, skip_special_tokens=True)

            for i in range(len(batch_insert_infos)):
                interpretation_dict['prompt'].append(original_prompt)
                interpretation_dict['interpretation'].append(cropped_interpretation[i])
                interpretation_dict['layer'].append(batch_insert_infos[i]['retrieve_layer'])
                interpretation_dict['token'].append(batch_insert_infos[i]['retrieve_token'])
                interpretation_dict['token_decoded'].append(
                    tokenizer.decode(original_prompt_inputs.input_ids[0, batch_insert_infos[i]['retrieve_token']]))
    return interpretation_dict


def interpret_vectors(vecs=None, model=None, interpretation_prompt=None, tokenizer=None, bs=8, k=2, max_new_tokens=30):
    tokenized_interpretation_prompt = interpretation_prompt.tokenized_interpretation_prompt
    tokenized_interpretation_prompt = tokenized_interpretation_prompt.to(model.device)
    insert_locations = interpretation_prompt.insert_locations

    all_interpretations = []

    batch_insert_infos = []

    for vec_idx, vec in enumerate(vecs):
        insert_info = {}
        insert_info['replacing_mode'] = 'normalized'
        insert_info['overlay_strength'] = 1

        # insert_info['replacing_mode'] = 'addition'
        # insert_info['overlay_strength'] = 1000

        insert_info[1] = (insert_locations, vec.repeat(1, len(insert_locations), 1))

        batch_insert_infos.append(insert_info)

        if len(batch_insert_infos) == bs or vec_idx == len(vecs) - 1:
            batched_interpretation_prompt_model_inputs = tokenizer(
                [interpretation_prompt.interpretation_prompt] * len(batch_insert_infos), return_tensors="pt").to(
                'cuda:0')
            repeat_prompt_n_tokens = tokenized_interpretation_prompt['input_ids'].shape[-1]
            output = my_generate_interpret(**batched_interpretation_prompt_model_inputs, model=model,
                                        max_new_tokens=max_new_tokens, insert_info=batch_insert_infos,
                                        pad_token_id=tokenizer.eos_token_id, output_attentions=False)

            cropped_interpretation_tokens = output[:, repeat_prompt_n_tokens:]
            cropped_interpretation = tokenizer.batch_decode(cropped_interpretation_tokens, skip_special_tokens=True)
            all_interpretations.extend(cropped_interpretation)
            batch_insert_infos = []

    return all_interpretations
