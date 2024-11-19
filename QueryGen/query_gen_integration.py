import math
import os

import torch
from torch import Tensor
from transformers import LogitsProcessorList, LogitsProcessor
from unsloth import FastLanguageModel
from collections import Counter

from GraphGen.graph_gen_integration import Triple, get_probable_tokens_in_tree


def disable_tokens(scores, banned_tokens):
    for token in banned_tokens:
        scores[0][token] = -math.inf
    return scores

def only_allow_tokens(scores, allowed_tokens) -> torch.Tensor:
    # Faster method
    new_scores = torch.tensor([[-math.inf] * len(scores[0])], dtype=torch.float16, device='cuda')

    # print(f"S{scores.shape}")
    # print(f"M{new_scores.shape}")
    # print(new_scores)

    # print(allowed_tokens)

    for token in allowed_tokens:
        new_scores[0][token] = scores[0][token]

    return new_scores

def find_largest_index(lst: list, value):
    try:
        return len(lst) - 1 - lst[::-1].index(value)
    except:
        return -1  # not found

def get_triple_index(ids: list, sep_token, keyword_start_token, var_token):
    start_pos = find_largest_index(ids, sep_token) # .
    if start_pos == -1:
        start_pos = 0
    count = Counter(ids[start_pos:])
    return count[keyword_start_token] + count[var_token] - 1

# Enforce structure, keywords
class SPARQLLogits(LogitsProcessor):

    t_s_tree = {} # Subject/Object keywords
    r_tree = {} # Predicate keywords

    def __init__(self, _tokenizer, t_s_tree, r_tree):
        self.tokenizer = _tokenizer

        self.eos_token = _tokenizer("<eos>")['input_ids'][1]                # <eos>
        self.pad_token = _tokenizer("<pad>")['input_ids'][1]                # <pad> DISABLE

        self.keyword_start_token = _tokenizer("<unused10>")['input_ids'][1] # <
        self.keyword_end_token = _tokenizer("<unused11>")['input_ids'][1]   # >

        self.query_start_token = _tokenizer("<unused12>")['input_ids'][1]   # {
        self.query_end_token = _tokenizer("<unused13>")['input_ids'][1]     # }

        self.variable_token = _tokenizer("<unused14>")['input_ids'][1]      # ?
        self.seperator_token = _tokenizer("<unused15>")['input_ids'][1]     # .

        self.space_token = _tokenizer(' ')['input_ids'][1]                  # Just Space

        self.t_s_tree = t_s_tree
        self.r_tree = r_tree


    def __call__(self, input_ids, scores) -> torch.Tensor:
        # get the closest token of interest
        ids_list = input_ids.tolist()[0]
        # print(ids_list)
        print(self.tokenizer.decode(ids_list).split("### SPARQL:")[1])

        # ... } SELECT
        if find_largest_index(ids_list, self.query_end_token) != -1:
            # Model can freely generate whatever
            print("END")
            return scores

        # {
        if find_largest_index(ids_list, self.query_start_token) == -1:
            # not started yet
            print("NOT SUPPOSED TO HAPPEN OOPS")
            return scores

        print("CONTENT")
        sep_near_pos = find_largest_index(ids_list, self.seperator_token) # .
        var_near_pos = find_largest_index(ids_list, self.variable_token) # ?
        key_start_near_pos = find_largest_index(ids_list, self.keyword_start_token) # <
        key_end_near_pos = find_largest_index(ids_list, self.keyword_start_token) # >

        near_pos = max(var_near_pos, key_start_near_pos, key_end_near_pos, sep_near_pos)

        disabled_tokens = [self.query_start_token, self.pad_token, self.eos_token]
        allowed_tokens = []

        content = ids_list[(find_largest_index(ids_list, self.query_start_token) + 1):]

        triple_index = get_triple_index(content, self.seperator_token, self.keyword_start_token, self.variable_token)


        # New special token enforce
        if near_pos == sep_near_pos: # .
            allowed_tokens += [self.query_end_token, self.keyword_start_token, self.variable_token]

        elif near_pos == var_near_pos: # ?
            if triple_index == 2: # object
                # TODO: CHANGE TO DISABLED SINCE ALLOWED DISABLES EVERYTHING ELSE AHAHAHAH
                allowed_tokens += [self.seperator_token, self.query_end_token] # content, ., } allowed
            else:
                allowed_tokens += [self.variable_token, self.keyword_start_token] # content, ?, < allowed

            allowed_tokens += [self.space_token] # Should've removed spaces when training. This might cause problems or improve things, who knows

        elif near_pos == key_start_near_pos: # <

            generated = ids_list[(near_pos + 1):]  # All currently generated keywords

            # allowed_tokens += [self.keyword_end_token] # In tree now
            if triple_index == 0 or triple_index == 2:
                # TS
                allowed_tokens += get_probable_tokens_in_tree(self.t_s_tree, generated)
            else:
                # R
                allowed_tokens += get_probable_tokens_in_tree(self.r_tree, generated)

            print(f"[CURRENT] {self.tokenizer.decode(generated)}")
            print(f"[ALLOW->] {self.tokenizer.decode(allowed_tokens)}")

        elif near_pos == key_end_near_pos: # >
            if triple_index == 2: # object
                allowed_tokens += [self.seperator_token, self.query_end_token]
            else:
                disabled_tokens += [self.seperator_token, self.query_end_token]

        else:
            print("jjifdoasjddio not supposed to happen arghh!")
            raise Exception

        print(triple_index)

        if near_pos == len(ids_list) - 1:
            # Just generated special token
            if near_pos == sep_near_pos: # .
                allowed_tokens += [self.query_end_token, self.keyword_start_token, self.variable_token, self.space_token]
            elif near_pos == var_near_pos:  # ?
                disabled_tokens += [self.query_end_token, self.eos_token,
                                    self.keyword_start_token, self.keyword_end_token,
                                    self.variable_token,
                                    self.seperator_token,
                                    self.space_token] # stop all tokens
            elif near_pos == key_start_near_pos: # <
                disabled_tokens += [self.keyword_start_token, self.keyword_end_token, self.query_end_token, self.seperator_token, self.eos_token]
            elif near_pos == key_end_near_pos: # >
                allowed_tokens += [self.space_token]
                disabled_tokens += [self.keyword_end_token]
            else:
                print("jjifdoasjddio not supposed to happen too arghh!")
                raise Exception

        if len(allowed_tokens) > 0:
            scores = only_allow_tokens(scores, allowed_tokens)
        scores = disable_tokens(scores, disabled_tokens)
        # print(input_ids)
        # print(f"{scores} -> {disabled_scores}")
        return scores


class QueryGen:
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
The user has provided a question. Convert the question in Natural Language to a SPARQL query. each triple in the SPARQL should follow the order of subject predicate object.
### User Question:
{}
### SPARQL:
<unused12>{}"""

    def __init__(self, existing_rdfs: list[Triple]): # Keyword is enforced
        # init tokenizer, models, etc.
        self.ts_tree = {}
        self.r_tree = {}

        self.model = None
        self.tokenizer = None

        self.setup_model()

        if existing_rdfs is not None and len(existing_rdfs) > 0:
            self.append_rdfs(existing_rdfs)

    def append_rdfs(self, rdfs:[Triple]):
        # Tokenize everything
        new_ts_tokens = [
            self.tokenizer(keyword)['input_ids'][1:]  # First token is <bos>, removed
            for keyword in set(
                [rdf.object for rdf in rdfs] + [rdf.subject for rdf in rdfs]
            )]

        new_r_tokens = [
            self.tokenizer(keyword)['input_ids'][1:]  # First token is <bos>, removed
            for keyword in set(
                [rdf.predicate for rdf in rdfs]
            )]

        # ts map
        ts_end_tokens = self.tokenizer("<unused11>")['input_ids'][1:] # >

        for tokens in new_ts_tokens:
            tokens += ts_end_tokens
            # print(tokens)

            current_depth = self.ts_tree

            for token in tokens:
                if token not in current_depth:  # token doesn't exist, create
                    current_depth[token] = {}

                current_depth = current_depth[token]

        # r map
        r_end_tokens = self.tokenizer("<unused11>")['input_ids'][1:] # >

        for tokens in new_r_tokens:
            tokens += r_end_tokens

            current_depth = self.r_tree

            for token in tokens:
                if token not in current_depth:  # token doesn't exist, create
                    current_depth[token] = {}

                current_depth = current_depth[token]

        # DEBUG
        if False:
            def get_all_keys(d, tab):
                for key, value in d.items():
                    yield key, tab
                    if isinstance(value, dict):
                        yield from get_all_keys(value, tab + "  ")

            print("TS-------------")
            for x, tab in get_all_keys(self.ts_tree, ""):
                print(tab + self.tokenizer.decode(x))

            print("R--------------")
            for x, tab in get_all_keys(self.r_tree, ""):
                print(tab + self.tokenizer.decode(x))


    def setup_model(self):
        # setup model helper function
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.path.join(os.path.dirname(__file__),"query_9b_4epoch"),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference

    def free_model(self):
        # Free model from GPU memory to load other models
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def reset_model(self, existing_rdfs):
        self.ts_tree = {}
        self.r_tree = {}

        if existing_rdfs is not None and len(existing_rdfs) > 0:
            self.append_rdfs(existing_rdfs)


    def generate(self, input_text):
        # SPARQL Gen
        inputs = self.tokenizer(
            [
                self.prompt.format(
                    input_text,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt"
        ).to("cuda")

        if len(self.ts_tree) > 0 and len(self.r_tree) > 0:
            outputs = self.model.generate(
                **inputs,

                # temperature = 0.9,
                # max_new_tokens = max(len(text) + 100, 700),

                logits_processor=LogitsProcessorList(
                    [SPARQLLogits(self.tokenizer, self.ts_tree, self.r_tree)]),
                # num_beams = 3,
                # early_stopping = True,
                repetition_penalty=1.05,
                renormalize_logits=True,

                use_cache=True,
                # Use cache = false is broken haha, but beam search is broken when not using cache hahahah ;-;
            )
        else:
            print("No existing RDFs, reverting to default gen without SPARQLLogits")
            outputs = self.model.generate(
                **inputs,

                # temperature = 0.9,
                # max_new_tokens = max(len(text) + 100, 700),
                # num_beams = 3,
                # early_stopping = True,
                repetition_penalty=1.05,
                renormalize_logits=True,

                use_cache=True,
                # Use cache = false is broken haha, but beam search is broken when not using cache hahahah ;-;
            )


        response = self.tokenizer.batch_decode(outputs)
        response = response[0].replace('\n', '')
        sparql_string = response.split("### SPARQL:")[1]
        sparql_string = sparql_string.removeprefix("<bos>").removesuffix("<eos>")

        # Convert special tokens back to original

        token_table = {
            "<unused10>": "<",
            "<unused11>": ">",
            "<unused12>": "{",
            "<unused13>": "}",
            "<unused14>": "?",
            "<unused15>": "."
        }

        for k, v in token_table.items():
            sparql_string = sparql_string.replace(k, v)

        if len(sparql_string.split("}")) != 2:
            print("AHAHHAHAHA noooooo")

        sparql_string = sparql_string.split("}")[1] + sparql_string.split("}")[0] + "}"

        return sparql_string