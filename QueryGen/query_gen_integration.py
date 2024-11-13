import math
import os

import torch
from transformers import LogitsProcessorList, LogitsProcessor
from unsloth import FastLanguageModel
from collections import Counter

from GraphGen.graph_gen_integration import Triple, get_probable_tokens_in_tree


def disable_tokens(scores, banned_tokens):
    for token in banned_tokens:
        scores[0][token] = -math.inf
    return scores

def only_allow_tokens(scores, allowed_tokens):
    for score in len(scores[0]):
        if score in allowed_tokens:
            continue
        scores[0][score] = -math.inf
    return scores

def find_largest_index(lst: list, value):
    try:
        return len(lst) - 1 - lst[::-1].index(value)
    except:
        return -1  # not found

def get_triple_index(ids: list, sep_token_idx, keyword_end_token, var_token):
    count = Counter(ids[sep_token_idx:])
    return count[keyword_end_token] + count[var_token]

# Enforce structure, keywords
class SPARQLLogits(LogitsProcessor):

    t_s_tree = {} # Subject/Object keywords
    r_tree = {} # Predicate keywords

    def __init__(self, _tokenizer, t_s_tree, r_tree):
        self.tokenizer = _tokenizer

        self.query_start_token = _tokenizer("{")['input_ids'][1]   # { ...
        self.query_end_token = _tokenizer("}")['input_ids'][1]     # }<eos>
        self.eos_token = _tokenizer("<eos>")['input_ids'][1]       # }<eos>

        self.variable_token = _tokenizer('?')['input_ids'][1]      # ?uri
        self.keyword_start_token = _tokenizer('<')['input_ids'][1] # <subject...
        self.keyword_end_token = _tokenizer('>')['input_ids'][1]   # ...subject>

        self.seperator_token = _tokenizer('.')['input_ids'][1]     # ...<object> . ?uri...

        self.t_s_tree = t_s_tree
        self.r_tree = r_tree


    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        # get the closest token of interest
        ids_list = input_ids.tolist()[0]
        # print(ids_list)

        # ... }
        if find_largest_index(ids_list, self.query_end_token) != -1:
            # end
            allowed_tokens = [self.eos_token]
            return only_allow_tokens(scores, allowed_tokens)

        # SELECT ... {
        if find_largest_index(ids_list, self.query_start_token) == -1:
            # not started yet
            return scores

        sep_near_pos = find_largest_index(ids_list, self.seperator_token) # .
        var_near_pos = find_largest_index(ids_list, self.variable_token) # ?
        key_start_near_pos = find_largest_index(ids_list, self.keyword_start_token) # <
        key_end_near_pos = find_largest_index(ids_list, self.keyword_start_token) # >

        near_pos = max(var_near_pos, key_start_near_pos, key_end_near_pos, sep_near_pos)

        disabled_tokens = [self.query_start_token]
        allowed_tokens = []

        triple_index = get_triple_index(ids_list, self.seperator_token, self.keyword_end_token, self.variable_token)

        # New special token enforce
        if near_pos == sep_near_pos: # .
            allowed_tokens += [self.query_end_token, self.keyword_start_token, self.variable_token]

        elif near_pos == var_near_pos: # ?
            if triple_index == 2: # object
                disabled_tokens += [self.variable_token, self.keyword_start_token, self.keyword_end_token] # content, ., } allowed
            else:
                disabled_tokens += [self.keyword_end_token, self.seperator_token, self.query_end_token] # content, <, ? allowed


        elif near_pos == key_start_near_pos: # <

            generated = ids_list[(near_pos + 1):]  # All currently generated keywords

            # allowed_tokens += [self.keyword_end_token] # In tree now
            if triple_index == 0 or triple_index == 2:
                # TS
                allowed_tokens += get_probable_tokens_in_tree(self.t_s_tree, generated)

            else:
                # R
                allowed_tokens += get_probable_tokens_in_tree(self.r_tree, generated)

        elif near_pos == key_end_near_pos: # >
            allowed_tokens += [self.seperator_token, self.query_end_token]

        else:
            print("jjifdoasjddio not supposed to happen arghh!")
            raise Exception

        if near_pos == len(ids_list) - 1:
            # Just generated start token
            if near_pos == sep_near_pos: # .
                allowed_tokens += [self.query_end_token, self.keyword_start_token, self.variable_token]
            elif near_pos == var_near_pos:  # ?
                disabled_tokens += [self.eos_token, self.keyword_start_token, self.keyword_end_token,
                                   self.variable_token,
                                   self.seperator_token] # stop all tokens
            elif near_pos == key_start_near_pos: # <
                disabled_tokens += [self.keyword_start_token, self.keyword_end_token, self.query_end_token, self.seperator_token, self.eos_token]
            elif near_pos == key_end_near_pos: # >
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
{}"""


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
        ts_end_tokens = self.tokenizer(">")['input_ids'][1:]

        for tokens in new_ts_tokens:
            tokens += ts_end_tokens
            # print(tokens)

            current_depth = self.ts_tree

            for token in tokens:
                if token not in current_depth:  # token doesn't exist, create
                    current_depth[token] = {}

                current_depth = current_depth[token]

        # r map
        r_end_tokens = self.tokenizer(">")['input_ids'][1:]

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
            model_name=os.path.join(os.path.dirname(__file__),"query_2b_600step"),
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
            ], return_tensors="pt").to("cuda")

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
            print(self.ts_tree, self.r_tree)
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

        return sparql_string