import math
import os

import torch
from transformers import LogitsProcessorList, LogitsProcessor
from unsloth import FastLanguageModel
from collections import Counter

from GraphGen.graph_gen_integration import Triple

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
    t_s_keywords = [] # Subject/Object keywords
    r_keywords = [] # Predicate keywords

    def __init__(self, _tokenizer, t_s_keywords, r_keywords):
        self.tokenizer = _tokenizer

        self.query_start_token = _tokenizer("{")['input_ids'][1]  # { ...
        self.query_end_token = _tokenizer("}")['input_ids'][1]  # }<eos>
        self.eos_token = _tokenizer("<eos>")['input_ids'][1]  # <EOS>

        self.variable_token = _tokenizer('?')['input_ids'][1]  # ?uri
        self.keyword_start_token = _tokenizer('<')['input_ids'][1] # <subject...
        self.keyword_end_token = _tokenizer('>')['input_ids'][1] # ...subject>

        self.seperator_token = _tokenizer('.')['input_ids'][1] # ...<object> . ?uri...

        self.add_keywords(t_s_keywords, r_keywords)

    def add_keywords(self, t_s_keywords, r_keywords):
        if len(t_s_keywords) > 0:
            # print(t_s_keywords)
            t_s_unflattened = self.tokenizer(list(t_s_keywords))['input_ids']
            self.t_s_keywords = list(set(x for xs in t_s_unflattened for x in xs))
        if len(r_keywords) > 0:
            # print(r_keywords)
            r_unflattened = self.tokenizer(list(r_keywords))['input_ids']
            self.r_keywords = list(set(x for xs in r_unflattened for x in xs))

    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        # get the closest token of interest
        ids_list = input_ids.tolist()[0]
        # print(ids_list)

        if find_largest_index(ids_list, self.query_end_token) != -1:
            # end
            allowed_tokens = [self.eos_token]
            return only_allow_tokens(scores, allowed_tokens)

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
            allowed_tokens += [self.keyword_end_token]
            if triple_index == 0 or triple_index == 2:
                allowed_tokens += self.t_s_keywords
            else:
                allowed_tokens += self.r_keywords

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
        self.t_s_keywords = set()
        self.r_keywords = set()
        self.model = None
        self.tokenizer = None

        if existing_rdfs is not None and len(existing_rdfs) > 0:
            self.append_rdfs(existing_rdfs)

        self.setup_model()

    def append_rdfs(self, rdfs:[Triple]):
        self.t_s_keywords = self.t_s_keywords.union(set([rdf.object for rdf in rdfs] + [rdf.subject for rdf in rdfs]))
        self.r_keywords = self.r_keywords.union(set([rdf.predicate for rdf in rdfs]))

    def setup_model(self):
        # setup model helper function
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.path.join(os.path.dirname(__file__),"query_2b_200step"),
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
        self.t_s_keywords = set()
        self.r_keywords = set()

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

        if len(self.t_s_keywords) > 0 and len(self.r_keywords) > 0:
            outputs = self.model.generate(
                **inputs,

                # temperature = 0.9,
                # max_new_tokens = max(len(text) + 100, 700),

                logits_processor=LogitsProcessorList(
                    [SPARQLLogits(self.tokenizer, self.t_s_keywords, self.r_keywords)]),
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

        return sparql_string