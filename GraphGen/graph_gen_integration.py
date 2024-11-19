import math
import os

import nltk.data
import random
import torch
from transformers.generation import LogitsProcessor, LogitsProcessorList
from unsloth import FastLanguageModel
from dataclasses import dataclass
import re
from pyvis.network import Network

# modified from the notebook to allow seamless Q&A application. It added keyword preference compared to the notebook.

def disable_tokens(scores, banned_tokens: list[int]):
    for token in banned_tokens:
        scores[0][token] = -math.inf
    return scores

def reward_signed_tokens(scores, rewarded_tokens: list[int], reward: float):
    reward_add = reward - 1
    for token in rewarded_tokens:
        if scores[0][token] <= -math.inf:
            continue
        scores[0][token] += reward_add * abs(scores[0][token])
    return scores

def find_largest_index(lst: list, value) -> int:
    try:
        return len(lst) - 1 - lst[::-1].index(value)
    except:
        return -1  # not found

def get_probable_tokens_in_tree(tree: dict[int], existing_tokens: list[int]) -> list[int]:
    current_depth = tree
    for existing_token in existing_tokens:
        if existing_token in current_depth:
            current_depth = current_depth[existing_token]
        else:
            return []
    return list(current_depth.keys())


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    id: int
    document_id: int

    def __eq__(self, other: "Triple"):  # Python type hinting sucks
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object

    def __str__(self):
        return f"<unused0>{self.subject}<unused1>{self.predicate}<unused2>{self.object}"



# Enforces <T><R><S> Structure
class TRSLogits(LogitsProcessor):
    def __init__(self, _tokenizer):
        self.tokenizer = _tokenizer
        self.t_token = _tokenizer("<unused0>")['input_ids'][1]  # <T>
        self.r_token = _tokenizer("<unused1>")['input_ids'][1]  # <R>
        self.s_token = _tokenizer("<unused2>")['input_ids'][1]  # <S>

        self.eos_token = _tokenizer("<eos>")['input_ids'][1]  # <EOS>

        # other banned tokens
        self.pad_token = _tokenizer("<pad>")['input_ids'][1]  # <pad>
        self.sot_token = _tokenizer("<start_of_turn>")['input_ids'][1]  # <start_of_turn> for assistant
        self.eot_token = _tokenizer("<end_of_turn>")['input_ids'][1]  # <end_of_turn> for assistant

        # self.response_template_token = _tokenizer(response_template)['input_ids'][1]

    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        # get the closest token of interest
        ids_list = input_ids.tolist()[0]
        # print(ids_list)

        t_near_pos = find_largest_index(ids_list, self.t_token)
        r_near_pos = find_largest_index(ids_list, self.r_token)
        s_near_pos = find_largest_index(ids_list, self.s_token)

        near_pos = max(t_near_pos, r_near_pos, s_near_pos)

        if near_pos == len(ids_list) - 1:
            # Just generated start token
            # Enforce content generation (no special tokens!)
            banned_tokens = [self.t_token, self.r_token, self.s_token, self.eos_token]  # No special tokens allowed
            # print(f"#BAN at [{self.tokenizer.batch_decode(input_ids)[0][-10:]}]")
        # New special token enforce
        elif near_pos == t_near_pos:
            # T - setup
            banned_tokens = [self.t_token, self.s_token, self.eos_token]  # R allowed
            # print(f"#T at [{self.tokenizer.batch_decode(input_ids)[0][-10:]}]")
        elif near_pos == r_near_pos:
            # R - setup
            banned_tokens = [self.t_token, self.r_token, self.eos_token]  # S allowed
            # print(f"#R at [{self.tokenizer.batch_decode(input_ids)[0][-10:]}]")
        elif near_pos == s_near_pos:
            # S - setup
            banned_tokens = [self.r_token, self.s_token, ]  # T, end allowed
            # print(f"#S at [{self.tokenizer.batch_decode(input_ids)[0][-10:]}]")
        else:
            print("jjifdoasjddio not supposed to happen arghh!")
            raise Exception

        banned_tokens += [self.pad_token, self.sot_token, self.eot_token]

        # print(f"B {banned_tokens}")
        disabled_scores = disable_tokens(scores, banned_tokens)
        # print(input_ids)
        # print(f"{scores} -> {disabled_scores}")
        return disabled_scores



# Tree-based autocomplete thingy
class PreferKeywordsLogit(LogitsProcessor):

    t_s_tree = {} # Subject/Object keywords
    r_tree = {} # Predicate keywords

    def __init__(self, _tokenizer, reward, t_s_tree, r_tree): # incase cross-document context
        self.tokenizer = _tokenizer
        self.t_token = _tokenizer("<unused0>")['input_ids'][1]  # <T>
        self.r_token = _tokenizer("<unused1>")['input_ids'][1]  # <R>
        self.s_token = _tokenizer("<unused2>")['input_ids'][1]  # <S>

        self.reward = reward

        self.t_s_tree = t_s_tree
        self.r_tree = r_tree


    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        # get the closest token of interest
        ids_list = input_ids.tolist()[0]
        # print(ids_list)

        t_near_pos = find_largest_index(ids_list, self.t_token)
        r_near_pos = find_largest_index(ids_list, self.r_token)
        s_near_pos = find_largest_index(ids_list, self.s_token)

        near_pos = max(t_near_pos, r_near_pos, s_near_pos)


        # Example: <T>New_York<R>City_Of<S>United_States_of_
        #                                  ^^^^^^^^^^^^^^^^^
        generated = ids_list[(near_pos+1):] # All currently generated keywords


        if near_pos == t_near_pos or near_pos == s_near_pos:
            # T/S
            rewarded_tokens = get_probable_tokens_in_tree(self.t_s_tree, generated)
            # print(f"[G] {self.tokenizer.decode(generated)}...")
            # print(rewarded_tokens)
            # print(f"[C] {self.tokenizer.decode(rewarded_tokens)}")

        elif near_pos == r_near_pos:
            # R
            rewarded_tokens = get_probable_tokens_in_tree(self.r_tree, generated)

        else:
            print("jjifdoasjddio not supposed to happen arghh!")
            raise Exception

        reward_scores = reward_signed_tokens(scores, rewarded_tokens, self.reward)
        # print(input_ids)
        # print(f"{scores} -> {disabled_scores}")
        return reward_scores



def draw_graph(triples):
    net = Network(bgcolor="#222222", font_color="white", notebook=True, directed=True)

    # Parse rdf_strings

    def add_triples(rdf: Triple, color: str):
        net.add_node(rdf.subject, color=color)
        net.add_node(rdf.object, color=color)
        # if not any(edge['from'] == rdf.subject and edge['to'] == rdf.object and edge['title'] == rdf.predicate for edge in net.edges): # should be deprecated later
        net.add_edge(rdf.subject, rdf.object, title=rdf.predicate, color=color)

    import random

    r = lambda: random.randint(0, 255)
    net.toggle_physics(True)

    colors = {}

    for idx, triple in enumerate(triples):
        if triple.id in colors:
            color = colors[triple.id]
        else:
            color = '#%02X%02X%02X' % (r(), r(), r())
            colors[triple.id] = color

        add_triples(triple, color)
        # net.show(f"{idx}.html", notebook=False)
    net.show("graph2.html", notebook=True)



class GraphGen:
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Extract as much information from the Input as possible, and express the relationships in RDF Triples that compliment the existing RDF triples created from previous text. Do not use information from common sense. Expressed information must be confidently present in the text.
### Existing RDF triples:
{}
### Input:
{}
### Response:
<unused0>{}"""  # start with <T>


    def __init__(self, existing_rdfs, document_id = 0, max_chunk_size=400, keyword_reward = 1.05):
        # init tokenizer, models, etc.
        self.ts_tree = {} # Autocomplete for AI kinda
        self.r_tree = {}

        self.model = None
        self.tokenizer = None
        self.pkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        self.max_chunk_length = max_chunk_size
        self.keyword_reward = keyword_reward

        self.document_id = document_id

        self.setup_model()

        if existing_rdfs is not None and len(existing_rdfs) > 0:
            self.append_rdfs(existing_rdfs)

    def append_rdfs(self, rdfs:[Triple]):

        # Tokenize everything
        new_ts_tokens = [
            self.tokenizer(keyword)['input_ids'][1:] # First token is <bos>, removed
            for keyword in set(
                [rdf.object for rdf in rdfs] + [rdf.subject for rdf in rdfs]
            )]

        new_r_tokens = [
            self.tokenizer(keyword)['input_ids'][1:] # First token is <bos>, removed
            for keyword in set(
                [rdf.predicate for rdf in rdfs]
            )]

        # ts map
        ts_end_tokens = self.tokenizer("<unused1><unused0><eos>")['input_ids'][1:]

        for tokens in new_ts_tokens:
            tokens += ts_end_tokens
            # print(tokens)

            current_depth = self.ts_tree

            for token in tokens:
                if token not in current_depth: # token doesn't exist, create
                    current_depth[token] = {}

                current_depth = current_depth[token]

        # r map
        r_end_tokens = self.tokenizer("<unused2>")['input_ids'][1:]

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
            model_name=os.path.join(os.path.dirname(__file__),"graph_9b_full"),
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

    def reset_model(self, existing_rdfs, document_id):
        self.ts_tree = {}
        self.r_tree = {}
        self.document_id = document_id

        if existing_rdfs is not None and len(existing_rdfs) > 0:
            self.append_rdfs(existing_rdfs)


    def generate(self, input_text):
        input_text = self.preprocess_document(input_text)
        merged_sentences = self.split_chunks(input_text)

        # Triples gen
        triples = []
        for idx, m_sentence in enumerate(merged_sentences):
            print(f"PROCESSING ({idx + 1}/{len(merged_sentences)})")
            new_triples = self.generate_rdf(triples, m_sentence, idx)

            self.append_rdfs(new_triples)
            triples += new_triples
            # print(triples)

        return triples

    def preprocess_document(self, input_text:str):
        return re.sub(r'\n\s*\n', '\n\n', input_text)

    def split_chunks(self, input_text:str):
        # Chunk splitting
        sentences = self.pkt_tokenizer.tokenize(input_text)
        merged_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            if len(sentence) <= 0:
                continue
            if i >= 1 and len(merged_sentences[-1]) + len(sentence) <= self.max_chunk_length:
                merged_sentences[-1] += " " + sentence
            else:
                if i >= 1:
                    merged_sentences.append(sentences[i - 1])
                merged_sentences.append(sentence)
        return merged_sentences


    def generate_rdf(self, context: list[Triple], text: str, _id: int) -> list[Triple]:
        # context pre-processing
        if len(context) == 0:
            context_str = "None"
        else:
            context_str = " ".join(str(con) for con in random.sample(context, min(len(context), 15)))

        inputs = self.tokenizer(
            [
                self.prompt.format(
                    context_str,
                    text,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt"
        ).to("cuda")

        print(f"Phase 1 text chunk \"{text[:50]}...{text[-20:]}\"")
        print(f"Phase 1 context \"{context_str[:30]}...{context_str[-20:]}\"")
        # print(self.prompt.format(
        #             context_str,
        #             text,  # input
        #             "",  # output - leave this blank for generation!
        #         ))
        print("---")

        print("Phase 1 model start")
        outputs = self.model.generate(
            **inputs,

            # temperature = 0.9,
            # max_new_tokens = max(len(text) + 100, 700),

            logits_processor=LogitsProcessorList([TRSLogits(self.tokenizer), PreferKeywordsLogit(self.tokenizer, self.keyword_reward, self.ts_tree, self.r_tree)]),
            # num_beams = 3,
            # early_stopping = True,
            repetition_penalty=1.05,
            renormalize_logits = True,

            use_cache=True,
            # Use cache = false is broken haha, but beam search is broken when not using cache hahahah ;-;
        )

        response = self.tokenizer.batch_decode(outputs)

        response = response[0].replace('\n', '')
        rdf_string = response.split("### Response:")[1]

        print(f"Phase 2 model generation done")
        # print(rdf_string)
        # convert rdf string to list
        rdfs = []
        rdf_string = rdf_string.removeprefix("<bos>").removesuffix("<eos>")
        # print(rdf_string)
        for _triple in rdf_string.split("<unused0>"):
            print(_triple)
            try:
                if _triple == "":
                    continue
                split = re.split("<unused1>|<unused2>", _triple)
                subject = split[0]
                predicate = split[1]
                _object = split[2]

                new_triple = Triple(subject, predicate, _object, _id, self.document_id)

                if not (any(con == new_triple for con in context) or any(con == new_triple for con in rdfs)):
                    rdfs.append(new_triple)

            except Exception as e:
                print(f"NON-STANDARD TRIPLE {_triple} ({e})")
                continue
        print("Phase 3 parsing done")
        print("chunk DONE")

        return rdfs


