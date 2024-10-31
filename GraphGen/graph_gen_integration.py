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

def disable_tokens(scores, banned_tokens):
    for token in banned_tokens:
        scores[0][token] = -math.inf
    return scores

def reward_signed_tokens(scores, rewarded_tokens, reward):
    reward_add = reward - 1
    for token in rewarded_tokens:
        scores[0][token] += reward_add * abs(scores[0][token])
    return scores

def find_largest_index(lst: list, value):
    try:
        return len(lst) - 1 - lst[::-1].index(value)
    except:
        return -1  # not found


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

        disabled_scores = disable_tokens(scores, banned_tokens)
        # print(input_ids)
        # print(f"{scores} -> {disabled_scores}")
        return disabled_scores


class PreferKeywordsLogit(LogitsProcessor):
    t_s_keywords = [] # Subject/Object keywords
    r_keywords = [] # Predicate keywords

    def __init__(self, _tokenizer, reward, t_s_keywords, r_keywords): # incase cross-document context
        self.tokenizer = _tokenizer
        self.t_token = _tokenizer("<unused0>")['input_ids'][1]  # <T>
        self.r_token = _tokenizer("<unused1>")['input_ids'][1]  # <R>
        self.s_token = _tokenizer("<unused2>")['input_ids'][1]  # <S>

        self.reward = reward

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

        t_near_pos = find_largest_index(ids_list, self.t_token)
        r_near_pos = find_largest_index(ids_list, self.r_token)
        s_near_pos = find_largest_index(ids_list, self.s_token)

        near_pos = max(t_near_pos, r_near_pos, s_near_pos)


        if near_pos == t_near_pos or near_pos == s_near_pos:
            # T/S - setup
            rewarded_tokens = self.t_s_keywords
        elif near_pos == r_near_pos:
            # R - setup
            rewarded_tokens = []
            # rewarded_tokens = self.r_keywords
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
Extract the most confident information in the sentence below as much as possible, and express the relationships in RDF Triples that complement the existing RDF triples. Do not use information from common sense.
### Existing RDF triples:
{}
### Input:
{}
### Response:
<unused0>{}"""  # start with <T>


    def __init__(self, existing_rdfs, document_id = 0, max_chunk_size=400, keyword_reward = 1.05):
        # init tokenizer, models, etc.
        self.t_s_keywords = set()
        self.r_keywords = set()
        self.model = None
        self.tokenizer = None
        self.pkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        self.max_chunk_length = max_chunk_size
        self.keyword_reward = keyword_reward

        self.document_id = document_id

        if existing_rdfs is not None and len(existing_rdfs) > 0:
            self.append_rdfs(existing_rdfs)

        self.setup_model()

    def append_rdfs(self, rdfs:[Triple]):
        self.t_s_keywords = self.t_s_keywords.union(set([rdf.object for rdf in rdfs] + [rdf.subject for rdf in rdfs]))
        self.r_keywords = self.r_keywords.union(set([rdf.predicate for rdf in rdfs]))

    def setup_model(self):
        # setup model helper function
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.path.join(os.path.dirname(__file__),"graph_2b_1000step"),
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
        self.t_s_keywords = set()
        self.r_keywords = set()
        self.document_id = document_id

        if existing_rdfs is not None and len(existing_rdfs) > 0:
            self.append_rdfs(existing_rdfs)


    def generate(self, input_text):
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

    def split_chunks(self, input_text):
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
            ], return_tensors="pt").to("cuda")
        print(f"Processing \"{text[:50]}...{text[-10:]}\"")
        print(self.prompt.format(
                    context_str,
                    text,  # input
                    "",  # output - leave this blank for generation!
                ))
        print("---")

        outputs = self.model.generate(
            **inputs,

            # temperature = 0.9,
            # max_new_tokens = max(len(text) + 100, 700),

            logits_processor=LogitsProcessorList([TRSLogits(self.tokenizer), PreferKeywordsLogit(self.tokenizer, self.keyword_reward, self.t_s_keywords, self.r_keywords)]),
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

        print(f"Done!")
        # print(rdf_string)
        # convert rdf string to list
        rdfs = []
        rdf_string = rdf_string.removeprefix("<bos>").removesuffix("<eos>")
        print(rdf_string)
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
        print("DONE")

        return rdfs
