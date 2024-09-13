import os

from GraphGen.graph_gen_integration import Triple
from unsloth import FastLanguageModel
import torch

class AnswerGen:
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Based on the context of the question, convert the query result for the question into a natural sounding contextual answer.
### User Question:
{}
### Query Result:
{}
### Natural Answer:
{}"""


    def __init__(self):
        # init tokenizer, models, etc.
        self.model = None
        self.tokenizer = None

        # self.max_chunk_length = max_chunk_size
        # self.keyword_reward = keyword_reward

        self.setup_model()

    def setup_model(self):
        # setup model helper function
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.path.join(os.path.dirname(__file__),"answer_2b_demo"),
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
        pass


    def generate(self, question, query_result):
        inputs = self.tokenizer(
            [
                self.prompt.format(
                    question,
                    query_result,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,

            # temperature = 0.9,
            # max_new_tokens = max(len(text) + 100, 700),
            # num_beams = 3,
            # early_stopping = True,

            use_cache=True,
            # Use cache = false is broken haha, but beam search is broken when not using cache hahahah ;-;
        )
        response = self.tokenizer.batch_decode(outputs)
        answer_text = response[0].split("### Natural Answer:\n")[1]

        return answer_text
