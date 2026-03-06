import json
from pydantic import BaseModel
import ray


class AnswerWithExplain(BaseModel):
    problem: str
    answer: int
    explain: str


class StructuredLLM:
    def __init__(self):
        import outlines
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "unsloth/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("mps")
        wrapped = outlines.from_transformers(model, tokenizer)
        self.generator = outlines.Generator(wrapped, outlines.json_schema(AnswerWithExplain))

    def __call__(self, batch: dict):
        prompts = [
            f"You are a math teacher. Give the answer to the equation and explain it. "
            f"Output the problem, answer and explanation in JSON.\n\n"
            f"3 * {row_id} + 5 = ?"
            for row_id in batch["id"]
        ]
        results = [self.generator(p) for p in prompts]
        return {"resp": results}


ds = ray.data.range(4)
ds = ds.map_batches(
    StructuredLLM,
    batch_size=4,
    concurrency=1,
    num_gpus=0,
)
ds = ds.materialize()

for out in ds.take_all():
    print(out["resp"])
