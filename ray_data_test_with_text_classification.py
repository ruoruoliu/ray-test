import ray
import pandas as pd

class ClassificationModel:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification")

    def __call__(self, batch: pd.DataFrame):
        results = self.pipe(list(batch["text"]))
        result_df = pd.DataFrame(results)
        return pd.concat([batch, result_df], axis=1)

ray.init()
print(ray.cluster_resources())  # 会显示 {'CPU': 8.0, ...}

ds = ray.data.read_text("s3://anonymous@ray-example-data/sms_spam_collection_subset.txt")
ds = ds.map_batches(
    ClassificationModel,
    compute=ray.data.ActorPoolStrategy(size=1),
    batch_size=64,
    batch_format="pandas"
)
ds.show(limit=1)
