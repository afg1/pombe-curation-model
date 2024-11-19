import argparse as ap
import os
from datasets import load_dataset
import evaluate
from evaluate import evaluator
from transformers import pipeline

import huggingface_hub
import runpod

# runpod.api_key = os.getenv("RUNPOD_API_KEY")


# hf_key = os.getenv("HF_KEY")
# huggingface_hub.login(token=hf_key)

def main(args):
    data = load_dataset(args.dataset, split="test").shuffle(seed=42)

    task_evaluator = evaluator("text-classification")

    pipe = pipeline(task="text-classification",model=args.model, device=args.device,max_length=512, padding='max_length', truncation=True)

    eval_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=data,
        metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        input_column=args.input_column,
        label_column=args.label_column,
        label_mapping={"LABEL_0": 0, "LABEL_1": 1},
         
    )

    print(eval_results)

    for name, value in eval_results.items():
        evaluate.push_to_hub(
            model_id=args.model,
            metric_value=value,
            metric_type=name,
            metric_name=name.title(),
            dataset_name=args.dataset,
            dataset_type="text-classification",
            dataset_split="test",
            task_type="text-classification",
            task_name="Text Classification",
            overwrite=True
        )


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("input_column")
    parser.add_argument("label_column")
    parser.add_argument("device")
    parser.add_argument("--stop", default=False, action='store_true')

    args = parser.parse_args()

    # this_pod = runpod.get_pods()[0]
    main(args)

    # if args.stop:
    #     runpod.stop_pod(this_pod['id'])