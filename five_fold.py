from sklearn.model_selection import KFold, train_test_split
import polars as pl
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate 
import datasets
import numpy as np
import wandb
import huggingface_hub
import datasets
import os
import runpod

runpod.api_key = os.getenv("RUNPOD_API_KEY")

## Login to stuff
wandb_key = os.getenv("WANDB_KEY")
wandb.login(key=wandb_key)

hf_key = os.getenv("HF_KEY")
huggingface_hub.login(token=hf_key)


def generate_train_test_splits(data_path, train_fraction, test_path, train_path):
    """
    Split the whole dataset once to get a train set and what will be a held out 
    test set. We will then do 5-fold on the train set, and keep the remaining
    test set to test the best model on.
    """

    df = pl.read_parquet(data_path)

    train, test = train_test_split(df, train_size=train_fraction)

    train.write_parquet(train_path)
    ## Important to note the test set is left untouched, not used in the five fold at all
    test.write_parquet(test_path)

    dataset = datasets.load_dataset("parquet", data_files={"train": train_path, "test":test_path})
    dataset.push_to_hub("afg1/pombe-canto-data", private=True)





def train_five_fold(train_path, model_name, max_length=-1, hub_id=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if max_length < 0:
        ## -2 is insurance against CLS and EOS tokens
        max_length = model.config.max_position_embeddings - 2
    def tokenize_function(examples):
        return tokenizer(str(examples["abstract"]), padding="max_length", max_length=max_length)
    
    kf = KFold(5)
    train_data = datasets.load_dataset("parquet", data_files={"train":train_path})
    tokenized_datasets = train_data.map(tokenize_function)

    splits = kf.split(np.zeros(tokenized_datasets["train"].num_rows))

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    

    for k, (train_idxs, val_idxs) in enumerate(splits):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        fold_dataset = tokenized_datasets.copy()
        fold_dataset["test"] = tokenized_datasets["train"].select(val_idxs)
        fold_dataset["train"] = tokenized_datasets["train"].select(train_idxs)

        wandb.init(name=f"pombe_curation_fold{k}")

        # Set up the training arguments. These worked on a P40 in colab, used about 10GB
        training_args = TrainingArguments(per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=1,
            fp16=True,
            output_dir=f"pombe_curation_fold{k}",
            eval_strategy="steps",
            eval_steps=200,
            logging_steps=100,
            run_name=f"pombe_curation_fold{k}",
            report_to='wandb',
            push_to_hub=True,
            hub_model_id=hub_id,
            gradient_checkpointing=True,
            hub_private_repo=True,
            num_train_epochs=1.0)

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=fold_dataset['train'],
            eval_dataset=fold_dataset['test'],
            compute_metrics=compute_metrics,
            )

        trainer.train()





if __name__ == "__main__":
    this_pod = runpod.get_pods()[0]
    data_path = "canto_pombe_pubs.parquet"
    
    base_model = os.getenv("BASE_MODEL", "allenai/longformer-base-4096")
    max_length = int(os.getenv("MAX_LENGTH", "-1"))
    tt_split_frac = float(os.getenv("TRAIN_TEST_SPLIT_FRAC", "0.8"))
    hub_id = os.getenv("HF_MODEL_OUTPUT_ID", "afg1/pombe_curation_model")
    try:
        generate_train_test_splits(data_path, tt_split_frac, "test_data.parquet", "train_data.parquet")
        train_five_fold("train_data.parquet", base_model, max_length=max_length, hub_id=hub_id)
    except Exception as e:
        print("Caught an exception")
        print(e)

    runpod.stop_pod(this_pod.id)