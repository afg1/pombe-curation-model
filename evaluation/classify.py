from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import argparse as ap
from tqdm import tqdm
label_lookup = {
    "LABEL_0": 0,
    "LABEL_1": 1
}

def main(args):
    data = load_dataset(args.dataset, split="test").shuffle(seed=42)

    pipe = pipeline(task="text-classification",model=args.model, device=args.device,max_length=512, padding='max_length', truncation=True)
    print(data)
    labels = data['label']
    abstracts = data['abstract']
    status = data['triage_status']
    pmids = data['pmid']
    
    false_positives = []

    for idx, out in tqdm(enumerate(pipe(KeyDataset(data, "abstract")))):
        pred = label_lookup[out['label']]
        if labels[idx] != pred:
            if pred > labels[idx]:
                false_positives.append((pmids[idx], abstracts[idx], status[idx]))

    print(len(false_positives))

    with open("false_positives.csv", 'w') as to_check:
        to_check.write("pmid,prediction,canto_label,triage_status, abstract\n")
        for fp in false_positives:
            to_check.write(f"{fp[0]},Curateable,NotCurated,{fp[2]},\"{fp[1]}\"\n")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("input_column")
    parser.add_argument("label_column")
    parser.add_argument("device")

    args = parser.parse_args()

    main(args)