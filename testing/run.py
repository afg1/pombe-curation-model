import argparse as ap

import polars as pl
from transformers import pipeline

label_map = {
    "LABEL_0" : "NOT_CURATEABLE",
    "LABEL_1" : "CURATEABLE"
}

def main(args):
    data = pl.read_csv(args.input, separator='\t')
    abstracts = data.get_column("abstract").to_list()

    p = pipeline(task="text-classification",model=args.model, device=args.device,max_length=512, padding='max_length', truncation=True)

    classes = p(abstracts)
    decisions = [label_map[c['label']] for c in classes]
    curate_probability = [c['score'] if c['label'] == "LABEL_1" else 1.0-c['score'] for c in classes]

    out = data.with_columns([pl.Series(decisions).alias("prediction"), pl.Series(curate_probability).alias("curation_probability")])
    out.write_csv(args.output, separator='\t')




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("model")
    parser.add_argument("--device")

    args = parser.parse_args()
    main(args)
