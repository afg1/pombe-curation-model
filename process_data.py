import polars as pl
from transformers import AutoTokenizer
from functools import lru_cache, partial

@lru_cache
def get_tokenizer(model_name="allenai/longformer-base-4096"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer

def count_tokens(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)

def convert_to_parquet(input_path, output_path):
    assert input_path.endswith("tsv")
    assert output_path.endswith("parquet")

    pl.scan_csv(input_path, separator='\t').sink_parquet(output_path)


def add_token_count(data_path, model_name="allenai/longformer-base-4096"):
    tokenizer = get_tokenizer(model_name)
    df = pl.read_parquet(data_path)
    df = df.filter(pl.col("abstract").is_not_null())
    _count_tokens = partial(count_tokens, tokenizer=tokenizer)
    df = df.with_columns(token_count=pl.col("abstract").map_elements(_count_tokens, return_dtype=pl.Int32))

    df.write_parquet(data_path)


def add_curateable_label(data_path, curateable_list=["Curatable", "Curatable, low priority"]):
    df = pl.read_parquet(data_path)
    df = df.with_columns(label=pl.col("triage_status").is_in(curateable_list).cast(pl.Int8))

    df.write_parquet(data_path)




if __name__ == "__main__":
    data_path = "canto_pombe_pubs.parquet"
    convert_to_parquet("canto_pombe_pubs.tsv", data_path)
    add_token_count(data_path)
    add_curateable_label(data_path)
    print(pl.read_parquet(data_path).describe())
