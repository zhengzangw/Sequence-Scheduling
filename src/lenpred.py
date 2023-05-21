import argparse

import tqdm

from . import utils
from .generate import Creator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/alpaca-train-10k.json")
    parser.add_argument("--model", type=str, default="./ckpts/vicuna-7b")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-data", type=int, default=None)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    utils.set_seed(args.seed)

    data = utils.EvalDataset(args.data_path)
    if args.num_data is not None:
        data = data.sample(args.num_data)

    # inference to get the length
    model = Creator(args.model)
    out_json = []
    batch_size = args.bs
    for i in tqdm.tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        kwargs = dict(strategy="stream", max_length=args.max_length)
        out = []
        temp = [0.0, 0.3, 0.5, 0.7]
        for t in temp:
            kwargs["temperature"] = t
            out.append(model(batch["input"], **kwargs))
        for j in range(len(batch["input"])):
            result = dict(id=batch["id"][j])
            result["L_gt"] = len(model.tokenizer(batch["output"][j]).input_ids)
            for k, t in enumerate(temp):
                result[f"L_t{t}"] = out[k][j]["num_output_tokens"]
            out_json.append(result)

        out_json.append(result)

    # save to json
    output_json_name = args.data_path.replace(".json", f"-length.json")
    utils.jdump(out_json, output_json_name)
