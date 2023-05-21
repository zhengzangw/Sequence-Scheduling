import argparse
import logging
import time

import tqdm

from . import utils
from .generate import Creator


def benchmark(
    model,
    data,
    max_length,
    group_size,
    batch_size,
    temperature,
    perception_strategy,
    vbs,
    fcr,
):
    logging.warning(
        f"Group size: {group_size}, Batch size: {batch_size}, Max length: {max_length}, Temperature: {temperature}"
    )
    logging.warning(f"Strategy: {perception_strategy}, VBS: {vbs}, FCR: {fcr}")

    num_tokens, num_finished, num_length = 0, 0, 0
    inputs_len, outputs_len, sentences_len = [], [], []
    T1 = utils.timeit()
    for i in tqdm.tqdm(range(0, len(data), group_size)):
        batch = data[i : i + group_size]
        inputs = batch["input"]
        ids = batch["id"]
        out = model(
            inputs,
            strategy="group",
            perception_strategy=perception_strategy,
            vbs=vbs,
            fcr=fcr,
            temperature=temperature,
            max_length=max_length,
            batch_size=batch_size,
            ids=ids,
        )
        for item in out:
            num_tokens += item["num_output_tokens"]
            num_finished += item["is_finished"]
            num_length += item["length"]
            inputs_len.append(item["num_input_tokens"])
            outputs_len.append(item["num_output_tokens"])
            sentences_len.append(item["num_total_tokens"])
    interval_s = utils.timeit(T1)

    throughput_sample = len(data) / interval_s
    average_length = num_length / len(data)
    effective_token_ratio = num_tokens / num_length
    failure_ratio = model.failed_number_batch_buffer / len(data)

    print(f"Strategy: {perception_strategy}, Total samples: {len(data)}")
    print(f"Group Size: {group_size}, Batch Size: {batch_size}")
    print(f"VBS: {vbs}, FCR: {fcr}")
    print(f"Time: {interval_s:.2f} s")
    print(f"Throughput: {throughput_sample:.2f} samples/s")
    print(f"Average length: {average_length:.2f} tokens")
    print(f"Effective token ratio: {effective_token_ratio*100:.2f} %")
    print(f"Failure ratio: {failure_ratio*100:.2f} %")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./data/alpaca-val-10k.json")
    parser.add_argument("--model", type=str, default="./ckpts/vicuna-7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-data", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=256)

    # sequence scheduling related
    parser.add_argument(
        "--strategy", choices=["seqsch", "vanilla", "gt", "po"], default="vanilla"
    )
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--vbs", action="store_true")
    parser.add_argument("--fcr", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data = utils.EvalDataset(args.data_path)
    if args.num_data is not None:
        data.sample(args.num_data, seed=args.seed)
    model = Creator(
        args.model,
        debug=args.debug,
        lora_path=args.lora_path,
    )

    # ===
    # benchmark
    # ===
    # --- group strategy ---
    result = benchmark(
        model,
        data,
        max_length=args.max_length,
        group_size=args.group_size,
        batch_size=args.batch_size,
        temperature=args.temperature,
        perception_strategy=args.strategy,
        vbs=args.vbs,
        fcr=args.fcr,
    )
