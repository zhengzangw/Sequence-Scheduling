import copy
import io
import json
import logging
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
from fastchat import conversation


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def timeit(T0=None):
    torch.cuda.synchronize()
    T1 = time.time()
    if T0 is not None:
        T1 = T1 - T0
    return T1


def describe(input_len, name=""):
    print(f"Statistics of {name}:")
    print(f"\tMean: {np.mean(input_len):.2f}, Std: {np.std(input_len):.2f}")
    # print(f"\tquartiles: {np.quantile(input_len, [0, 0.25, 0.5, 0.75, 1])}")


def buckit(x, cell=50):
    x = int(x)
    x = (x // cell + 1) * cell if x % cell != 0 else x
    return x


def extract_all_numbers(string):
    all_number = [int(s) for s in re.findall(r"\d+", string)]
    if len(all_number) == 1:
        return all_number[0]
    elif len(all_number) == 0:
        return 0
    else:
        return (all_number[0] + all_number[1]) / 2


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jsort(obj, key, integer=False):
    assert isinstance(obj, list)
    if integer:
        return sorted(obj, key=lambda x: int(x[key]))
    return sorted(obj, key=lambda x: x[key])


# ===
# Dataset
# ===


class EvalDataset:
    def __init__(self, data_path: str, conv_template: str = "vicuna_v1.1"):
        super().__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict_ori = list_data_dict
        self.list_data_dict = list_data_dict
        self.list_data_dict_remain = None
        self.conv = conversation.conv_templates[conv_template].copy()

    def sample(self, num=100, seed=1):
        np.random.seed(seed)
        rand_order = np.random.permutation(len(self.list_data_dict_ori))
        self.list_data_dict = [self.list_data_dict_ori[i] for i in rand_order[:num]]
        self.list_data_dict_remain = [
            self.list_data_dict_ori[i] for i in rand_order[num:]
        ]
        logging.warning(f"Sampling {num} data points")
        return self

    def reverse(self):
        self.list_data_dict_remain, self.list_data_dict = (
            self.list_data_dict,
            self.list_data_dict_remain,
        )

    def dump(self, path):
        jdump(self.list_data_dict, path)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        data_json = copy.deepcopy(self.list_data_dict[idx])
        if isinstance(idx, int):
            data_ = [data_json["conversations"]]
            ids = [data_json["id"]]
        else:
            data_ = [d["conversations"] for d in data_json]
            ids = [d["id"] for d in data_json]

        inputs = []
        outputs = []
        for d in data_:
            conv = self.conv.copy()
            conv.append_message(conv.roles[0], d[0]["value"])
            conv.append_message(conv.roles[1], None)
            inputs.append(conv.get_prompt())
            outputs.append(d[1]["value"])

        if isinstance(idx, int):
            return dict(input=inputs[0], output=outputs[0], id=ids[0])
        else:
            return dict(input=inputs, output=outputs, id=ids)


# ===
# Predictor
# ===


PIA_PROMPT = "\nBefore responding to the above instruction, you have to predict the length of your response. Print the estimated number of tokens in your response in the first line. Then change to a new line to respond to the instruction."
QUERY_PROMPT = "\nDon't output the response for the above instruction. Instead, you need to predict the number of tokens in your response. Output one number only."


class Predictor:
    def __init__(
        self,
        model,
        tokenizer,
        length_predict_strategy="vanilla",
        max_new_tokens=512,
        length_predictor=None,
    ) -> None:
        assert length_predict_strategy in ["seqsch", "vanilla", "gt", "po"]

        self.length_predict_strategy = length_predict_strategy
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.model = model
        self.length_predictor = length_predictor

        # length predictor
        if length_predict_strategy == "gt":
            gt_json = jload("./data/alpaca-val-10k-length.json")
            print("Loading ground truth length predictor")
            temp = [0.0, 0.3, 0.5, 0.7]
            gt_dict = {d["id"]: max([d[f"L_t{t}"] for t in temp]) for d in gt_json}
            self.gt_dict = gt_dict

    def predict_length(self, inputs, ids=None):
        if self.length_predict_strategy == "vanilla":
            return [self.max_new_tokens] * len(inputs)
        elif self.length_predict_strategy == "gt":
            length = [self.gt_dict[id] for id in ids]
            return length
        elif self.length_predict_strategy in ["seqsch", "po"]:
            # FIXME: hard-coded prompt
            if self.length_predict_strategy == "seqsch":
                inputs = [p[:-11] + QUERY_PROMPT + p[-11:] for p in inputs]
            else:
                inputs = [p[:-11] + PIA_PROMPT + p[-11:] for p in inputs]

            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
            l_prompt = inputs["input_ids"].shape[1]
            input_ids = inputs["input_ids"].cuda()
            attn_mask = inputs["attention_mask"].cuda()

            if self.length_predict_strategy == "seqsch":
                self.length_predictor.enable_adapter_layers()
            output_ids = self.length_predictor.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=8,
                do_sample=False,
            )
            if self.length_predict_strategy == "seqsch":
                self.length_predictor.disable_adapter_layers()

            outputs = [
                self.tokenizer.decode(x[l_prompt:], skip_special_tokens=True)
                for x in output_ids
            ]

            if self.length_predict_strategy == "seqsch":
                ret = [int(s.strip()) for s in outputs]
            else:
                ret = [extract_all_numbers(x) for x in outputs]
            return ret


# ===
# Scheduler
# ===

len_bs_dict = {
    50: 256,
    100: 128,
    150: 64,
    200: 32,
    250: 16,
    300: 16,
}


def len_bs_dict_fn(x):
    if x in len_bs_dict:
        bs = len_bs_dict[x]
    else:
        bs = 16
    return bs


def schedule(
    lengths,
    mini_batch_size=1,
    vbs=False,
):
    # sort ids by length
    lengths_with_id = [(i, l) for i, l in enumerate(lengths)]
    sorted_lengths_with_id = sorted(lengths_with_id, key=lambda x: x[1], reverse=False)

    # batchify
    batches = []
    if not vbs:
        for i in range(0, len(lengths), mini_batch_size):
            batch = sorted_lengths_with_id[i : i + mini_batch_size]
            batch_ids = [x[0] for x in batch]
            max_len = max([x[1] for x in batch])
            batches.append((batch_ids, max_len))
    else:
        # group by length
        ids_len_dict = defaultdict(list)
        for i, l in sorted_lengths_with_id:
            ids_len_dict[buckit(l)].append(i)
        # batchify
        max_l = max(lengths)
        for l, ids in ids_len_dict.items():
            bs = len_bs_dict_fn(l)
            for i in range(0, len(ids), bs):
                batch = ids[i : i + bs]
                if l < max_l and len(batch) < max(len_bs_dict_fn(l) // 2, 16):
                    l_ = l + 50
                    while l_ not in ids_len_dict:
                        l_ += 50
                    ids_len_dict[l_].extend(batch)
                    break
                batches.append((batch, l))
    return batches
