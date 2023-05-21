import argparse

from . import utils

QUERY_PROMPT = "\nDon't output the response for the above instruction. Instead, you need to predict the number of tokens in your response. Output one number only."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/alpaca-train-10k.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data = utils.jload(args.data_path)
    len_info_path = args.data_path.replace(".json", "-length.json")
    len_info = utils.jload(len_info_path)

    for i in range(len(data)):
        assert data[i]["id"] == len_info[i]["id"]

        # user
        data[i]["conversations"][0]["value"] += QUERY_PROMPT
        # response
        temp = [0.0, 0.3, 0.5, 0.7]
        lens = [len_info[i]["L_t{}".format(t)] for t in temp]
        maxlen = max(lens)
        len_to_predict = utils.buckit(maxlen)
        data[i]["conversations"][1]["value"] = "{}".format(len_to_predict)

    output_path = args.data_path.replace(".json", "-instruct.json")
    utils.jdump(data, output_path)
