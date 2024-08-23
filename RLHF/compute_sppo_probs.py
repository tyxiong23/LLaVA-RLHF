# select sppo data and compute scores
import json
import numpy as np
import os
from matplotlib import pyplot as plt

from argparse import ArgumentParser

def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def get_sppo_score(scores, idx, a=10.0):
    scores_relative = 1 / (1 + np.exp((scores - scores[idx]) * a))
    return np.mean(scores_relative)



def main(args):
    dataset = load_jsonl(args.in_file)
    print("len dataset", len(dataset))
    out_dir = os.path.dirname(args.in_file)

    A = args.a

    sppo_scores = []; new_dataset = []
    for data in dataset:
        pred = data['pred']
        pred_rewards = np.array(data['pred_rewards'])

        max_id = np.argmax(pred_rewards)
        min_id = np.argmin(pred_rewards)
        # print(max_id, min_id, pred_rewards)

        data['chosen'] = pred[max_id]
        data['rejected'] = pred[min_id]
        # assert max_id != min_id
        assert pred_rewards[max_id] >= pred_rewards[min_id]

        # if pred_rewards[max_id] == pred_rewards[min_id]:
        #     continue
        sppo_chosen = get_sppo_score(pred_rewards, max_id, a=A)
        sppo_reject = get_sppo_score(pred_rewards, min_id, a=A)
        # print("new", sppo_chosen, sppo_reject)
        data['chosen_score_sppo'] = sppo_chosen
        data['rejected_score_sppo'] = sppo_reject
        data['chosen_score'] = pred_rewards[max_id]
        data['rejected_score'] = pred_rewards[min_id]

        sppo_scores.append([sppo_chosen, sppo_reject])

        new_dataset.append(data)

    sppo_scores = np.array(sppo_scores)
    plt.hist(sppo_scores, bins=[i*0.05 for i in range(21)])
    plt.legend(["chosen", 'rejected'])
    plt.title('SPPO_scores distribution')
    plt.savefig(os.path.join(out_dir, 'sppo_scores.png'))

    with open(os.path.join(out_dir, f"chosen_sppo_k5_a{A}.json"), 'w') as outf:
        print("output len", len(new_dataset))
        json.dump(new_dataset, outf, indent=4, separators=(',', ': '))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--a", type=int, default=10)

    args = parser.parse_args()
    main(args)
    
    