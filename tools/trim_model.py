import argparse
import torch


KEYS = [
    "roi_heads.box.predictor.cls_score.weight",
    "roi_heads.box.predictor.cls_score.bias",
    "roi_heads.box.predictor.bbox_pred.weight",
    "roi_heads.box.predictor.bbox_pred.bias",
    "roi_heads.mask.predictor.mask_fcn_logits.weight",
    "roi_heads.mask.predictor.mask_fcn_logits.bias"
]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True
    )
    return parser


def main(args):
    sd = torch.load(args.input_file)
    for k in KEYS:
        print(k)
        del sd["model"][k]
    torch.save(sd, args.output_file)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
