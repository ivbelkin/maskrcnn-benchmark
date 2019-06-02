import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="~/.torch/models/_detectron_36761843_12_2017_baselines_e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="./e2e_mask_rcnn_X-101-32x8d-FPN_1x_no_last_layer.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/rtsd/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

newdict['model'] = removekey(_d['model'],
                             ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))
