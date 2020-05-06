# -*- coding: utf-8 -*-

import argparse
import os
import torch
from collections import OrderedDict

def get_item_values(model_dict, enc_dec, kitem, max_span, nheads=4, span_ramp=2):
    item_values = OrderedDict()
    for key, value in model_dict.items():
        if enc_dec in key and kitem in key:
            nlayer = int(key.split('.')[2])
            if kitem in ['span_size']:
                item_values[nlayer] = (max_span * value + span_ramp).view(nheads).tolist()
            elif kitem in ['span_ratio']:
                item_values[nlayer] = value.view(nheads).tolist()
            else:
                raise NotImplementedError

    out_strs = [f'| head{i+1:d} |' for i in range(nheads)]
    out_strs.insert(0, '| layer |')
    out_strs.insert(nheads+1, '| average |')
    for k in item_values.keys():
        out_strs[0] += f' {int(k+1):d} |'
        for i in range(1, nheads+1):
            if kitem in ['span_size']:
                out_strs[i] += f' {int(item_values[k][i-1]):d} |'
            elif kitem in ['span_ratio']:
                out_strs[i] += f' {item_values[k][i-1]:.2f} |'
            else:
                raise NotImplementedError
        out_strs[nheads+1] += f' {float(sum(item_values[k])/nheads):.2f} |'

    for out_str in out_strs:
        print(out_str)


def main(args):
    assert os.path.exists(args.model)
    model_dict = torch.load(args.model, map_location=lambda storage, loc: storage)
    if 'model' in model_dict.keys():
        model_dict = model_dict['model']

    for x in args.print_enc_dec:
        get_item_values(model_dict, x, 'span_size', args.max_span, 4, args.span_ramp)
        get_item_values(model_dict, x, 'span_ratio', args.max_span, 4, args.span_ramp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="span size printer.")
    parser.add_argument('model', default=None, type=str,
                        help="The model to be printed.")
    parser.add_argument('--max-span', default=50, type=int,
                        help="The max span sizes.")
    parser.add_argument('--span-ramp', default=2, type=int,
                        help="The span ramp size.")
    parser.add_argument('--print-enc-dec', default=['enc'], nargs='+',
                        help="print encoder decoder.")
    args = parser.parse_args()
    main(args)
