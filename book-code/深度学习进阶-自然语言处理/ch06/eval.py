# coding: utf-8
import sys
sys.path.append('..')

import os
root_path = os.path.dirname(os.path.abspath(__file__))
previous_path = os.path.dirname(root_path)
sys.path.append(root_path)
sys.path.append(previous_path)

from rnnlm import Rnnlm
from better_rnnlm import BetterRnnlm
from dataset import ptb
from common.util import eval_perplexity


if __name__ == '__main__':
    model = Rnnlm()
    #model = BetterRnnlm()

    # 读入训练完的超参数
    model.load_params()

    corpus, _, _ = ptb.load_data('test')

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus)
    print('test perplexity: ', ppl_test)
