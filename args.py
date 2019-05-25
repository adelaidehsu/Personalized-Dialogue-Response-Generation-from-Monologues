import argparse

def parse():
    parser = argparse.ArgumentParser(
        description='You have to set the parameters for seq2seq, \
        including both maximum likelihood estimation and \
        generative adversarial learning.')
    
    parser.add_argument("--file-head", type=str, default='None')
    parser.add_argument("--pre-model-dir", type=str, default='None')
    parser.add_argument("--pre-D-model-dir", type=str, default='None')
    parser.add_argument("--model-dir", type=str, default='results/gan')
    parser.add_argument("--pretrain-dir", type=str, default='results/pretrain')
    parser.add_argument("--gan-dir", type=str, default='results/gan')
    
    parser.add_argument("--glove-model", type=str, default='glove_model/corpus_op+fri')
    
    parser.add_argument("--data-dir", type=str, default='data/')
    parser.add_argument("--data-path", type=str, default='data/opensubtitles.txt')

    parser.add_argument("--feature-path", type=str, default='data/feature.txt')
    parser.add_argument("--feature-size", type=int, default=6)

    parser.add_argument("--train-path", type=str, default='data/friends.txt')
    
    parser.add_argument("--test-path", type=str, default='data/friends.txt')
    
    parser.add_argument("--steps-per-checkpoint", type=int, default=200)

    parser.add_argument("--lambda-one", type=float, default=0.5)
    parser.add_argument("--lambda-two", type=float, default=0.5)
    parser.add_argument("--lambda-dis", type=float, default=0.5)
    parser.add_argument("--baseline", type=float, default=1.5)
    parser.add_argument("--iteration", type=int, default=5000)
    parser.add_argument("--Dstep", type=int, default=5)
    parser.add_argument("--Gstep", type=int, default=1)

    # s2s: for encoder and decoder
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--grad-norm", type=float, default=5.0)
    parser.add_argument("--use-attn", type=bool, default=False)
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--output-sample", type=bool, default=False)
    parser.add_argument("--input_embed", type=bool, default=True)
    # s2s: training setting
    parser.add_argument("--buckets", type=str, default='[(10, 5)]')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--max-train-data-size", type=int, default=0) # 0: no limit

    # for value function
    parser.add_argument("--v-lr", type=float, default=1e-4)
    parser.add_argument("--v-lr-decay-factor", type=float, default=0.99)
    # gan
    parser.add_argument("--D-lr", type=float, default=1e-4)
    parser.add_argument("--D-lr-decay-factor", type=float, default=0.99)
    parser.add_argument("--gan-type", type=str, default='None')
    parser.add_argument("--gan-size", type=int)
    parser.add_argument("--gan-num-layers", type=int)
    parser.add_argument("--G-step", type=int)
    parser.add_argument("--D-step", type=int)
    parser.add_argument("--option", type=str, default='None')

    # test
    parser.add_argument("--test-type", type=str, default='accuracy')
    parser.add_argument("--test-critic", type=str, default='None')
    parser.add_argument("--test-data", type=str, default='None')
    parser.add_argument("--test-fout", type=str, default='None')
    
    return parser.parse_args()
