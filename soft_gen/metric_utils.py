from typing import List
import itertools
def calc_diversity(hypotheses: List[List[str]]):
    unigram_list = list(itertools.chain(*hypotheses))
    total_num_unigram = len(unigram_list)
    unique_num_unigram = len(set(unigram_list))
    bigram_list = []
    for hyp in hypotheses:
        hyp_bigram_list = list(zip(hyp[:-1], hyp[1:]))
        bigram_list += hyp_bigram_list
    total_num_bigram = len(bigram_list)
    unique_num_bigram = len(set(bigram_list))
    dist_1 = unique_num_unigram / total_num_unigram
    dist_2 = unique_num_bigram / total_num_bigram

    return dist_1, dist_2