import math
import re
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.matutils import corpus2csc


def compute_purity(df_purity: pd.DataFrame):
    """df columns 包括 topic_labels & college_labels"""
    topic_group = df_purity.groupby("topic_labels")["college_labels"]
    max_amount_of_group = []
    result_purity = []
    for group, df in topic_group:
        distribution = df.value_counts()
        max_amount_of_group.append(distribution.max())
        result_purity.append(
            [
                group,
                distribution.idxmax(),
                (distribution.max() / distribution.sum()),
            ]
        )

    result_purity = pd.DataFrame(
        result_purity, columns=["topic", "topic_label", "purity"]
    )

    purity = sum(max_amount_of_group) / len(df_purity)
    return purity, result_purity


ranges = [
    {"from": ord("\u3300"), "to": ord("\u33ff")},  # compatibility ideographs
    {"from": ord("\ufe30"), "to": ord("\ufe4f")},  # compatibility ideographs
    {"from": ord("\uf900"), "to": ord("\ufaff")},  # compatibility ideographs
    {"from": ord("\U0002F800"), "to": ord("\U0002fa1f")},  # compatibility ideographs
    {"from": ord("\u3040"), "to": ord("\u309f")},  # Japanese Hiragana
    {"from": ord("\u30a0"), "to": ord("\u30ff")},  # Japanese Katakana
    {"from": ord("\u2e80"), "to": ord("\u2eff")},  # cjk radicals supplement
    {"from": ord("\u4e00"), "to": ord("\u9fff")},
    {"from": ord("\u3400"), "to": ord("\u4dbf")},
    {"from": ord("\U00020000"), "to": ord("\U0002a6df")},
    {"from": ord("\U0002a700"), "to": ord("\U0002b73f")},
    {"from": ord("\U0002b740"), "to": ord("\U0002b81f")},
    {"from": ord("\U0002b820"), "to": ord("\U0002ceaf")},  # included as of Unicode 8.0
]


def is_cjk(char):
    """
    reference: https://stackoverflow.com/a/30070664/12872141
    """
    return any([range["from"] <= ord(char) <= range["to"] for range in ranges])


## check language
def check_language(word):
    # 以正規表示法判斷語言
    try:
        # 可以解碼為 ascii 的為英文單字
        word.encode(encoding="utf-8").decode("ascii")
        return "lang1"
    except UnicodeDecodeError:
        # 為什麼還有一正規判斷?
        if any(map(is_cjk, word)):
            return "lang2"
        else:
            return None


def split_topk_topic_word(topic_word_matrix, feature, topk=10):
    feature = np.array(feature)
    lang_labels = np.array(list(map(check_language, feature)))
    assert lang_labels.shape[0] == len(feature)

    feature_en_idx = np.where(lang_labels == "lang1")[0]
    feature_zh_idx = np.where(lang_labels == "lang2")[0]
    en_topic = []
    cn_topic = []
    for i in range(topic_word_matrix.shape[0]):
        # en
        en_topk_idx = feature_en_idx[
            topic_word_matrix[i, feature_en_idx].argsort()[-topk:]
        ]
        en_topk_words = feature[en_topk_idx][::-1]
        en_topic.append(en_topk_words.tolist())
        # cn
        cn_topk_idx = feature_zh_idx[
            topic_word_matrix[i, feature_zh_idx].argsort()[-topk:]
        ]
        cn_topk_words = feature[cn_topk_idx][::-1]
        cn_topic.append(cn_topk_words.tolist())
    return en_topic, cn_topic


## NPMI
### code from 家暄學長<https://github.com/ponshane/CLTM/blob/master/src/codebase/topic_evaluator.py>
def documents_to_cooccurence_matrix(
    source_language_documents, target_language_documents
):
    """
    Reference: code from 家暄學長<https://github.com/ponshane/CLTM/blob/master/src/codebase/topic_evaluator.py>
    """

    compound_documents = [
        doc_in_source + doc_in_target
        for doc_in_source, doc_in_target in zip(
            source_language_documents, target_language_documents
        )
    ]

    # turn into gensim's corpora
    compound_dictionary = corpora.Dictionary(compound_documents)
    compund_corpus = [compound_dictionary.doc2bow(text) for text in compound_documents]

    # transform into term_document matrix, each element represents as frequency
    term_document_matrix = corpus2csc(compund_corpus)

    # 利用 corpus2csc 轉換後每個元素為該詞於該篇的詞頻(會大於1)，但 umass score 需要的是 the count of documents containing the word
    # 因此得利用 np.where 重新轉換矩陣，使每個元素單純標記該詞是否出現於該篇(1 or 0)
    # np.where 無法在 csc matrix 故使用以下解決
    term_document_matrix[term_document_matrix >= 1] = 1
    cooccurence_matrix = term_document_matrix @ term_document_matrix.T

    print(f"type of cooccurence_matrix: {type(cooccurence_matrix)}")
    print(
        f"shape of tdm and cooccurence_matrix: {(term_document_matrix.shape, cooccurence_matrix.shape)}"
    )
    return (
        cooccurence_matrix,
        term_document_matrix,
        compound_dictionary,
        len(compund_corpus),
    )


def NPMI(cooccurence_matrix, word_i, word_j, num_of_documents):
    epsilon = 1e-12
    co_count = cooccurence_matrix[word_i, word_j] / num_of_documents
    single_count_i = cooccurence_matrix[word_i, word_i] / num_of_documents
    single_count_j = cooccurence_matrix[word_j, word_j] / num_of_documents
    pmi = math.log((co_count + epsilon) / (single_count_i * single_count_j))
    return pmi / (math.log(co_count + epsilon) * (-1))


def npmi_score(
    cn_topic,
    en_topic,
    topk,
    cooccurence_matrix,
    compound_dictionary,
    num_of_documents,
    coherence_method,
):
    """
    Input: list of list: cn_topic(中文分群結果), en_topic(英文分群結果); scalar: topk(衡量topk個字); matrix: cooccurence_matrix
    (儲存每個字出現次數和兩兩個字的共同出現次數 － 以篇為單位); compound_dictionary: gensim 的字典
    Output: umass, npmi coherence score
    Reference:
        1) http://qpleple.com/topic-coherence-to-evaluate-topic-models/
        2) Mimno, D., Wallach, H. M., Talley, E., Leenders, M., & McCallum, A. (2011, July). Optimizing semantic coherence in topic models.
           #In Proceedings of the conference on empirical methods in natural language processing (pp. 262-272). Association for Computational Linguistics.
    Issue:
        1) [Solve!] Original metric uses count of documents containing the words
    """
    each_topic_coher = []
    for ctopic, etopic in zip(cn_topic, en_topic):
        # below two assertion is very important because
        # 1) minor problem split_language method is a risky method because it may strips some words
        # 2) continue LDAs can not promise to produce the same vocabularies size across languages,
        #    and be a extreme imbalance distribution. (單語言主題群，僅有少數跨語言詞彙)

        assert len(ctopic) >= topk
        assert len(etopic) >= topk

        cn_idx = [
            compound_dictionary.token2id[cn]
            for cn in ctopic[:topk]
            if cn in compound_dictionary.token2id
        ]
        en_idx = [
            compound_dictionary.token2id[en]
            for en in etopic[:topk]
            if en in compound_dictionary.token2id
        ]

        """
        debug line
        print(ctopic[:topk])
        print(etopic[:topk])
        """

        coherences = []
        for each_cn in cn_idx:
            for each_en in en_idx:
                if coherence_method == "umass":
                    # calculate_umass_score_between_two_words
                    co_count = cooccurence_matrix[each_cn, each_en]
                    single_count = cooccurence_matrix[each_en, each_en]
                    pmi = math.log((co_count + 1) / single_count)
                    coherences.append(pmi)
                elif coherence_method == "npmi":
                    npmi = NPMI(cooccurence_matrix, each_cn, each_en, num_of_documents)
                    coherences.append(npmi)

        each_topic_coher.append(sum(coherences) / len(coherences))
    return sum(each_topic_coher) / len(each_topic_coher)


# diversity
def divetsity(topic_num: int, lang_topic_words: List[List[str]]):
    total = topic_num * len(lang_topic_words[0])
    unique_num = len(set(word for word in chain(*lang_topic_words)))
    return unique_num / total
