# -*- coding: utf-8 -*-
"""
This code is used to transform raw sample (denoted as "line") from DuReader to new_sample in json format:

input:
line = {"documents": [
                      {"is_selected": true,
                       "title": str,
                       "most_related_para": 0,  # 定位目标context是paragraphs里的哪个str
                       "segmented_title":list[token, token, …],
                       "segmented_paragraphs": list[lst, lst, …]   # 其中内部lst=[token, token, …]
                       "paragraphs": list[str, str, …]}, # str对应segmented_paragraphs的lst

                       {"is_selected": true,
                       "title": str,
                       "most_related_para": 0,
                       "segmented_title":list[token, token, …],
                       "segmented_paragraphs": list[lst, lst, …]
                       "paragraphs": list[str, str, …]},

                       ......
                    ],
         "answer_spans": [[0, 9]],
         "fake_answers": ["用鲁大师，鲁大师有一个游戏模式。"],
         "question": "win10 游戏模式 怎么打开",
         "segmented_answers": list[lst, lst, …]   # 其中内部lst=[token, token, …],
         "answers": list=[str, str, …],
         "answer_docs": [4],  # 定位目标context是documents里的哪个paragraph
         "segmented_question": ["win", "10", "游戏", "模式", "怎么", "打开"],
         "question_type": "DESCRIPTION",
         "question_id": 186575,
         "fact_or_opinion": "FACT",
         "match_scores": [1.0]
         }

output:
new_sample = {"title": '',
              "paragraphs": [{"context": string,
                              "segmented_context": list=[token, token, ...],
                              "qas": [{"question": string,
                                       "segmented_question": list=[token, token, ...],
                                       "question_no_mask": string,
                                       "segmented_question_no_mask": list=[token, token, ...],
                                       "answers": [{"text": string, "answer_span": [int, int]}],
                                       "id": string}]
                                   }]  # paragraph now
                   }  # sample now
"""

import re
import json


def save_data(lst, save_path):
    # input: lst = [json, json, ...]
    #        save_path is a string like xxx.json
    if len(lst) <= 0:
        return None
    else:
        d = {"document": lst}
        with open(save_path, 'w') as f:
            json.dump(d, f)
    return None


def get_sample(json_now):
    # simple filter
    if json_now:
        g = {"true": 0, "false": 1}
        line = eval(json_now, g)
    else:
        return -1
    # if line["question_type"] is not "ENTITY":  # ENTITY/YES_NO/DESCRIPTION
    #         return None
    if len(line["answer_docs"]) == 0 or len(line["answer_spans"]) == 0:
        return None
    try:
        # get indexes
        index_doc = line["answer_docs"][0]  # which paragraph
        index_para = line["documents"][index_doc]["most_related_para"]  # which sentence in the paragraph
        # get values
        c = line["documents"][index_doc]["paragraphs"][index_para]
        seg_c = line["documents"][index_doc]["segmented_paragraphs"][index_para]
        q = line["question"]
        seg_q = line["segmented_question"]
        i_b = line["answer_spans"][0][0]  # the begin index of the answer
        i_e = line["answer_spans"][0][1] + 1  # the end index of the answer
        i_id = line["question_id"]
        ans = "".join(seg_c[i_b:i_e])
    except:
        return None
    t = line["documents"][index_doc]["title"]
    # further filter
    if len(ans) == 0:
        return None
    if len(re.sub('\W', '', ans)) / len(ans) < 0.3:  # 答案中乱码太多
        return None
    if re.match('={3,}', ans) is not None:  # 答案中出现连续等号
        return None
    if re.match('%{2,}', ans) is not None:  # 答案中出现连续百分号
        return None
    if re.match('\*{2,}', ans) is not None:  # 答案中出现连续星号
        return None
    if re.match('\W', ans[0]) is not None:  # 起首句为特殊符号
        return None
    # build new json for the current sample
    json_sample = {'title': t,
                   'paragraphs': [{'context': c,
                                   'segmented_context': seg_c,
                                   'qas': [{'question': q,
                                            'segmented_question': seg_q,
                                            'answers': [{'text': ans, 'answer_span': [i_b, i_e]}],
                                            'id': i_id}]
                                   }]  # paragraph now
                   }  # sample now
    return json_sample


def gen_data(load_path, save_path):
    i = 0
    j = 0
    lst = []
    with open(load_path, "r") as f:
        while True:
            i += 1
            line = f.readline()
            line = get_sample(line)
            if line == -1:
                break
            if line is not None:
                j += 1
                lst.append(line)
                print("{}/{}".format(j, i))
    save_data(lst, save_path)
    return None


if __name__ == "__main__":
    gen_data(load_path="../DuReader_raw_prepro/trainset/zhidao.train.json", save_path="Dutrain.json")
    gen_data(load_path="../DuReader_raw_prepro/devset/zhidao.dev.json", save_path="Dudev.json")
    # Dureader的test里没有answer:
    # gen_data(load_path = "../DuReader_raw_prepro/testset/zhidao.test.json", save_path="Dutest.xlsx")
