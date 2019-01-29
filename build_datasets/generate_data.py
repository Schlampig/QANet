# -*- coding: utf-8 -*-
"""
This code is used to transform raw_sample from mongodb to new_sample to json file:

raw_sample = {"_id" : ObjectId("5c499e995f627d223d589184"),
              "context" : context_string,
              "triples" : entity_string\trelation_string\tvalue_string\n,
              "relation" : entity_string}

new_sample = {'title': '',
              'paragraphs': [{'context': string,
                              'segmented_context': list=[token, token, ...],
                              'qas': [{'question': string,
                                       'segmented_question': list=[token, token, ...],
                                       'question_no_mask': string,
                                       'segmented_question_no_mask': list=[token, token, ...],
                                       'answers': [{'text': string, 'answer_span': [int, int]}],
                                       'id': string}]
                                   }]  # paragraph now
                   }  # sample now
"""


import json
import pandas as pd
from random import choice
from tqdm import tqdm
import jieba
import pymongo


# global initialization
jieba.add_word("XXX")
client = pymongo.MongoClient('127.0.0.1')  # open mongodb client
table_raw = client['triples']['triples']  # open the raw database


def split_train_and_test(load_path=None):
    if load_path is None:
        lst_train = ["作者", "连载网站", "小说进度", "书名", "出生日期", "出版社", "职业", "国籍", "出生地", "中文学名",
                     "性别", "公司名称", "民族", "外文名", "别名", "经营范围", "类型", "毕业院校", "总部地点", "作品名称",
                     "酒店星级", "出版时间", "成立时间", "主要食材", "地理位置", "逝世日期", "面积", "地区", "创作年代",
                     "餐馆名称", "所属地区", "类别", "主要成就", "行政区类别", "主要原料", "运动项目", "科", "地点", "游戏类型",
                     "人口", "周围景观", "拼音", "公司性质", "拉丁学名", "创办时间", "种", "本名", "特点", "分布区域", "性质",
                     "代表作品", "楼盘名", "属", "游戏大小", "释义", "主演", "籍贯", "占地面积", "时间", "字号", "分类", "解释",
                     "副标题", "导演", "文学体裁", "药品名称", "地址", "位置", "英文名", "海拔", "产品类型", "人均价格", "简称",
                     "隶属", "属性", "出处", "营业时间", "位于", "所处时代", "目", "口味", "建筑面积", "去世时间", "场上位置",
                     "其他名称", "所属运动队", "辅料", "作用", "著名景点", "成立于", "外文名称", "大小", "政治面貌", "品牌",
                     "下辖地区", "型号", "开发商", "制片地区", "原料", "始建于", "配料", "简介", "含义", "定义", "丛书", "主料",
                     "上映时间", "歌曲原唱", "学历", "调料", "信仰", "编剧", "物业类别", "用途", "年平均气温", "材料",
                     "气候条件", "屏幕尺寸", "属于", "纲", "房间数量", "目的", "出品时间", "内容", "实质", "对象"]
        lst_test = ["耕地面积", "组成", "功能", "楼盘地址", "英文名称", "登场作品", "填词", "国家", "年代", "优点", "谱曲",
                    "应用", "员工数", "门", "所属专辑", "公司口号", "版次", "译者", "功效", "发行时间", "上市时间", "原版名称",
                    "主要院系", "主要奖项", "出品公司", "注音", "界", "装帧", "页数", "定价", "宗旨", "身高", "CPU型号",
                    "亚科", "主要营养成分", "编曲", "集数", "校训", "适宜人群", "游戏平台", "体重", "酒店地址", "发行日期",
                    "开本", "其它译名", "系列", "制片人", "语言", "亚目", "尺寸", "音乐风格", "发行公司", "邮政区码", "亚门",
                    "血型", "歌曲语言", "星座", "族"]
        return lst_train, lst_test
    else:
        raise ValueError("No training and test relations information!")
        # TODO: loading codes


def gen_question(load_path):
    df_q = pd.read_excel(load_path, header=None)
    df_q = df_q.fillna('NQ')  # No Question
    dict_q = dict()
    for i in range(df_q.shape[0]):
        for j in range(1, df_q.shape[1]):
            v_now = df_q.iloc[i, j]
            if v_now is 'NQ':
                continue
            if df_q.iloc[i, 0] in dict_q:
                dict_q[df_q.iloc[i, 0]].append(v_now)
            else:
                dict_q[df_q.iloc[i, 0]] = [v_now]
    return dict_q


def match_answer(seg_c, ans):
    # input: seg_c = [token, token, ...]
    #        ans is a string
    # output: i_b and i_e are the begining and ending indexes of ans in seg_c, respectively
    i_b, i_e = None, None
    window_size = len(jieba.lcut(ans))
    if window_size < len(seg_c):
        for i in range(len(seg_c) - window_size):
            window = ''.join(seg_c[i:(i+window_size)])
            if window == ans:
                i_b = i
                i_e = i + window_size
    return i_b, i_e


def get_sample(json_now, dict_q):
    # 读取一条爬取的样本，按照约定的格式生成新的样本
    # input: json_now, is the raw dictionary including _id, context, relation, triple
    #        dict_q, is the question dictionary
    # output: json_sample, is the new json for the current sample
    # get data from the input json
    if json_now["relation"] not in dict_q:
        return None
    c = json_now["context"]
    ans = json_now["triples"].strip("\n").split("\t")[-1]
    if (len(c) < 0) or (ans not in c):
        return None
    seg_c = jieba.lcut(c)
    i_b, i_e = match_answer(seg_c, ans)
    if i_b is None or i_e is None or i_b < len(seg_c)*0.25:
        return None
    q = choice(dict_q[json_now["relation"]])  # randomly choose a question
    seg_q = jieba.lcut(q)
    entity = json_now["triples"].strip("\n").split("\t")[0]  # build normal question, replace XXX with the entity mention
    q_no_mask = q.replace("XXX", entity)
    seg_q_no_mask = jieba.lcut(q_no_mask)
    i_id = str(json_now["_id"])
    # build new json for the current sample
    json_sample = {'title': '',
                   'paragraphs': [{'context': c,
                                   'segmented_context': seg_c,
                                   'qas': [{'question': q,
                                            'segmented_question': seg_q,
                                            'question_no_mask': q_no_mask,
                                            'segmented_question_no_mask': seg_q_no_mask,
                                            'answers': [{'text': ans, 'answer_span': [i_b, i_e]}],
                                            'id': i_id}]
                                   }]  # paragraph now
                   }  # sample now
    return json_sample


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


def gen_data(lst, dict_q, n=100, ratio=-1.0):
    # input: lst = [relation1, relation2, ...]
    #        n is the number of samples
    # output: None, train_data saved
    global client, table_raw
    lst_train, lst_dev, lst_test = [], [], []
    c_train, c_dev, c_test = 0, 0, 0
    for i_r, r in enumerate(lst):
        print("Operate the {}/{} relation: {}.".format(i_r+1, len(lst), r))
        lst_sample = list(table_raw.find({"relation": r}).limit(n))
        if 0.0 < ratio < 1.0:
            i_split = int(round(n*max(ratio, 1-ratio)))
            print("generate training and development data ...")
            for i_sample, sample in enumerate(tqdm(lst_sample)):
                d_new = get_sample(sample, dict_q)
                if d_new is not None:
                    if i_sample < i_split:
                        lst_train.append(d_new)
                        c_train += 1
                    else:
                        lst_dev.append(d_new)
                        c_dev += 1
        else:
            print("generate test data ...")
            for sample in tqdm(lst_sample):
                d_new = get_sample(sample, dict_q)
                if d_new is not None:
                    lst_test.append(d_new)
                    c_test += 1
    # save data
    save_data(lst_train, "train.json")
    save_data(lst_dev, "dev.json")
    save_data(lst_test, "test.json")
    print("Dataset Size: Training-{}, Dev-{}, Test-{}.".format(c_train, c_dev, c_test))
    return None


if __name__ == '__main__':
    d_q = gen_question(load_path="questions_for_200_relation.xlsx")
    l_train, l_test = split_train_and_test()
    gen_data(l_train, d_q, n=3000, ratio=0.8)
    gen_data(l_test, d_q, n=300)
    print("finished")
