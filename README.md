# QANet for Chinese

## Source:
This work is mainly coresponding to:
  * **paper**: [QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension](https://arxiv.org/abs/1804.09541), [DuReader- a Chinese Machine Reading Comprehension Dataset](https://arxiv.org/abs/1711.05073v2) <br>
  * **code**: a tensorflow implemention from [NLPLearn](https://github.com/NLPLearn/QANet), a tensorflow implemention from [SeanLee97](https://github.com/SeanLee97/QANet_dureader), a keras approach from [ewrfcas](https://github.com/ewrfcas/QANet_keras), a basic introduction for Reading Comprehension (RC) task from [facebookresearch](https://github.com/facebookresearch/DrQA) <br>
  * **embedding dictionary**: [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/zh/embedding.html) <br>
  * **embedding dictionary example**: Each line of Tencent_AILab_ChineseEmbedding.txt is like \[str(1 dimensions), vec(200 dimensions)\] below:
  ```
  ['的', '0.209092', '-0.165459', '-0.058054', '0.281176', '0.102982', '0.099868', '0.047287', '0.113531', '0.202805', '0.240482', '0.026028', '0.073504', '0.010873', '0.010201', '-0.056060', '-0.063864', '-0.025928', '-0.158832', '-0.019444', '-0.144610', '-0.124821', '0.000499', '-0.050971', '0.113983', '0.088150', '0.080318', '-0.145976', '0.093325', '0.139695', '-0.082682', '-0.034356', '0.061241', '-0.090153', '0.053166', '-0.171991', '-0.187834', '0.115600', '0.219545', '-0.200234', '-0.106904', '0.033836', '0.005707', '0.484198', '0.147382', '-0.165274', '0.094883', '-0.202281', '-0.638371', '-0.127920', '-0.212338', '-0.250738', '-0.022411', '-0.315008', '0.169237', '-0.002799', '0.019125', '0.017462', '0.028013', '0.195060', '0.036385', '-0.051681', '0.154037', '0.214785', '-0.179985', '-0.020429', '-0.044819', '-0.074923', '0.105441', '-0.081715', '-0.034099', '-0.096518', '-0.004290', '0.095423', '0.234515', '-0.138332', '0.134917', '0.082070', '0.051714', '0.159327', '0.061818', '0.037091', '0.239265', '0.073274', '0.170960', '0.223636', '-0.187691', '-0.206850', '-0.051000', '-0.269477', '-0.116970', '0.213069', '-0.096122', '0.035362', '-0.254648', '0.021978', '0.071687', '0.109870', '-0.104643', '-0.175653', '0.097061', '-0.068692', '0.196374', '0.007704', '0.072367', '-0.275905', '0.217282', '-0.056664', '-0.321484', '-0.004813', '-0.041167', '-0.118400', '-0.159937', '0.065294', '-0.092538', '0.013975', '-0.219047', '-0.058431', '-0.177256', '-0.043169', '-0.151647', '-0.006049', '-0.279595', '-0.005488', '0.096733', '0.147219', '0.197677', '-0.088133', '0.053465', '0.038738', '0.059665', '-0.132819', '0.019606', '0.224926', '-0.176136', '-0.411968', '-0.044071', '-0.120198', '-0.107929', '-0.001640', '0.036719', '-0.243131', '-0.273457', '-0.317418', '-0.079236', '0.054842', '-0.143945', '0.168189', '-0.013057', '-0.145664', '0.135278', '0.029447', '-0.141014', '-0.183899', '-0.080112', '-0.113538', '0.071163', '0.134968', '0.141939', '0.144405', '-0.249114', '0.454654', '-0.077072', '-0.001521', '0.298252', '0.160275', '0.085942', '-0.213363', '0.083022', '-0.000400', '0.134826', '-0.000681', '-0.017328', '-0.026751', '0.111903', '0.010307', '-0.124723', '0.031472', '0.081697', '0.071449', '0.011486', '-0.091571', '-0.039319', '-0.112756', '0.171106', '0.026869', '-0.077058', '-0.052948', '0.252645', '-0.035071', '0.040870', '0.277828', '0.085193', '0.006959', '-0.048913', '0.279133', '0.169515', '0.068156', '-0.278624', '-0.173408', '0.035439']
```
<br>

## Dataset
* **source**: The dataset used for this work is mainly from [DuReader Dataset](http://ai.baidu.com/broad/subordinate?dataset=dureader). <br>
* **format**: Training, validation, and test data are stored in three large json files, with the structure like this: json = {'data': lst_data} where lst_data = \[sample_1, sample_2, ..., sample_N\]. In detail, sample_i is expanded as follows: <br>
```

sample_i = {'title': '', 
            'paragraphs': [{'context': string, 
                            'segmented_context': list, 
                            'qas': [{'question': string, 
                                     'segmented_question': list, 
                                     'answers': [{'text': string, 'answer_span': [int_start, int_end]}], 
                            'id': int_index}]
                           }]  # paragraph now
            }  # sample now
```
* **example**: a sample would be like:
```
{'title': '', 'paragraphs': [{'context': '布袋线，为位于台湾台南市新营区与嘉义县布袋镇间，由台湾糖业股份有限公司新营总厂经营之轻便铁路，今既废止。此线为糖业铁路客运化之始，开办于1909年。纵贯线铁路在选线时，在嘉义曾文溪间即有经过盐水港（今盐水）或新营庄（今新营）之议。详见纵贯线(南段)。当时虽曾研议另建新营经由盐水至北门屿（今北门区）之官线铁道支线弥补不足，唯此案未成真。新营＝盐水＝布袋间铁道运输稍后由制糖会社完成，属于盐水港制糖兼办之客运业务。据铁道部之资料，新营庄＝盐水港间五哩三分于1909年（明治42年）5月20日开始营业，亦为台湾首条糖业铁路定期营业线。然而，根据同年3月4日《台湾日日新报》资料，在官方核准开办营业线之前，当时新营-{庄}-＝岸内-{庄}-间即对外办理客运（一日4往复），车资内地人（即日本人）15钱、本岛人（台湾人）10钱；1913年（大正2年）3月8日，营业区间延至布袋嘴（今布袋）。战后布袋线曾进行数次改动。包括糖铁新营车站（原址位于台铁车站旁，今已成停车场）因破烂不堪，1950年迁移百余米至今址。另外，布袋车站因位于市区之外，曾应民众要求，利用盐业铁路（台盐公司所有）延伸营业间区750m至贴近市区的半路店、但仅维持数年。沿线各车站亦有重修，大部份皆非日治时期原貌。布袋线为762mm狭轨铁路，但在新营＝岸内间，另有一条并行之新岸线，为762mm及1067mm轨距之三线轨道。后者可允许台铁货车驶至岸内。新营-厂前-（南信号所）-工作站前-东太子宫-太子宫-南门-盐水-岸内-义竹-埤子头-安溪寮-前东港-振寮-布袋（-半路店）支线：东子宫-纸浆厂', 'segmented_context': ['布袋', '线', '，', '为', '位于', '台湾', '台南市', '新', '营区', '与', '嘉义县', '布袋镇', '间', '，', '由', '台湾糖业', '股份', '有限公司', '新营', '总厂', '经营', '之', '轻便', '铁路', '，', '今', '既', '废止', '。', '此线', '为', '糖业', '铁路', '客运', '化之始', '，', '开办', '于', '1909', '年', '。', '纵贯线', '铁路', '在', '选线', '时', '，', '在', '嘉义', '曾文溪', '间', '即', '有', '经过', '盐水', '港', '（', '今', '盐水', '）', '或', '新营', '庄', '（', '今', '新营', '）', '之议', '。', '详见', '纵贯线', '(', '南段', ')', '。', '当时', '虽', '曾', '研议', '另', '建新', '营', '经由', '盐水', '至', '北门', '屿', '（', '今', '北门', '区', '）', '之', '官线', '铁道', '支线', '弥补', '不足', '，', '唯', '此案', '未成', '真', '。', '新营', '＝', '盐水', '＝', '布袋', '间', '铁道', '运输', '稍后', '由', '制糖', '会社', '完成', '，', '属于', '盐水', '港', '制糖', '兼办', '之', '客运', '业务', '。', '据', '铁道部', '之', '资料', '，', '新营', '庄', '＝', '盐水', '港间', '五哩', '三分', '于', '1909', '年', '（', '明治', '42', '年', '）', '5', '月', '20', '日', '开始', '营业', '，', '亦', '为', '台湾', '首条', '糖业', '铁路', '定期', '营业', '线', '。', '然而', '，', '根据', '同年', '3', '月', '4', '日', '《', '台湾', '日日', '新报', '》', '资料', '，', '在', '官方', '核准', '开办', '营业', '线', '之前', '，', '当时', '新营', '-', '{', '庄', '}', '-', '＝', '岸内', '-', '{', '庄', '}', '-', '间', '即', '对外', '办理', '客运', '（', '一日', '4', '往复', '）', '，', '车资', '内地', '人', '（', '即', '日本', '人', '）', '15', '钱', '、', '本岛人', '（', '台湾人', '）', '10', '钱', '；', '1913', '年', '（', '大正', '2', '年', '）', '3', '月', '8', '日', '，', '营业', '区间', '延至', '布袋', '嘴', '（', '今', '布袋', '）', '。', '战后', '布袋', '线', '曾', '进行', '数次', '改动', '。', '包括', '糖铁', '新营', '车站', '（', '原址', '位于', '台铁', '车站', '旁', '，', '今', '已成', '停车场', '）', '因', '破烂不堪', '，', '1950', '年', '迁移', '百余米', '至今', '址', '。', '另外', '，', '布袋', '车站', '因', '位于', '市区', '之外', '，', '曾应', '民众', '要求', '，', '利用', '盐业', '铁路', '（', '台盐', '公司', '所有', '）', '延伸', '营业', '间区', '750m', '至', '贴近', '市区', '的', '半路', '店', '、', '但仅', '维持', '数年', '。', '沿线', '各', '车站', '亦', '有', '重修', '，', '大部份', '皆', '非日治', '时期', '原貌', '。', '布袋', '线为', '762mm', '狭轨', '铁路', '，', '但', '在', '新营', '＝', '岸内间', '，', '另有', '一条', '并行', '之新', '岸线', '，', '为', '762mm', '及', '1067mm', '轨距', '之', '三线', '轨道', '。', '后者', '可', '允许', '台铁', '货车', '驶至岸', '内', '。', '新营', '-', '厂前', '-', '（', '南', '信号', '所', '）', '-', '工作站', '前', '-', '东', '太子', '宫', '-', '太子', '宫', '-', '南门', '-', '盐水', '-', '岸内', '-', '义竹', '-', '埤子头', '-', '安溪', '寮', '-', '前', '东港', '-', '振', '寮', '-', '布袋', '（', '-', '半路', '店', '）', '支线', '：', '东', '子宫', '-', '纸浆厂'], 'qas': [{'question': '布袋线是哪家公司经营的轻便铁路？', 'segmented_question': ['布袋', '线', '是', '哪家', '公司', '经营', '的', '轻便', '铁路', '？'], 'answers': [{'text': '台湾糖业股份有限公司新营总厂', 'answer_span': [15, 19]}], 'id': 157}]}]}
```
<br>

## Codes Dependency:
```
config -> prepro

     | -> main(train|test) -> model -> layers
                         | -> util

predict -> preprocess
      | -> model -> layers
      | -> config    

server -> preprocess
      | -> model -> layers
      | -> config 
      
```
<br>

## Command Line:
* **preprocess**: preprocess the used datasets, get word embeddings and word dictionaries.
```bash
python config.py --mode prepro
```
* **train**: train, evaluate, and store the model.
```bash
python config.py --mode train
```
* **test**: test the model.
```bash
python config.py --mode test
```
* **predict**: predict single example defined in predict.py.
```bash
python predict.py
```
* **open server**: run the flask server, and do the example via Postman.
```bash
python server.py
```

* **get prediction**: Postman is the first choice, or use the following script:
```
import requests

url = "http://xxx.xxx.xxx/"

payload = "{\"context\": \"星空凛是日本二次元偶像企划《lovelive!》的主要人物之一。15岁。现读高中一年级。在体育会系中一向开朗活泼，与其闷闷不乐不如身体先行动起来的类型。自己对于偶像活动最初并没有什么热情，起初想加入田径部，后来在帮助小泉花阳加入μ's之后受到邀请加入了μ's。在动画第一季中，因为高坂穗乃果生病而退出LoveLive！的比赛。在第二季中与其他八人一起再次参加LoveLive！并荣获冠军。\",\"question\":\"星空凛多少岁？\"}".encode("utf-8")

headers = {'Content-Type': "application/json", 
           'cache-control': "no-cache", 
           'Postman-Token': "57b70148-73b7-4f8b-b300-24ad2d51ecf7"}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)
```
<br>

## Requirements
  * Python>=3.5
  * TensorFlow>=1.5
  * numpy
  * jieba
  * tqdm
  * ujson(optional)
  * pandas(optional, if creating dataset)
  * Flask(optional, if runing the server.py)
<br>

## TODO
- [x] Simplize the code in model.py, layer.py, utils.py, main.py.
- [x] Add prepare.py for training and testing DuReader-based Dataset.
- [x] Train and test a weak baseline.
- [x] Write predict.py and preprocess.py (for operate predicted questions and contexts).
- [x] Modify the code for efficiency.
- [x] Build dataset for the model.
- [x] Train and test a new baseline.
- [ ] Try to improve the performance.

<br>

<img src="https://github.com/Schlampig/Knowledge_Graph_Wander/blob/master/content/daily_ai_paper_view.png" height=25% width=25% />
