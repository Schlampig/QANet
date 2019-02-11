# About the Dataset

## Two different ways to build the dataset
  * **Re-format some open-source Chinese QA dataset** <br>
  * **Generate new Chinese QA dataset from scratch**<br>
  

## Re-format Open-source Dataset
* [clean_DuReader.py](https://github.com/Schlampig/QANet_for_Chinese/blob/master/build_datasets/clean_DuReader.py) is used for transform the [preprocessed DuReader](http://ai.baidu.com/broad/introduction?dataset=dureader) Dataset to the dataset we need. The code just does some cleansing. <br>

## Generate Dataset from Open-Source Chinese Knowledge Graph
* We are inspired by the paper [Zero-Shot Relation Extraction via Reading Comprehension](http://aclweb.org/anthology/K17-1034). In that paper, the authors transform the slot filling task to a QA task by generating a question from a triple for the corresponding context. For a triple <Entity, Attribute, Value> or <Entity 1, Relation, Entity 2>, we could design a question containing the Attribute or Relation and masked Entity [1] for an answer including Value or Entity2. For instance, Triple: <小明，国籍， 中国> Question: XXX的国籍是哪里？ Answer:中国。 | Triple: <Tom，Nationality， USA> Question: What is the nationality of XXX？ Answer:USA. <br>
* We then use triples from [知识工厂](http://kw.fudan.edu.cn/) and context from [百度百科](https://baike.baidu.com/) to build our own Question-Answer-Context dataset. <br>
* [generate_data.py](https://github.com/Schlampig/QANet_for_Chinese/blob/master/build_datasets/generate_data.py) is used to generate the dataset from selected triples. <br>
* Note: Questions for each triple are manually created in file "questions_for_relation.xlsx".


