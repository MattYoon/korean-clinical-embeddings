# korean-clinical-embeddings

2005년부터 2021년까지 건국대학교병원의 다양한 진료과에서 수집된 초진기록 말뭉치로부터 학습된 워드임베딩입니다. 약 1,800만 개의 단어로 학습된 300차원의 Skip-gram word2vec 모델입니다.

## Requirements
~~~
gensim
konlpy
~~~

## 사용방법
Gensim의 KeyedVectors API를 사용했습니다. 추가적인 사용법은 [공식문서](https://radimrehurek.com/gensim/models/keyedvectors.html)를 참고해주세요.

### Load Model
```python
from gensim.models import KeyedVectors
import numpy as np

wv = KeyedVectors.load('kumed-w2v-300.model')
print(wv.most_similar_cosmul('안압'))
```
```
[('녹내장', 0.8615068793296814), ('cosopt', 0.8171359896659851), ('xalatan', 0.8163099884986877), ('IOP', 0.81460040807724), ('압증', 0.802762508392334), ('고안', 0.8004114627838135), ('alphagan', 0.7999970316886902), ('travatan', 0.7930376529693604), ('점안', 0.7927384972572327), ('Baseline', 0.7914065718650818)]
```

### Embed Sentence
사전학습 말뭉치를 형태소 분석기를 이용해 토큰화했기 때문에, 사용 시에도 반드시 형태소 분석기로 문장을 토큰화를 진행하셔야 합니다.

```python
from konlpy.tag import Mecab

mecab = Mecab()
tokens = mecab.morphs('right clavicular head가 돌출되어 있고 세게 만지면 아프다.')
print(tokens)
```
```
['right', 'clavicular', 'head', '가', '돌출', '되', '어', '있', '고', '세', '게', '만지', '면', '아프', '다', '.']
```
```python
vectors = []

for token in tokens:
    try:
        vectors.append(wv[token])
    except KeyError:
        pass # UNK Token일 경우 처리 로직

vectors = np.stack(vectors, axis=0)
print(vectors.shape)
```
```
(16, 300)
```

**본 연구에서 사용된 사전학습 데이터셋은 건국대학교병원의 IRB 승인(2021-11-064)을 받음.**


**The pretraining data used for the project was approved (2021-11-064) by the Institutional Review Board of Konkuk University Hospital.**
