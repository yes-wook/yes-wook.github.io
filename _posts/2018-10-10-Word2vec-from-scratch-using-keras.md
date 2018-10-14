---
layout: post
title: Word2vec from scratch using keras
---

본 글에서는 Word2vec의 개녑을 간단히 알아보고, Keras 등을 이용하여 구현해보도록 하겠습니다.  
(그림은 없습니다...꾸준히 정성들여 그림 수식 다 넣어서 포스팅하시는분들 진짜 존경...)

# Word2vec?

이제는 학계에서 모르는 사람이 없을 정도로 식상해져 버린 단어기도 하고 Gensim, SpaCy 등 수많은 관련 패키지들이 나와있는 단어입니다, Word2Vec.  
Bengio교수님이 NNLM(Neural network language model)로 제시한 모형을 구글의(지금은 페이스북으로 옮겼지만) T.Mikolov가 개선한 모형으로 많이 알려져 있습니다.  
단어의 분산표현(Distributed representation of words)으로도 알려져 있는데, 기존에 원핫(One-hot)으로 벡터화 하였던 <i><b>단어들에 대하여, 아무 의미없는 원핫 말고 어떤 의미를 가지는(비록 인간이 해석할 수 없지만) 분산된 벡터로 나타내자</b></i> 라는 아이디어입니다.  
그렇기 때문에 몇가지 키워드를 중심으로 간략하게 설명하겠습니다.  
  
## Distributional hypothesis

한국어로 뭐라고 하는지 잘 모르겠는데, 본 글에서는 '분산가설' 정도로 번역하여 사용하겠습니다.  
분산가설은 <i><b>'비슷한 문맥(Context)에서 등장하는 단어는 비슷한 시맨틱스(Semantics)를 가질것이다'</b></i> (라는 가정) 정도로 이해할 수 있습니다. 예를 들어,
- 나는 바나나를 먹는다
- 나는 사과를 먹는다
- 나는 과자를 먹는다  

라는 문장들이 있다고 할때, 바나나/사과/과자가 유사한 문맥(나는 ~~를 먹는다)에서 등장합니다.  
따라서 바나나/사과/과자는 비슷한 Semantics를 가지고 있는 것으로 파악할 수 있습니다.  
  
Word2vec은 주변 단어를 예측하는 방향으로 학습이 진행되기 때문에, 분산가설을 기본적으로 가정하는 이론입니다. 물론 NNLM계열 모형은 전부 분산가설을 기본 가정으로 하고 있습니다.  

## Hirachical softmax

계층적 소프트맥스는 아래에서 설명하는 네거티브 샘플링과 함께 Word2Vec에서 처음 제시된 개념인데요.  
네거티브샘플링에 비해서는 중요도가 조금 떨어지는 감이 있습니다. 최근에는 많이 사용 안하는것 같은데, 간단히 개념만 설명드리면 아래와 같습니다.  
Vocabrary size가 100K이상인 경우가 부지기수이기 때문에, 기본적으로 NLP에서 소프트맥스 함수는 비용(Cost)이 굉장히 높습니다. 그래서 <i><b>연산량을 줄이기 위해서 소프트맥스함수를 트리(Tree)구조로 만든것</b></i>이 계층적 소프트맥스입니다.  
그렇게되면, 모든 leaf node(Word)에 대해 계산할 필요가 없기 때문에 연산량이 줄어듭니다.  
O(N^2)에서 O(N.logN)으로 줄어드는것으로 알고 있습니다(확인필요).

## Negative sampling

네거티브 샘플링이 Word2Vec의 사실상 가장 큰 기여라고 할 수 있는데, 계층적 소프트맥스가 단순히 소프트맥스 연산을 빠르게 한것이라고 하면, 네거티브 샘플링은 손실함수(Loss function)자체를 근사해서 계산할 수 있도록 한 기술입니다.  
말 그대로 네거티브 샘플링(부정적인 예제 뽑기)인데, 앞에서 말씀드린 것 처럼 소프트맥스는 모든 단어에 대해서 값을 계산하기 때문에 비용이 큰 함수입니다. 따라서 네거티브 샘플링은 <i><b>모든 단어에 대해서 계산하지 말고 실제로 불가능한 예제(네거티브 샘플)에 대해서만 손실함수를 계산하자</b></i>는 아이디어입니다. NCE(Noise-contrastive estimation) loss 라고도 불리는데, 식은 아래와 같습니다.  
![nce loss equation](https://media.licdn.com/dms/image/C5112AQERD1NOGV7T1g/article-inline_image-shrink_1500_2232/0?e=1544659200&v=beta&t=u9_HXUSY2sJC8H57_q1w9B7dQe8JHJSfxZTn-th7hyU)  
(그림 출처: https://www.linkedin.com/pulse/heavy-softmax-use-nce-loss-shamane-siriwardhana/)  

간단히 식을 보면 W_bar는 네거티브 샘플입니다. 해당 단어(w) 주변에 절대 등장하지 않는(데이터 상에서) 단어이고, NCE loss를 Minimize하는 것은 어떤 단어의 Embedded vector를 주변단어와 가깝게, 주변에 온적 없는 단어와는 멀게 만드는 역할을 합니다.  
NCE loss를 사용함으로써, 소프트맥스(다중분류)문제는 입력단어에 대해 문맥 주변에 있는 단어와 문맥 주변에 없는 단어를 이진분류하는 문제로 변경되었습니다. 손실함수의 비용이 어마어마하게 줄어든 것은 말할것도 없지요. 벤지오교수님의 NNLM이 학습시간이 어마어마하게 오래걸리는 것이 가장 큰 단점이었기 때문에 네거티브 샘플링은 한마디로 혁신이었습니다.  
이후에 ~~2Vec이라는 이름을 달고 있는 논문들은 거의 다 이 NCE loss를 사용한 것으로 보시면 될정도입니다.  


# Code from scratch using keras

Keras를 이용해서 Word2vec의 Skip-gram 모형을 학습하는 코드입니다.  
Skip-gram 모형과 CBOW모형의 차이는 목표 단어와 주변단어 중에서 어떤 단어를 입력/출력으로 사용하느냐 하는 차이입니다.  
어떤 모형이 더 좋냐 하는 얘기에 대해서 논란이 있었는데, 요즘 보면 승자는 Skip-gram인 것으로 보입니다.  
(CBOW는 여러개의 주변 단어를 입력으로 하나의 목표 단어를 학습하는데 비해, Skip-gram은 하나의 목표 단어를 입력으로 여러개의 주변 단어를 학습하기 때문에 하나의 단어에 대한 학습횟수가 더 많아서 라는 이야기가 있는것 같은데..뭐...네...)  
  
아래 버전에서 실행되었습니다.

- Python 3.6.5
- Tensorflow 1.11.0
- Keras 2.2.2 (With tensorflow backend)
- Numpy 1.14.5
- Pandas 0.23.0
- NLTK 3.3

개별 코드에 대해 차례차례 설명하고, 전체 코드는 제일 하단에서 확인할 수 있습니다.    

## Background works

코드에서 사용하는 Imports와 몇가지 환경설정에 대한 부분입니다.  
np.random.seed(777) 는 재현성을 위해 Random seed를 고정하는 코드고,
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'는 CUDA를 지원하는 장비가 없는 것으로 설정하여, GPU연산을 하지 않고 CPU로 연산하기 위한 설정입니다.  
  
(Word2vec모형을 GPU연산으로 진행 할 경우 배치데이터를 GPU RAM으로 이전하는데에 소모되는 시간이 과도하여 작은 단위의 배치에서는 CPU연산에 비해 시간이 많이 소모되는 것으로 알려져 있습니다. 대략적으로 512배치정도가 그 선으로 보입니다.)  

```Python
import os
from collections import Counter
from time import time

import numpy as np
import pandas as pd
from keras.layers import Dense, Dot, Embedding, Input, Reshape
from keras.models import Model
from keras.preprocessing.sequence import skipgrams
from nltk.corpus import stopwords

np.random.seed(777)
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
```

## Corpus preprocessing

Corpus를 전처리하는 함수인데, 여기서 Corpus는 Pandas datacolumn의 형태입니다.  
Pandas datacolumn에 들어있는 텍스트에 대해 모두 소문자로 변경하고, 정규식(Regex)을 이용하여 숫자/알파벳/공백 을 제외하고 전부 제거합니다.  
sampling_rate 변수는 테스트를 위해 전체문서 중에서 일부만 샘플링해서 사용하려고 할 때 사용하는 값이며 0~1 사이의 값입니다.  
  
이 코드에 있는 모든 함수의 반환값은 List형태를 기본으로 하고 있습니다.  

```Python
def preprocessing_corpus(corpus, sampling_rate=1.0):
    if sampling_rate is not 1.0:
        corpus = corpus.sample(frac=sampling_rate, replace=False)
    corpus = corpus.str.lower()
    corpus = corpus.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    return corpus.values.tolist()
```

## Making vocabrary

Corpus에서 단어를 추출해서 단어집(Vocabrary)을 구성하는 코드입니다.  
top_n_ratio는 Corpus 내에서 단어의 출현 빈도 기준 상위 몇%의 단어들을 이용하여 어휘집을 구성할 건지에 대한 파라미터 이고, 범위는 0~1입니다.  
또한, NLTK에 있는 영어 불용어를 이용하여 불용어 제거를 진행하였고, 불용어 및 등장횟수가 적어서 단어집에 포함되지 않은 단어들은 UNK로 처리하였습니다.  
당연히 반환값은 List로 입니다.

```Python
def making_vocab(corpus, top_n_ratio=1.0):
    words = np.concatenate(np.core.defchararray.split(corpus)).tolist()

    stopWords = set(stopwords.words('english'))
    words = [word for word in words if word not in stopWords]

    counter = Counter(words)
    if top_n_ratio is not 1.0:
        counter = Counter(dict(counter.most_common(int(top_n_ratio*len(counter)))))
    unique_words = list(counter) + ['UNK']
    return unique_words
```

## Indexing vocabrary

단어집을 이용하여 단어를 숫자로, 숫자를 단어로 인덱싱(Indexing) 및 역 인덱싱(Reverse indexing)하는 작업입니다.  
사실 역 인덱싱은 불필요할 수도 있는데, 일단 만들어둡니다...
크게 어려운 코드는 없고, 반환값은 Dictionary입니다.  

```Python
def vocab_indexing(vocab):
    word2index = {word:index for index, word in enumerate(vocab)}
    index2word = {index:word for word, index in word2index.items()}
    return word2index, index2word
```

## Changing word to index in corpus

이전 함수에서 인덱싱 된 단어들을 이용하여, Corpus상의 단어들을 인덱스로 바꿔주는 함수입니다.  
'A B C A'라는 문장이 있다고 가정하고, A:0, B:1, C:2 로 인덱싱 되었다고 할 때, 원래 문장은 [0,1,2,0]으로 바뀌게 됩니다.  
이렇게 하는 이유는, 학습모형의 Embedding layer의 크기는 Vocabrary_size * Embedding dimension이고, Embedding layer에서 각 인덱스는 인덱스에 해당하는 단어에 대한 Embedding vector이기 때문입니다.  
  
반환값은 List of lists입니다.  

```Python
def word_index_into_corpus(word2index, corpus):
    indexed_corpus = []
    for doc in corpus:
        indexed_corpus.append([word2index[word] if word in word2index else word2index['UNK'] for word in doc.split()])
    return indexed_corpus
```

## Generating traning pairs

학습에 사용될 Pairs를 만드는 함수입니다. Pairs라고 하는 이유는 위에 네거티브샘플링에서 언급한 것 처럼, 네거티브샘플링의 도입으로 다항분류문제는 Positive sample(실제로 주변에 위치하는 단어)는 1, Negative sample(데이터상 단어 주변에 위치하지 않는 단어)는 0으로 예측하는 이진분류 문제로 변경되었습니다.  
  
따라서 기존 데이터셋을 이진분류에 맞게 변경해주는 작업이 필요합니다.  
이 함수는 그 작업을 해주는 함수입니다. Keras의 skipgrams를 사용합니다. 예를 들면 아래와 같은 식입니다.  

- [[1,2,3,4,5,6]] -> [[[2,3], 1], [[2,6], 0]]  
(설정된 Window size 안에 있는 단어끼리는 1, 아닌 단어끼리는 0을 Label로 만들어줍니다.)

```Python
def generating_wordpairs(indexed_corpus, vocab_size, window_size=4):
    X = []
    Y = []
    for row in indexed_corpus:
        x, y = skipgrams(sequence=row, vocabulary_size=vocab_size, window_size=window_size,
                        negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
        X = X + list(x)
        Y = Y + list(y)
    return X, Y
```

## Constructing model

Keras API를 사용하여 학습에 사용될 모형을 구축하는 단계입니다. Word2vec이나 NNLM 논문을 보시면, Target과 Context 두 부분에 대하여 각각의 Embedding layer를 구축하여 따로 학습을 시킨 뒤에, 두 레이어를 Average하여 사용하거나 둘 중 하나만 사용하는 경우가 많으나 저는 하나의 Embedding layer를 사용하도록 하겠습니다.  
또, 모형의 뒷부분에 Dense layer는 Dense(1) 외에는 사용하지 않는 경우가 많은데 이 경우에도 저는 제맘대로 한층을 더 쌓아보았습니다. Word2vec에서 중요한 것은 Embedding layer의 사용이며, 모형의 뒤쪽은 마음대로 구축하셔도 큰 차이는 없을 것 입니다.  

```Python
def consructing_model(vocab_size, embedding_dim=300):
    input_target = Input((1,))
    input_context = Input((1,))

    embedding_layer = Embedding(vocab_size, embedding_dim, input_length=1)

    target_embedding = embedding_layer(input_target)
    target_embedding = Reshape((embedding_dim, 1))(target_embedding)
    context_embedding = embedding_layer(input_context)
    context_embedding = Reshape((embedding_dim, 1))(context_embedding)

    hidden_layer = Dot(axes=1)([target_embedding, context_embedding])
    hidden_layer = Reshape((1,))(hidden_layer)

    output = Dense(16, activation='sigmoid')(hidden_layer)
    output = Dense(1, activation='sigmoid')(output)
    
    model = Model(inputs=[input_target, input_context], outputs=output)
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model
```

## Traning model

모형을 학습하는 함수입니다. Indexed corpus에서 정해진 batch size만큼의 문장을 샘플링해서 네거티브 샘플링을 진행한 뒤에 모형을 학습하는 순서입니다.  
처음에는 Indexed corpus 전체에 대하여 학습셋을 구축하고 그 뒤에 모형 학습을 진행하려고 했는데, Corpus의 사이즈가 커질수록 학습셋 구축에(정확히는 Keras의 skipgrams가) 소모되는 시간이 과도하여서 몇몇 문장만 샘플링하고 학습하는 것을 반복하도록 하였습니다.  

```Python
def training_model(model, epochs, batch_size, indexed_corpus, vocab_size):
    for i in range(epochs):
        idx_batch = np.random.choice(len(indexed_corpus), batch_size)
        X, Y = generating_wordpairs(np.array(indexed_corpus)[idx_batch].tolist(), vocab_size)

        word_target, word_context = zip(*X)
        word_target = np.array(word_target, dtype=np.int32)
        word_context = np.array(word_context, dtype=np.int32)

        target = np.zeros((1,))
        context = np.zeros((1,))
        label = np.zeros((1,))
        idx = np.random.randint(0, len(Y)-1)
        target[0,] = word_target[idx]
        context[0,] = word_context[idx]
        label[0,] = Y[idx]
        loss = model.train_on_batch([target, context], label)
        if i % 1000 == 0:
            print("Iteration {}, loss={}".format(i, loss))
    return model
```

## Saving vector

학습된 Embedded vector를 저장하는 함수입니다. Gensim에서 불러올 수 있는 형태로 저장하고 있으며, 아래와 같은 함수를 통해 불러올 수 있습니다.  
```Python
from gensim.models.keyedvectors import Word2VecKeyedVectors

file_name = saved file path
word_vectors = Word2VecKeyedVectors.load_word2vec_format(file_name, binary=False)

```

사실 Gensim형식이라는 것이 크게 다른것은 아니고, tsv형식입니다. 따라서 엑셀등으로도 열어보실 수 있습니다.  

```Python
def save_vectors(file_path, vocab_size, embedding_dim, model, word2index):
    f = open(file_path, 'w')
    f.write('{} {}\n'.format(vocab_size-1, embedding_dim))
    vectors = model.get_weights()[0]
    for word, i in word2index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()
    return file_path
```

## Executing

실행하는 부분입니다. 각 단계마다 소요되는 시간을 측정하는 것 이외에는 크게 특별한 부분은 없습니다.  
corpus load하는 부분만 신경써주시고 (file_path 설정 및 column index 설정),
embedding_dim, epochs, batch_sentence_size 등은 원하는대로 설정하시면 됩니다.  

```Python
if __name__ == "__main__":
    time_start = time()
    time_check = time()
    
    corpus = pd.read_csv(file_path).iloc[:,1] 
    corpus = preprocessing_corpus(corpus, sampling_rate=1.0)
    print("Corpus was loaded in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    vocab = make_vocab(corpus, top_n_ratio=0.8)
    vocab_size = len(vocab)
    print("Vocabulary was made in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    word2index, index2word = vocab_indexing(vocab)
    print("Vocabulary was indexed in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    indexed_corpus = word_index_into_corpus(word2index, corpus)
    print("Corpus was indexed in\t{time} sec".format(time=time()-time_check)); time_check = time()

    embedding_dim = 100
    model = consructing_model(vocab_size, embedding_dim=embedding_dim)
    print("Model was constructed in\t{time} sec".format(time=time()-time_check)); time_check = time()

    epochs = 100001
    batch_sentence_size = 512
    model = training_model(model, epochs, 512, indexed_corpus, vocab_size)
    print("Traning was done in\t{time} sec".format(time=time()-time_check)); time_check = time()

    save_path = save_vectors('vectors_on_batch.txt', vocab_size, embedding_dim, model, word2index)
    print("Trained vector was saved in\t{time} sec".format(time=time()-time_check)); time_check = time()

    print("Done: overall process consumes\t{time} sec".format(time=time()-time_start))

```

# Full code

전체 코드는 아래와 같습니다.  

```Python
import os
from collections import Counter
from time import time

import numpy as np
import pandas as pd
from keras.layers import Dense, Dot, Embedding, Input, Reshape
from keras.models import Model
from keras.preprocessing.sequence import skipgrams
from nltk.corpus import stopwords

np.random.seed(777)
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def preprocessing_corpus(corpus, sampling_rate=1.0):
    if sampling_rate is not 1.0:
        corpus = corpus.sample(frac=sampling_rate, replace=False)
    corpus = corpus.str.lower()
    corpus = corpus.str.replace(r'[^A-Za-z0-9\s]', ' ', regex=True)
    return corpus.values.tolist()

def making_vocab(corpus, top_n_ratio=1.0):
    words = np.concatenate(np.core.defchararray.split(corpus)).tolist()

    stopWords = set(stopwords.words('english'))
    words = [word for word in words if word not in stopWords]

    counter = Counter(words)
    if top_n_ratio is not 1.0:
        counter = Counter(dict(counter.most_common(int(top_n_ratio*len(counter)))))
    unique_words = list(counter) + ['UNK']
    return unique_words
    
def vocab_indexing(vocab):
    word2index = {word:index for index, word in enumerate(vocab)}
    index2word = {index:word for word, index in word2index.items()}
    return word2index, index2word

def word_index_into_corpus(word2index, corpus):
    indexed_corpus = []
    for doc in corpus:
        indexed_corpus.append([word2index[word] if word in word2index else word2index['UNK'] for word in doc.split()])
    return indexed_corpus

def generating_wordpairs(indexed_corpus, vocab_size, window_size=4):
    X = []
    Y = []
    for row in indexed_corpus:
        x, y = skipgrams(sequence=row, vocabulary_size=vocab_size, window_size=window_size,
                        negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
        X = X + list(x)
        Y = Y + list(y)
    return X, Y

def consructing_model(vocab_size, embedding_dim=300):
    input_target = Input((1,))
    input_context = Input((1,))

    embedding_layer = Embedding(vocab_size, embedding_dim, input_length=1)

    target_embedding = embedding_layer(input_target)
    target_embedding = Reshape((embedding_dim, 1))(target_embedding)
    context_embedding = embedding_layer(input_context)
    context_embedding = Reshape((embedding_dim, 1))(context_embedding)

    hidden_layer = Dot(axes=1)([target_embedding, context_embedding])
    hidden_layer = Reshape((1,))(hidden_layer)

    output = Dense(16, activation='sigmoid')(hidden_layer)
    output = Dense(1, activation='sigmoid')(output)
    
    model = Model(inputs=[input_target, input_context], outputs=output)
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

def training_model(model, epochs, batch_size, indexed_corpus, vocab_size):
    for i in range(epochs):
        idx_batch = np.random.choice(len(indexed_corpus), batch_size)
        X, Y = generating_wordpairs(np.array(indexed_corpus)[idx_batch].tolist(), vocab_size)

        word_target, word_context = zip(*X)
        word_target = np.array(word_target, dtype=np.int32)
        word_context = np.array(word_context, dtype=np.int32)

        target = np.zeros((1,))
        context = np.zeros((1,))
        label = np.zeros((1,))
        idx = np.random.randint(0, len(Y)-1)
        target[0,] = word_target[idx]
        context[0,] = word_context[idx]
        label[0,] = Y[idx]
        loss = model.train_on_batch([target, context], label)
        if i % 1000 == 0:
            print("Iteration {}, loss={}".format(i, loss))
    return model

def save_vectors(file_path, vocab_size, embedding_dim, model, word2index):
    f = open(file_path, 'w')
    f.write('{} {}\n'.format(vocab_size-1, embedding_dim))
    vectors = model.get_weights()[0]
    for word, i in word2index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()
    return file_path

if __name__ == "__main__":
    time_start = time()
    time_check = time()
    
    corpus = pd.read_csv("abcnews-date-text.csv").iloc[:,1] 
    corpus = preprocessing_corpus(corpus, sampling_rate=1.0)
    print("Corpus was loaded in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    vocab = make_vocab(corpus, top_n_ratio=0.8)
    vocab_size = len(vocab)
    print("Vocabulary was made in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    word2index, index2word = vocab_indexing(vocab)
    print("Vocabulary was indexed in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    indexed_corpus = word_index_into_corpus(word2index, corpus)
    print("Corpus was indexed in\t{time} sec".format(time=time()-time_check)); time_check = time()

    embedding_dim = 100
    model = consructing_model(vocab_size, embedding_dim=embedding_dim)
    print("Model was constructed in\t{time} sec".format(time=time()-time_check)); time_check = time()

    epochs = 100001
    batch_sentence_size = 512
    model = training_model(model, epochs, 512, indexed_corpus, vocab_size)
    print("Traning was done in\t{time} sec".format(time=time()-time_check)); time_check = time()

    save_path = save_vectors('vectors_on_batch.txt', vocab_size, embedding_dim, model, word2index)
    print("Trained vector was saved in\t{time} sec".format(time=time()-time_check)); time_check = time()

    print("Done: overall process consumes\t{time} sec".format(time=time()-time_start))

```