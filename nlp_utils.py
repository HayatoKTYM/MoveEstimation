import torch
import MeCab
import numpy as np
import re
from collections import Counter

saleVal = re.compile('[:.0-9a-z_A-Z!?%()|]')
def clean_sentence(text:str) -> str :
    """
    """
    #text = re.sub(r"<笑>",'',text)
    #text = "(D_ウズ)でもそれ鑑賞の一部だからしようと思っているんだけど:"
    text = saleVal.sub("",text)
    text = re.sub(r"(\()(.*?)\)",'',text)
    text = re.sub(r"<.?>",'',text)
    return text

usr2id = {'A:':0, 'B:':1, 'C:':2}
label2id = {'s':0, 'b':1, 'f':2, 'r':3, 'i':4}

def get_usrID(users :list) -> np.ndarray:
    """
    直前の発話に対して，
        発話者が変わった場合 1
        発話者が同じ場合は 0
    :param: users : list 各発話の発話者(A,B,C)
    return: userID : list 発話者の変化(0,1)
    """
    userID = [0 if users[i] == users[i+1] else 1 for i in range(len(users)-1)]
    return np.array([0] + userID)

def encode_sentence(text_list:list,  MAX_SIZE: int = 10,
                    start: int = 0, end: int = 10) :
    """
    :param text_list:
    :return: sentences 単語をIDに変換したベクトル
             X_tag     単語の品詞をone-hot-eoncodingしたベクトル
             X_tag2    単語の品詞<詳細>をone-hot-eoncodingしたベクトル
             user      発話したuserが切り替わったかどうかのIDベクトル
             label     moveラベル
    """
    sentences = []
    label = []
    users = []
    for i in range(start,end):
        try:
            with open(text_list[i], encoding='shift-jis') as f:
                lines = f.readlines()
        except:
            with open(text_list[i], encoding='utf-8') as f:
                lines = f.readlines()

        lines = [line.rstrip().split() for line in lines]
        for line in lines:
            text = clean_sentence(line[-2])
            if text == "" or text in ["<笑>", "<息>", "<声>"]: continue
            label.append(label2id[line[-1]])
            sentences.append(text)
            users.append(line[-3])

    users = get_usrID(users)
    print(Counter(label[:4500]))
    feature , X_tag, X_tag2 = wakati(sentences, MAX_SIZE= MAX_SIZE)

    return torch.tensor(feature,dtype=torch.long), torch.tensor(X_tag,dtype=torch.float), torch.tensor(X_tag2,dtype=torch.float), torch.tensor(users,dtype=torch.float), torch.tensor(label, dtype=torch.long)


def wakati(sentences: list , MAX_SIZE: int = 10):
    """
    分かち書きして返す関数
    param: [私は学生です,私は犬です]
    return:
    """
    texts = []
    word2tag = {}
    word2tag2 = {}
    tag_list = []
    tag_list2 = []


    m = MeCab.Tagger("-d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    for sentence in sentences:

        tags = [0] * MAX_SIZE
        tags2 = [0] * MAX_SIZE
        word_list = ["0"] * MAX_SIZE
        words_chasen = m.parse(sentence).split('\n')
        for i, word_chasen in enumerate(words_chasen[-MAX_SIZE:]):
            """
            ['映画', '名詞,一般,*,*,*,*,映画,エイガ,エイガ']
            """
            if word_chasen == 'EOS': break
            tag1 = word_chasen.split("\t")[1].split(',')[0]  # 映画
            tag2 = word_chasen.split("\t")[1].split(',')[1]  # 名詞

            if tag1 not in word2tag:
                word2tag[tag1] = len(word2tag)
            if tag2 not in word2tag2:
                word2tag2[tag2] = len(word2tag2)

            # word = word_chasen.split("\t")[0]
            word_list[i] = word_chasen.split("\t")[0]

            tags[i] = word2tag[tag1]
            tags2[i] = word2tag2[tag2]
        tag_list.append(tags)
        tag_list2.append(tags2)
        texts.append(word_list)

    feature = [word2id(words, MAX_SIZE=MAX_SIZE) for words in texts]

    return np.array(feature,dtype=np.int32), \
           np.array([np.eye(len(word2tag))[i] for i in tag_list]),\
           np.array([np.eye(len(word2tag2))[i] for i in tag_list2])


def word2id(x: list, MAX_SIZE: int = 10) -> list:
    """
    param: x...分かち書きされた１文章
    return: t...分かち書きされた各単語をidに変換したもの
    """
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    import json
    #with open('/mnt/aoni02/katayama/new_word_index.json') as w:
    with open('/mnt/aoni02/katayama/nwjc_sudachi_full_abc_w2v/word_index.json') as w:
        word_index = json.load(w)

    t = np.zeros(MAX_SIZE)
    for i, word in enumerate(x):
        word = word.lower()
        if word in filters: continue
        try:
            t[i] = word_index[word]
        except:
            if word != '0': print(word, end=' ')
            t[i] = 0
    t = np.array(t).astype('int32')
    assert t.min() >= 0 , print(t)
    return t



