# coding=utf-8

"""
Created by jayveehe on 2017/8/4.
http://code.dianpingoa.com/hejiawei03
"""
import json
import os

import datetime
import gensim
import sys
from gensim import corpora


PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_PATH)
PROJECT_PATH = os.path.dirname(PARENT_PATH)
print 'Related file: %s\tPROJECT path = %s\nPARENT PATH = %s' % (__file__, PROJECT_PATH, PARENT_PATH)
sys.path.append(PROJECT_PATH)


def build_event_dictionay(event_seq_set, save_path=None):
    event_dictionary = corpora.Dictionary(event_seq_set)
    if save_path:
        event_dictionary.save(save_path)
    return event_dictionary


def train_word2vec(input_data_root, model_path, n_iter=50, min_count=2, sorted_vocab=1, workers=4, window=5,
                   dim_size=50):
    t = MySentences(input_data_root)
    wv_model = gensim.models.Word2Vec(t, min_count=min_count, sorted_vocab=sorted_vocab, workers=workers,
                                      window=window,
                                      size=dim_size, sg=0, iter=n_iter)
    wv_model.save(model_path)
    return wv_model


def _train_model():
    wvmodel = train_word2vec('./datas/test', './models/event2vec.model', n_iter=50)
    query_event = 'meishi_home_sort_select_tap'
    print '%s most similar:' % query_event
    print wvmodel.wv[query_event]
    for i in wvmodel.most_similar(query_event, topn=20):
        print i[0], i[1]
        # train_word2vec('./datas/test','./models/event2vec.model')


def _load_model():
    loaded_model = gensim.models.Word2Vec.load('./models/event2vec_100d.model')
    query_event = 'home_homesearch_tap'
    try:
        print '%s most similar:' % query_event
        print loaded_model.wv[query_event]
        for i in loaded_model.most_similar(query_event, topn=20):
            print i[0], i[1]
    except Exception, e:
        print 'event: %s not in vocabulary, error details=%s' % (query_event, e)
        print loaded_model.vector_size


def dump_id2vec_json(model_path='./models/event2vec_100d.model', output_path='./datas/event2vec_map.json'):
    loaded_model = gensim.models.Word2Vec.load(model_path)
    vocabulary = loaded_model.wv.vocab
    res_dict = {}
    for k in vocabulary.keys():
        vec = loaded_model.wv[k]
        res_dict[k] = vec.tolist()
    fout = open(output_path, 'w')
    fout.write(json.dumps(res_dict))
    return res_dict


if __name__ == '__main__':
    # with open('/Users/jayveehe/Jobs/arts-sceneengine/offline-process/word2vec_embedding/datas/event_seq_list.json',
    #           'r')as fin:
    #     event_seq_list = []
    #     for line in fin:
    #         event_seq = json.loads(line)
    #         event_seq_list.append(event_seq)
    #     build_event_dictionay(event_seq_list, './datas/event_dictionary.dict')
    #     wvmodel = train_word2vec(event_seq_list, './models/event2vec.model')
    # cur_time = datetime.datetime.strptime('2017-07-02 13:44:21', '%Y-%m-%d %H:%M:%S')
    # print cur_time.minute
    # a = cur_time.today()
    # dump_id2vec_json()
    _load_model()
