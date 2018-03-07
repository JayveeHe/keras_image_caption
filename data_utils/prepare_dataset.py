# coding=utf-8

"""
Created by jayveehe on 2018/3/6.
http://code.dianpingoa.com/hejiawei03
"""
import json

import jieba


def process_raw_hive_data(fin_path, fout_path, line_fout_path):
    with open(fin_path, 'r') as fin, open(fout_path, 'w') as fout, open(line_fout_path, 'w') as line_fout:
        refined_dict = {}
        for line in fin:
            reviewid, userid, shopid, cityid, shoptype, reviewbody, picid, pickey, picurl = line.strip().split('\t')
            if reviewid in refined_dict:
                refined_dict[reviewid]['piclist'].append({'picid': picid, 'pickey': pickey, 'picurl': picurl})
            else:
                refined_dict[reviewid] = {'reviewid': reviewid, 'userid': userid, 'shopid': shopid, 'cityid': cityid,
                                          'shoptype': shoptype, 'reviewbody': reviewbody,
                                          'piclist': [{'picid': picid, 'pickey': pickey, 'picurl': picurl}]}
        # 按行保存处理后的数据
        count = 0
        for reviewid in refined_dict:
            line_fout.write(json.dumps(refined_dict[reviewid], ensure_ascii=False) + '\n')
            count += 1
            if count % 1000 == 0:
                print 'saved %s reviews' % count
                line_fout.flush()
        # dump refined dict
        fout.write(json.dumps(refined_dict, ensure_ascii=False))
        print 'done'


def prepare_splitted_traindata(fin_line_path, fout_path):
    with open(fin_line_path, 'r') as fin, open(fout_path, 'w') as fout:
        count = 0
        for line in fin:
            review_item = json.loads(line.strip())
            review_body = review_item['reviewbody']
            cut_res = jieba.cut(review_body)
            cut_list = [a.encode('utf8') for a in cut_res]
            fout.write(' '.join(cut_list) + '\n')
            count += 1
            if count % 1000 == 0:
                print 'process %s reviews' % count
                fout.flush()


if __name__ == '__main__':
    data_root = '/Users/jayveehe/Jobs/keras_image_caption/datas'
    # process_raw_hive_data('%s/review_pic_data.out' % data_root, '%s/100w_refined_dict.json' % data_root,
    #                       '%s/100w_line_refined.json' % data_root)
    prepare_splitted_traindata('%s/100w_line_refined.json' % data_root, '%s/100w_splitted_train.txt' % data_root)
