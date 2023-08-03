from config import *

import json
import glob
import pickle
#import psutil

import logging
import argparse

import pandas as pd
from datetime import datetime, timedelta
import time
import re
from itertools import accumulate

from konlpy.tag import Komoran
from hanspell import spell_checker
from collections import Counter

import Mobigen.Common.Log_PY3 as Log; Log.Init()

######## LOGGING
def get_logger(cls_name):
    logger = logging.getLogger(cls_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(LOG_PATH, cls_name.split('.')[0] + '.log'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    
    return logger

######## DATA LOAD
def data_load():

    start = datetime.now().replace(microsecond = 0 )
    __LOG__.Trace("=============== Data Load ===============")
    __LOG__.Trace(f"[start] : {str(start)}")
    __LOG__.Trace(f"[DATA_PATH] : {DATA_PATH}")
    
    # data path 
    load_data = os.path.join(DATA_PATH, FILE_NAME)
    
    data = pd.read_excel(load_data)
    
    __LOG__.Trace(f"[DATA_SHAPE] : {data.shape}")
    __LOG__.Trace("Data Load finished")
    return data


######## DATA PREPROCESS
def df_split(df, k):
    leng = len(df)
    share = leng // k
    reminder = leng % k
    result = [share] * k
    
    for i in range(reminder):
        result[k - i - 1] += 1
    
    result = list(accumulate(result))
    
    k_cnt = 0
    for j, i in enumerate(result):
        gf = df.iloc[k_cnt:i,]
        k_cnt = i
                
        file_name = f"dataset_{str(j).zfill(2)}.pkl"        
        save_path = os.path.join(SAVE_PATH, file_name)  
                
        with open(save_path, 'wb') as f:
            pickle.dump(gf, f)
                             
            
def imogi_process(x):
    try :
        emoji_pattern = re.compile("[" 
                                    u"\U0001F600-\U0001F64F"  
                                    u"\U0001F300-\U0001F5FF"  
                                    u"\U0001F680-\U0001F6FF"  
                                    u"\U0001F1E0-\U0001F1FF"  
                                    u"\u2600-\u2B55" 
                                    u"\u25A0-\u25FF"  
                                                           "]+", flags=re.UNICODE)

        x = emoji_pattern.sub(r'', x)
    except :  # nan, float type  
        pass

    return x

def special_char_process(x):
    try:
        x = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', x)
    
    except : # nan, float type 
        pass
    
    return x

def check_word(x):
    
    try:
        # assert type(x) == str
        s = spell_checker.check(x)
        if s.checked == "" :
            x = x
        else:
            x = s.checked
    except: 
        x = x
    return x

def extract_korean(x):
    try:
        x = re.sub(r'[^가-힣]', ' ', x)
    
    except : # nan, float type 
        pass
    
    return x



def morpheme_analysis(x):  
    try:
        with open(os.path.join(DATA_PATH,"stopword_correction.txt"), 'r') as f:
            stopword_list = f.read().split('\n')

        dics = os.path.join(DATA_PATH,'사용자정의사전.txt')

        # 형태소분석 - Komoran
        words = []
        if x.strip() != '':
            kon = Komoran(userdic = dics)
            morpheme_anal = kon.pos(x)
            
            for key, value in morpheme_anal.items():
                
                if (key not in stopword_list) and (len(key) > 1):
                    # words += [v[0]]
                    words += [key, value]
    except:
        pass
    
    return words


# 연관어용
def extract_word2(x):
    global words, a
    pos = ['NNG', 'NNP', 'NNB']
    try:
        x = dict(x)
        words = []
        for key, value in x.items():
            if value in pos:
                words.append(key)
    except SyntaxError:
        pass
    return words


# 긍부정용
def extract_word(x):
    global words, a
    pos = ['NNG', 'NNP', 'NNB', 'VV', 'VA']
    try:
        x = dict(x)
        words = []
        for key, value in x.items():
            if value in pos:
                words.append(key)
    except SyntaxError:
        pass
    return words


######## LABELING

def morpheme_extraction(x):
    with open(os.path.join(DATA_PATH,'SentiWord_info.json'), encoding='utf-8-sig', mode='r') as f:
        senti_dic = json.load(f)
    
    dic_list = [x['word'] for x in senti_dic]
    snt_list = [x['polarity'] for x in senti_dic]
    
    data = [(word, count) for word, count in zip(dic_list, snt_list) if word in x]
    
    return data

def labeling_by_count(x):
    
    dic = {'neg' : 0, 'pos' : 0, 'neu' : 0}
    
    if x == []:
        dic = ('neu', 0)
        
        x = dic[0].replace('neu', '0')
    else:
        for _, label in x:
            if (label == '-2') or (label == '-1'):
                dic['neg'] += 1
            elif (label == '1') or (label == '2'):
                dic['pos'] += 1
            else:
                dic['neu'] += 1
    
        dic = max(dic.items(), key = lambda x: x[1])
        
        x = dic[0].replace('neu', '0').replace('neg', '1').replace('pos', '2')
    return x
