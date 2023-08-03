#!/usr/bin/env python
# coding: utf-8
import numpy as np
import joblib
import gc; gc.collect()
from collections import Counter
import ast

from sklearn.model_selection import train_test_split

from utils import *
from config import *

class SentiDataMart :
    def __init__(self, debug=False) :
        pass  

    def get_file(self, path, patn) :

        __LOG__.Trace("=============== Load Data ===============")
        #predata_path_list = glob.glob(os.path.join(OUT_PATH,'labeled*.xlsx'))
        path_list = glob.glob(os.path.join(path,patn))
        filst = sorted(path_list, key=os.path.getctime)

        # 최신 저장 전처리 데이터 로드
        #data = pd.read_excel(f'{filst[-1]}')
        __LOG__.Trace(f'Load Data : {filst[-1]}')

        return filst[-1]

    def word_pred(self, df, snt_model):

        real_fin_df = pd.DataFrame()

        for num, row in df.iterrows():
            tmp_df = pd.DataFrame()

            for word in row['total_text_PN']:
                try:
                    #word_wv = wv_model.wv[word].reshape(1, -1)
                    pred_word = snt_model.predict([word])

                    df_reshape = pd.DataFrame(row.values.reshape(1, -1), columns=df.columns)

                    df_reshape['word'] = word
                    df_reshape['pred_word'] = pred_word

                    tmp_df = pd.concat([tmp_df, df_reshape])

                except:
                    # print(f'{word}에 대한 분석결과가 없습니다.')
                    continue

            if not [] == list(tmp_df.values):

                data_list = tmp_df['word'].tolist()
                count_list = Counter(data_list).most_common()
                word_list = [tup[0] for tup in count_list if len(tup[0]) > 1]
                real_count_list = [tup[1] for tup in count_list if len(tup[0]) > 1]
                tmp_df = tmp_df[['WRT_DTTM', 'SRCHWRD', 'MSMDA', 'word', 'pred_word']]
                fin_df = pd.DataFrame({"word" : word_list,  "count" : real_count_list})
                meg_df = pd.merge(tmp_df, fin_df)
                real_fin_df = pd.concat([real_fin_df, meg_df])

        real_fin_df.reset_index(drop=True, inplace=True)

        return real_fin_df

    def weekinmonth(self,dates):
        
        firstday_in_month = dates - pd.to_timedelta(dates.dt.day - 1, unit='d')
        return (dates.dt.day-1 + firstday_in_month.dt.weekday) // 7 + 1

    def make(self, save_path='./', arg1 = None) :
        self.save_path = save_path
        DIC = {}

        start = datetime.now()
    
        __LOG__.Trace("=============== Load Preprocessed Data ===============")
        predata_path_list = glob.glob(os.path.join(SAVE_PATH,'labeled*.xlsx'))
        filst = sorted(predata_path_list, key=os.path.getctime)
        
        # 최신 저장 전처리 데이터 로드
        data = pd.read_csv(self.get_file(OUT_PATH,'preprocessing_*.csv'))
        __LOG__.Trace(f"[DATA_SHAPE] : \n {data.shape}")
        
    
        # 한글지 이상의 단어만 사용
        data['total_text_PN'] = data['total_text_PN'].apply(lambda x : ast.literal_eval(x))
        data['total_text_PN'] = data['total_text_PN'].apply(lambda x : [''.join(tmp) for tmp in x if len(tmp) > 1])

        data = data[['WRT_DTTM', 'SRCHWRD', 'MSMDA', 'total_text_PN']]
        data = data.dropna()
        data = data[data['total_text_PN'] != '[]']
    
        __LOG__.Trace("=============== Load Model ===============")
        # 최신 저장 word2vec 모델 로드
        #wv_model = joblib.load(self.get_file(MODEL_PATH,'Senti_word2vec*.model'))
        # 최신 저장 sentiment 모델 로드
        snt_model = joblib.load(self.get_file(MODEL_PATH,'lgbm_model*.pkl'))                    
         
        __LOG__.Trace("=============== Start DataMart Creation ===============")
        
        real_fin_df = self.word_pred(data, snt_model)

        real_fin_df['pred_word'] = real_fin_df['pred_word'].replace(0, '중립').replace(1, '부정').replace(2, '긍정')
                                        
        one_fin_df = real_fin_df.sort_values(by='WRT_DTTM').reset_index(drop=True)
        two_fin_df = one_fin_df.groupby(['WRT_DTTM', 'SRCHWRD', 'MSMDA', 'word', 'pred_word'], sort=False, as_index=False).agg({"count":"sum"})                 
        
        two_fin_df['WRT_DTTM'] = pd.to_datetime(two_fin_df["WRT_DTTM"], format='%Y%m%d%H%M%S')
        
        two_fin_df['WRT_DTTM'] = two_fin_df.WRT_DTTM.map(lambda x: x.strftime('%Y-%m-%d %H:00:00'))
        
        two_fin_df.loc[:, 'week_of_month'] = self.weekinmonth(two_fin_df.loc[:, 'WRT_DTTM'].astype('datetime64'))
        two_fin_df.loc[:, 'WRT_DTTM'] = two_fin_df.loc[:, 'WRT_DTTM'].astype(str)
        two_fin_df.loc[:, 'tmp_dt'] = two_fin_df.loc[:, 'WRT_DTTM'] + '-' + two_fin_df.loc[:, 'week_of_month'].astype(str)
        two_fin_df.loc[:, 'week_in_month'] = two_fin_df.loc[:, 'tmp_dt'].apply(lambda x: x[:4]+'년 '+ x[5:7]+'월 '+ x[-1]+'주차')
        two_fin_df = two_fin_df.loc[:, ['WRT_DTTM', 'SRCHWRD', 'MSMDA', 'word', 'pred_word', 'count', 'week_in_month']]
        two_fin_df.columns = ['PSTG_DT', 'SRCH_WRD', 'MEDIA', 'SEMSE_WRD', 'SEMSE_TYPE', 'SEMSE_WRD_CO', 'MT_WEEK']
        
        __LOG__.Trace("=============== DataMart Creation Finished ===============")
        #two_fin_df.to_csv('DM_AN3_SEMSE_WRD_INFO.csv', index=False)                                
                                        
        end = datetime.now()
        sec = end - start
        times = str(timedelta(seconds=sec.seconds)).split(".")
        __LOG__.Trace(f"=============== Time elapsed: {times[0]} ===============")

        DIC['TB_AN3_SEMSE_WRD_INFO'] = {'Tabledf':two_fin_df}
        return DIC

    def make_csv(self, fname, data) :
        now = datetime.now().strftime('%Y%m%d%H%M%S')

        if data.shape[0] == 0 :
            __LOG__.Trace('No Data to write csv file')
            return
        else : 
            out_fname = os.path.join(self.save_path,fname+'_'+now+'.csv')
            data.to_csv(out_fname, index=False, header=True)
            return out_fname


if __name__ == "__main__":
   pass 
                                    
