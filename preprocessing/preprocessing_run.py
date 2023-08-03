from config import *
from utils import *

from multiprocessing import Process, Pool
import Mobigen.API.M6_PY3 as M6
import ast
import Mobigen.Common.Log_PY3 as Log; Log.Init()


class Preprocessing :
    def __init__(self, debug=False) :
        self.debug = debug
        self.sql = """
	 select *
         from KALIS.DL_MSMDA_CNTNTS_DALY
         where WRT_DTTM != ''
         group by MSMDA, SRCHWRD, MSMDA_URL
         having substr(WRT_DTTM, 1, 10) = strftime('%Y%m%d%H', 'now', '+8 hours')
         and max(WRT_DTTM)
         order by MSMDA,SRCHWRD,MSMDA_URL,WRT_DTTM desc;
 
        """

    def getData(self, sql):
        __LOG__.Trace('========== Data Load Start ==============')
        try :
               
            conn = M6.Connection('192.168.11.23:5050','kalis','Kalis4114!' ,Database='KALIS')
            curs = conn.Cursor()
            curs.Execute2( sql )
            meta_data = curs.Metadata()

            columns = meta_data['ColumnName']
            data = [row for row in curs]

            curs.Close()
            conn.close()
            
            return pd.DataFrame(data=data, columns=columns)
        except :
            __LOG__.Exception()


    def preprocess(self,read_file):

        # load Data
        load_file = os.path.join(SAVE_PATH, read_file)
        with open(load_file, 'rb')as f:
            data = pickle.load(f) 
        
        # 이모지 처리
        data['total_text'] = data['total_text'].apply(imogi_process)
        
        # 특수문자 처리
        data['total_text'] = data['total_text'].apply(special_char_process)
        
        # 맞춤법 교정
        #data['total_text'] = data['total_text'].apply(check_word)
        
        # 한글 추출
        data['total_text'] = data['total_text'].apply(extract_korean)
        
        # 형태소분석 ( 불용어제거, 1글자 제거 )
        data['total_text'] = data['total_text'].apply(morpheme_analysis)
         
        # 긍부정용 텍스트 데이터 추출
        data['total_text_PN'] = data['total_text'].apply(extract_word)

        # 연관어용 텍스트 데이터 추출
        data['total_text'] = data['total_text'].apply(extract_word2)        
        
        save_file = 'preprocessed_dataset_' + str(read_file[-6:])   
        save_path = os.path.join(SAVE_PATH, save_file)

        with open(save_path, 'wb')as f:
            pickle.dump(data, f)  

    def make(self, save_path='./', arg1 = None):
        self.save_path = save_path
        start = datetime.now()
        
        __LOG__.Trace("=============== Data Load and Preprocess ===============")
        
        # data load
        #data = data_load()
        data = self.getData(self.sql)
        
        __LOG__.Trace(data)
        
        # sampling                        -> 체크 후 제거
        #__LOG__.Trace("=============== Sampling ===============")

        __LOG__.Trace(f"[Sampled DATA_SHAPE] : {data.shape}")
        
        # title + text (instagram - text) 
        __LOG__.Trace("=============== title + text start ===============")
        
        data.loc[data['MSMDA']!="instagram","total_text"] = data['NTT_SJ'] + ' ' + data['NTT_CN']
        data.loc[data['MSMDA']=="instagram","total_text"] = data['NTT_CN']
        
        __LOG__.Trace("=============== title + text Done. ===============")
        
        __LOG__.Trace("=============== Multithreading Preprocess ===============")
        
        k = 12 #core count   
        df_split(data, k)
        
        files = []
        for j in range(k):
            file = "dataset_{}.pkl".format(str(j).zfill(2))
            files.append(file)  


        n_worker = k
        try:
            p = Pool(processes = n_worker)
            p.map(self.preprocess, files)
        except : 
            __LOG__.Exception()
        finally:
            p.close()
            p.join

        
        results = pd.DataFrame()
        for i in range(k):
            file = f"preprocessed_dataset_{str(i).zfill(2)}.pkl"
            path = os.path.join(SAVE_PATH, file)

            with open(path, 'rb') as f:
                df = pickle.load(f)
            results = pd.concat([results, df], axis=0)
        
        
        file_name = "preprocessed_data.xlsx"
        file_save = os.path.join(OUT_PATH, file_name)
        
        #results.to_excel(file_save)
        
        __LOG__.Trace("=============== Multithreading Preprocess Done. ===============")
        
        # 라벨링
        __LOG__.Trace("=============== Labeling ===============")
        
        # 감성사전 
        __LOG__.Trace("=============== Check the words included in the sentiment dictionary ===============")
        
        
        results['total_text_PN_label'] = results['total_text_PN'].apply(morpheme_extraction)
        
        # 개수별 라벨태그 
        __LOG__.Trace("=============== Label tags by count included in the sentiment dictionary ===============")
        results['label'] = results['total_text_PN_label'].apply(labeling_by_count)
        
        results = results.drop(columns='total_text_PN_label')
        
        __LOG__.Trace(f"[LABEL_COUNT] : \n {results['label'].value_counts()}")
        
        
        file_name = "labeled_preprocessed_data.xlsx"
        file_save = os.path.join(OUT_PATH, file_name)
        
        #results.to_excel(file_save)
        
        
        end = datetime.now()
        sec = end - start
        times = str(timedelta(seconds=sec.seconds)).split(".")
        
        __LOG__.Trace("=============== Data Load and Preprocess Finished ===============")
        
        __LOG__.Trace(f"[DATA_COLUMNS] : \n {results.columns.tolist()}")
        __LOG__.Trace(f"[DATA_SHAPE] : {results.shape}")
        
        __LOG__.Trace(f"=============== Time elapsed: {times[0]} ===============")

        inpdict = {}

        inpdict['Tabledf'] = results
        

        return {'preprocessing':inpdict}

    def make_csv(self,fname,data) :
        now = datetime.now().strftime('%Y%m%d%H%M%S')

        if data.shape[0] == 0 :
            __LOG__.Trace('No Data to write csv file')
            return
        else : 
            try :
                out_fname = os.path.join(self.save_path,fname+'_'+now+'.csv')
                print(out_fname)
                data.to_csv(out_fname, index=False, header=True)
                return out_fname
            except :
                __LOG__.Trace('Fail to Write DF to CSV file')
                raise Exception('Fail to Write DF to CSV file')


if __name__ == "__main__":

    data_reg_dt = time.strftime('%Y%m%d%H%M%S')
    result = Preprocessing(debug=True).make('./', data_reg_dt)
    result['preprocessing']['Tabledf'].to_csv('sample.csv', index=False, header=True)

