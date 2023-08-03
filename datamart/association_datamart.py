import glob, os
import numpy as np
from datetime import datetime, timedelta
from gensim.models.word2vec import Word2Vec

from utils import *
from config import *

class AssDataMart :
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


    # 일자 추출 함수
    def to_dt(self,x):
        try:
            x = datetime.strptime(x,'%Y%m%d%H%M%S')
            x = datetime.strftime(x, '%Y-%m-%d %H:00:00')
            #x = datetime.strptime(x, '%Y-%m-%d')
        except ValueError:  # ex) 리공사.-20-22 00:00:00 
            x = np.nan
        except TypeError:
            x = np.nan
        return x

    def get_date(self,y, m, d):
        s = f'{y:04d}-{m:02d}-{d:02d}'
        return datetime.strptime(s, '%Y-%m-%d')

    def get_week_no(self,y, m, d):
        target = self.get_date(y, m, d)
        firstday = target.replace(day=1)
        if firstday.weekday() == 6:
            origin = firstday
        elif firstday.weekday() < 3:
            origin = firstday - timedelta(days=firstday.weekday() + 1)
        else:
            origin = firstday + timedelta(days=6-firstday.weekday())
        return (target - origin).days // 7 + 1

    def to_dt_w(self,x):
        try:
            
            x = datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date()
        
            yr = x.year
            m = x.month
            d = x.day
            x = self.get_week_no(yr, m, d)
            x = str(yr) + '년 ' + str(m) + '월 ' + str(x) + "주차"
        except ValueError:  # ex) 리공사.-20-22 00:00:00 
            x = np.nan
        except TypeError:
            x = np.nan
        return str(x)

    def relation_score(self, data, date, domain, MODEL, results):
        
        data = data[(data['published_ymd']==date)&(data['MSMDA']==domain)]
        
        keyword = ["건설","안전","사고","사망","현장","시설","붕괴","지진","정밀","진단","점검","씽크홀","싱크홀","국토안전관리원","한국시설안전공단","국토부","중처법","건설법","중대재해","처벌법","특별법","스마트건설","건설관리공사","공공기관","하자","심사","분쟁조정","건설사고","시설물사고","시설사고"]
        
        for key in keyword:
            res = pd.DataFrame(columns=['PSTG_DT','SRCH_WRD','MEDIA','RELATE_WRD','RELATE_SCORE','RELATE_RANK','COLOR','MT_WEEK'])
            
            try:
                ms = MODEL.wv.most_similar([f'{key}'], topn=25)
                
                relate_wrd = []
                relate_score = []
                ranks = []
                rank=0
                for i, j in ms:
                    rank+=1
                    ranks.append(rank)
                    relate_wrd.append(i)
                    relate_score.append(int(round(j,2)*100))

                res['RELATE_WRD'] = relate_wrd 
                res['RELATE_RANK'] = ranks 
                res['RELATE_SCORE'] = relate_score
                res['PSTG_DT'] = date
                res['SRCH_WRD'] = key
                res['MEDIA'] = domain
                
                results = pd.concat([results, res], axis=0)
                
            except KeyError:
                print(f'{key}에 대한 분석결과가 없습니다.')
                continue
       
        return results

    def result_mart(self, data, date, domain, model):
        results = pd.DataFrame(columns=['PSTG_DT','SRCH_WRD','MEDIA','RELATE_WRD','RELATE_SCORE','RELATE_RANK','COLOR','MT_WEEK'])
        
        #d = datetime.today() + timedelta(hours=9)
        #print(date)
        #d = datetime.strptime(date, '%Y%m%d %H:%M:%S') 
        #d = d.strftime('%Y-%m-%d %H:00:00')

        for i in domain:
            #date
            for j in date:
            #for j in date:
                results = self.relation_score(data, j, i, model, results)
                
        return results

    def make(self, save_path='./', arg1 = None) :
        self.save_path = save_path
        DIC = {}
        start = datetime.now()

        data = pd.read_csv(self.get_file(OUT_PATH,'preprocessing_*.csv'))

        __LOG__.Trace(f"[DATA_SHAPE] : \n {data.shape}")
        
        model = Word2Vec.load(self.get_file(MODEL_PATH,'word2vec*.model'))
        data['WRT_DTTM'] = data['WRT_DTTM'].astype(str)
        
        data['published_ymd'] = data['WRT_DTTM'].apply(self.to_dt)

        data = data[data['published_ymd'].notnull()]
        date = np.sort(data['published_ymd'].unique())
        domain = ['youtube','instagram','google-news','twitter']
        
                
        color_dicts={0:'#CB62E9',1:'#CB62E9',2:'#CB62E9',3:'#CB62E9',4:'#CB62E9',5:'#CB62E9',6:'#CB62E9',7:'#CB62E9',8:'#CB62E9',9:'#CB62E9',10:'#A36EF9',11:'#A36EF9',12:'#A36EF9',13:'#A36EF9',14:'#A36EF9',15:'#A36EF9',16:'#A36EF9',17:'#A36EF9',18:'#A36EF9',19:'#A36EF9',20:'#7A87FF',21:'#7A87FF',22:'#7A87FF',23:'#7A87FF',24:'#7A87FF',25:'#7A87FF',26:'#7A87FF',27:'#7A87FF',28:'#7A87FF',29:'#7A87FF',30:'#51AEFF',31:'#51AEFF',32:'#51AEFF',33:'#51AEFF',34:'#51AEFF',35:'#51AEFF',36:'#51AEFF',37:'#51AEFF',38:'#51AEFF',39:'#51AEFF',40:'#7CCF8F',41:'#7CCF8F',42:'#7CCF8F',43:'#7CCF8F',44:'#7CCF8F',45:'#7CCF8F',46:'#7CCF8F',47:'#7CCF8F',48:'#7CCF8F',49:'#7CCF8F',50:'#B2DB4C',51:'#B2DB4C',52:'#B2DB4C',53:'#B2DB4C',54:'#B2DB4C',55:'#B2DB4C',56:'#B2DB4C',57:'#B2DB4C',58:'#B2DB4C',59:'#B2DB4C',60:'#F9CA4C',61:'#F9CA4C',62:'#F9CA4C',63:'#F9CA4C',64:'#F9CA4C',65:'#F9CA4C',66:'#F9CA4C',67:'#F9CA4C',68:'#F9CA4C',69:'#F9CA4C',70:'#FAAE4C',71:'#FAAE4C',72:'#FAAE4C',73:'#FAAE4C',74:'#FAAE4C',75:'#FAAE4C',76:'#FAAE4C',77:'#FAAE4C',78:'#FAAE4C',79:'#FAAE4C',80:'#FA8D4E',81:'#FA8D4E',82:'#FA8D4E',83:'#FA8D4E',84:'#FA8D4E',85:'#FA8D4E',86:'#FA8D4E',87:'#FA8D4E',88:'#FA8D4E',89:'#FA8D4E',90:'#E96462',91:'#E96462',92:'#E96462',93:'#E96462',94:'#E96462',95:'#E96462',96:'#E96462',97:'#E96462',98:'#E96462',99:'#E96462',100:'#E96462'}

        __LOG__.Trace("=============== Start DataMart Creation ===============")
        results = self.result_mart(data, date, domain, model)

        results['COLOR'] = results['RELATE_SCORE'].apply(lambda x : color_dicts[x])
        
        results['MT_WEEK'] = results['PSTG_DT'].apply(self.to_dt_w)
        
        __LOG__.Trace(f"[DM Result_SHAPE] : \n  {results.shape}")
        
        __LOG__.Trace("=============== DataMart Creation Finished ===============")
        #results.to_csv('DM_AN3_SRCH_RLT_INFO.csv', index=False)
        DIC['TB_AN3_SRCH_RLT_INFO'] = {'Tabledf':results}
        
        end = datetime.now()
        sec = end - start
        times = str(timedelta(seconds=sec.seconds)).split(".")
        __LOG__.Trace(f"=============== Time elapsed: {times[0]} ===============")

        return DIC

    def make_csv(self, fname, data) :
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        print('data head:',data.head())
        if data.shape[0] == 0 :
            __LOG__.Trace('No Data to write csv file')
            return
        else : 
            out_fname = os.path.join(self.save_path,fname+'_'+now+'.csv')
            data.to_csv(out_fname, index=False, header=True)
            return out_fname



if __name__ == "__main__":
    pass
