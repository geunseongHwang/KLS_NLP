from config import *
from utils import *
import numpy as np
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy import linalg


import Mobigen.Common.Log_PY3 as Log; Log.Init()


def data_cleansing(x):

        # 특정 문자 처리
        x = re.sub('[◦]', '', x)
        x = re.sub('[\n]', '', x)
        x = x.replace('__x000D', '')
        x = re.sub('[_x000D_]', "", x)
        

        # 빈칸 제거
        x = re.sub(r"^\s+|\s+$", "", x, flags=re.UNICODE)
        
        return x

class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        
        self.sentences = self.sent_tokenize.text2sentences(text)
        
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        
        self.graph_matrix = GraphMatrix()

        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        index.sort()
        for idx in index:
            self.sentences = [tmp for tmp in self.sentences if tmp != '']
            summary.append(self.sentences[idx])
        
        return summary
    

class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.Komoran = Komoran()

  
    def text2sentences(self, text):
        # print('kkma start')
        sentences = self.kkma.sentences(text)

        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
        
        return sentences

    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.Komoran.nouns(str(sentence)) 
                                       if noun != len(noun) > 1]))
        
        return nouns
    
class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
        
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return  self.graph_sentence
        
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}
    
class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        
        matrix_size = A.shape[0]
        
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로 
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
            
        B = (1-d) * np.ones((matrix_size, 1))
        # ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        ranks = linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}

def summurize(x, num=1):
    
    x = x.strip()
    if x != '':
        textrank = TextRank(x)
        # print('TextRank done')
        text_smr = textrank.summarize(num)[0]
        # print('TextRank summarize done')
        return text_smr

def morpheme_analysis(x):

    # 형태소분석 - Komoran
    x = x.strip()
    
    words = []
    
    kon = Komoran()
    if x != '':
        morpheme_analysis = kon.pos(x)
        for v in morpheme_analysis:
            if v[1] == 'NNG' or v[1] == 'NNP' or v[1] == 'NNB':
                words+=[v[0]]

    return words

def word_count(x):
    x = [w for w in x if len(w) > 1]
    ct_x = Counter(x).most_common()
    word_x = [i[0] for i in ct_x][:3]
    return word_x


class SurveySummary:
    def __init__(self, debug=False):
        self.debug = debug
        #self.logger = get_logger(os.path.basename("Data Load and Preprocess"))
        #self.fname = 'preprocessing'


    def make(self, save_path='./', arg1 = None):
        self.save_path = save_path
        DIC = {}
        start = datetime.now()
        
        #__LOG__.Trace = get_logger(os.path.basename("Data Load and Preprocess"))
        
        # data load
        start = datetime.now().replace(microsecond = 0 )
        __LOG__.Trace("=============== Data Load ===============")
        __LOG__.Trace(f"[start] : {str(start)}")
        __LOG__.Trace(f"[DATA_PATH] : {DATA_PATH}")

    
        FILE_NAME = 'DL_QUSTNR_EXMN_ANS.csv'
        load_data = os.path.join(DATA_PATH, FILE_NAME)
        
        data = pd.read_csv(load_data)
        # data = data.iloc[:20,:]
        
        __LOG__.Trace(f"[DATA_SHAPE] : {data.shape}")
        __LOG__.Trace("Data Load finished")

        __LOG__.Trace(f"[DATA_SHAPE] : {data.shape}")


        __LOG__.Trace("=============== Fill Null ===============")
        data['SBJCT_ANS'].fillna('-', inplace=True)

        __LOG__.Trace("=============== GroupBy ==============")
        df_g = data.groupby(['DEPT_NM', 'EXMN_NM', 'QESTN_NM', 'ANS_EX_NM'])['SBJCT_ANS'].count().reset_index()
        
        __LOG__.Trace("=============== Data Cleansing ===============")
        data['SBJCT_ANS'] = data['SBJCT_ANS'].apply(data_cleansing)

        df_sub = data.groupby(['DEPT_NM', 'EXMN_NM', 'QESTN_NM', 'ANS_EX_NM'])['SBJCT_ANS'].apply(lambda x: ' '.join([tmp for tmp in x if tmp != '-'])).reset_index()['SBJCT_ANS']

        __LOG__.Trace("=============== Summurize ===============")
        df_sub = df_sub.apply(summurize)

        tmp_df = data.groupby(['DEPT_NM', 'EXMN_NM', 'QESTN_NM', 'ANS_EX_NM'])['SBJCT_ANS'].apply(lambda x : " ".join(x)).reset_index()

        __LOG__.Trace("=============== Morpheme Analysis ===============")
        tmp_df['SBJCT_ANS'] = tmp_df['SBJCT_ANS'].apply(morpheme_analysis)
        
        __LOG__.Trace("=============== Word Counting ===============")
        tmp_count = tmp_df['SBJCT_ANS'].apply(word_count)

        __LOG__.Trace("=============== Create keywords ===============")
        list_count = tmp_count.to_list()
        kw_df = pd.DataFrame(list_count, columns=['keyword1', 'keyword2', 'keyword3'])

        __LOG__.Trace("=============== Concatenate ===============")
        results = pd.concat([df_g, df_sub, kw_df], axis=1)
        results.insert(0, 'EXMN_DT', '2022-09-01')
        
        __LOG__.Trace("=============== Finished ===============")
        
        __LOG__.Trace(f"[DATA_COLUMNS] : \n {results.columns.tolist()}")
        __LOG__.Trace(f"[DATA_SHAPE] : {results.shape}")
        
        end = datetime.now()
        sec = end - start
        times = str(timedelta(seconds=sec.seconds)).split(".")
        __LOG__.Trace(f"=============== Time elapsed: {times[0]} ===============")

        DIC['TB_AN4_QUSTNR_RESULT_INFO'] = {'Tabledf':results}
        

        return DIC

    def make_csv(self, fname, data) :
        now = datetime.now().strftime('%Y%m%d000000')

        if data.shape[0] == 0 :
            __LOG__.Trace('No Data to write csv file')
            return
        else : 
            try :
                out_fname = os.path.join(self.save_path, fname+'_'+now+'.csv')
                data.to_csv(out_fname, index=False, header=True)
                return out_fname
            except :
                __LOG__.Trace('Fail to Write DF to CSV file')
                raise Exception('Fail to Write DF to CSV file')


if __name__ == "__main__":

    data_reg_dt = time.strftime('%Y%m%d%H%M%S')
    result = SurveySummary(debug=True).make('./', data_reg_dt)