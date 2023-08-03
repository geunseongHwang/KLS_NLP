import ast

from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import FastText

from config import *
from utils import *
import Mobigen.Common.Log_PY3 as Log; Log.Init()

class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

class AssModelTrain :
    def __init__(self, debug=False) :
        self.fname = ''

    def get_file(self) :
        __LOG__.Trace("=============== Load Preprocessed Data ===============")
        #predata_path_list = glob.glob(os.path.join(OUT_PATH,'labeled*.xlsx'))
        predata_path_list = glob.glob(os.path.join(OUT_PATH,'*.csv'))
        filst = sorted(predata_path_list, key=os.path.getctime)
        
        # 최신 저장 전처리 데이터 로드
        #data = pd.read_excel(f'{filst[-1]}')
        __LOG__.Trace(f'Load Data : {filst[-1]}')
        data = pd.read_csv(f'{filst[-1]}')
        #data = pd.concat([pd.read_csv(file) for file in filst], ignore_index=True)

        return data

    def make(self, save_path='./', arg1 = None) :

        start = datetime.now()
        data = self.get_file()

        data = data[data['total_text'].notnull()]
        
        data['total_text'] = data['total_text'].apply(lambda x : ast.literal_eval(x) if isinstance(x, str) else x)
        
        sentences = list(data['total_text'])
        
    
        model_path_list = glob.glob(os.path.join(MODEL_PATH,'word2vec*.model'))
        
        if len(model_path_list)==0:
        
            # 모델 학습
            model = FastText(sentences=sentences
                            , vector_size = 100
                            , window = 5  
                            , min_count = 1
                            , workers = 3
                            , sg = 1
                            , hs = 0
                            , ns_exponent = 0.75
                            #, compute_loss=True
                            )
        
            model.build_vocab(sentences)
            total_example = model.corpus_count
            model.train(sentences, total_examples=total_example, epochs=20) 

            # 모델 저장
            #model.save(os.path.join(MODEL_PATH, f"word2vec_{model_created_date}.model"))
        else :
            filst = sorted(model_path_list, key=os.path.getctime)
            
            # 최신 저장 모델 로드
            model = FastText.load(f"{filst[-1]}")
            
            # 최신 데이터로 재학습
            model.build_vocab(sentences, update = True)
            total_examples = model.corpus_count
            model.train(sentences, total_examples=total_examples, epochs=100, callbacks=[callback()])
        
            # 모델 저장
            #model.save(os.path.join(MODEL_PATH, f"word2vec_{model_created_date}.model"))

        end = datetime.now()
        sec = end - start
        times = str(timedelta(seconds=sec.seconds)).split(".")
        
        __LOG__.Trace("=============== Model Training Finished ===============")
        __LOG__.Trace(f"=============== Time elapsed: {times[0]} ===============")

        DIC = {}

        inpdict = {}
        inpdict['Tabledf'] = model
        DIC['word2vec'] = inpdict

        return DIC

    def make_model(self,fname,data) :
        model_created_date = datetime.now().strftime('%Y%m%d000000')

        filename = os.path.join(MODEL_PATH,f"{fname}_{model_created_date}.model")
        data.save(filename)

        return filename

if __name__== '__main__':
    
    data_reg_dt = time.strftime('%Y%m%d%H%M%S')
    result = AssModelTrain(debug=True).make('./', data_reg_dt)
