#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pandas import MultiIndex, Int16Dtype
from sklearn.model_selection import train_test_split
import joblib
import gc; gc.collect()
from collections import Counter
import joblib
import psutil
import ast

# tokenize
from konlpy.tag import Komoran

# embedding
from gensim.models import Word2Vec

# model
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from config import *
from utils import *

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class GensimWord2VecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, vector_size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None,
                 sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5,
                 ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                 callbacks=(), max_final_vocab=None):
        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.max_final_vocab = max_final_vocab

    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            corpus_file=None,
            vector_size=self.vector_size, alpha=self.alpha, window=self.window, min_count=self.min_count,
            max_vocab_size=self.max_vocab_size, sample=self.sample, seed=self.seed,
            workers=self.workers, min_alpha=self.min_alpha, sg=self.sg, hs=self.hs,
            negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn, epochs=self.iter, null_word=self.null_word,
            trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words,
            compute_loss=self.compute_loss, callbacks=self.callbacks,
            max_final_vocab=self.max_final_vocab)
        
        
        self.model_.build_vocab(X)
        self.model_.train(X, total_examples=self.model_.corpus_count, epochs=self.model_.epochs, report_delay=1)
        
        return self 
    
    def transform(self, X):
        
        X_embeddings = np.array([self._get_embedding(words) for words in X])
        return X_embeddings

    def _get_embedding(self, words):
        valid_words = [word for word in words if word in self.model_.wv.key_to_index]
        if valid_words:
            embedding = np.zeros((len(valid_words), self.vector_size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                embedding[idx] = self.model_.wv[word]

            return np.mean(embedding, axis=0)
        
        else:
            return np.zeros(self.vector_size)

class SentiModelTrain:
        
    def __init__(self, debug=False):
        pass
        
    def get_file(self) :
        __LOG__.Trace("=============== Load Preprocessed Data ===============")
        #predata_path_list = glob.glob(os.path.join(OUT_PATH,'labeled*.xlsx'))
        predata_path_list = glob.glob(os.path.join(OUT_PATH,'*.csv'))
        filst = sorted(predata_path_list, key=os.path.getctime)

        # 최신 저장 전처리 데이터 로드
        __LOG__.Trace(f'Load Data : {filst}')
        #data = pd.read_csv(f'{filst}')
        data = pd.concat([pd.read_csv(file) for file in filst], ignore_index=True)
        
        return data
    
    def data_split(self, X, y, test_size=config['test_size'], random_state=config['random_state']):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=y,
                random_state=random_state
                )
        
        return X_train, X_test, y_train, y_test
   

    def make(self, save_path='./', arg1 = None) :
        
        DIC = {}
        start = datetime.now()
        data = self.get_file()
        
        print(len(data))

        data['total_text_PN'] = data['total_text_PN'].apply(lambda x : ast.literal_eval(x))

        sel_df = data[['total_text_PN', 'label']]
        
        X = sel_df['total_text_PN'].apply(lambda x : ' '.join(x))

        y = sel_df['label']
        
        X_train, X_test, y_train, y_test = self.data_split(X, y)


        gensim_word2vec = GensimWord2VecVectorizer(
                                          vector_size=100,
                                          window=5,
                                          min_count=3,
                                          sg=1,
                                          workers=8,
                                          iter=10
                                          )

        lgbm_wrapper = LGBMClassifier(
            objective='multiclass',
        )
        
        pipeline = Pipeline([
            ('w2v', gensim_word2vec),
            ('lgb', lgbm_wrapper)
        ])
        '''
        params = {
                'lgb__num_leaves' :[20, 30, 50],
                'lgb__max_depth': [-1, 5, 10],
                'lgb__learning_rate': [0.01, 0.05, 0.1],
                'lgb__n_estimators': [1000, 2000],
                'lgb__min_child_samples' : [5, 10, 20],
                "lgb__random_state": [42],
            }
        '''

        param_dict = {
                'lgb__n_estimators': [300, 500, 1000, 2000],
                'lgb__max_depth' : [5, 10, 15, 20],
                'lgb__num_leaves' : [20, 30, 50],
                'lgb__learning_rate' : [0.01, 0.05, 0.1],
                'lgb__min_child_samples' : [5, 10, 20],
                'lgb__random_state': [42]
                }


        '''
        grid = GridSearchCV(
            estimator = Tf_lgb,
            param_grid = params,
            scoring = 'f1_macro',
            cv = 5,
            n_jobs = 12,
            verbose = 3,
            refit=True
        )
        '''
        grid = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dict, n_iter=10, cv=5, verbose=1, n_jobs=12)

        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
        __LOG__.Trace('========================Metric Score============================')

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        accuracy = np.round(accuracy, 4)
        recall = np.round(recall, 4)
        precision = np.round(precision, 4)
        f1 = np.round(f1, 4)

        __LOG__.Trace(f'accuracy: {accuracy}')
        __LOG__.Trace(f'recall: {recall}')
        __LOG__.Trace(f'precision: {precision}')
        __LOG__.Trace(f'f1_score: {f1}')
        
        lgbm_md_file = glob.glob(os.path.join(MODEL_PATH, 'lgbm_model*.pkl'))
        
        if lgbm_md_file != []:
            latest_md = sorted(lgbm_md_file, key=os.path.getctime)[-1]
            lgbm_md_before = joblib.load(latest_md)
        
            lgbm_md_before_score = lgbm_md_before.best_score_
        
            lgbm_md_after_score = grid.best_score_
        
            if lgbm_md_before_score > lgbm_md_after_score:
            
    
                match = re.search(r'\d{4}\d{2}\d{2}', latest_md)
                date_tmp = match.group()
                date = date_tmp + '000000'
        
                DIC[f'{date}'] = {'Tabledf':lgbm_md_before}
        
            else:
                date = datetime.now().strftime('%Y%m%d000000')
                DIC[f'{date}'] = {'Tabledf': grid}
        else:
            DIC['lgbm_model'] = {'Tabledf' : grid}

        return DIC
        
    def make_model(self,fname,data) :
        model_created_date = datetime.now().strftime('%Y%m%d000000')

        filename = os.path.join(MODEL_PATH,f"{fname}_{model_created_date}.model")
        joblib.dump(data,filename)

        return filename

    def make_pkl(self, fname, data) :
        pkl_created_date = datetime.now().strftime('%Y%m%d000000')

        filename = os.path.join(MODEL_PATH, f'lgbm_model_{pkl_created_date}_{fname}.pkl')
        joblib.dump(data,filename)

        return filename
    
    
if __name__ == "__main__":
    pass

