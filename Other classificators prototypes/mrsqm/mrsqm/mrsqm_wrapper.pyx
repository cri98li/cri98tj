from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

from tqdm.auto import tqdm

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold



import logging

def debug_logging(message):
    logging.info(message)

def from_nested_to_2d_array(X, return_numpy=False):
    """Convert nested Panel to 2D numpy Panel.

    Convert nested pandas DataFrame or Series with NumPy arrays or
    pandas Series in cells into tabular
    pandas DataFrame with primitives in cells, i.e. a data frame with the
    same number of rows as the input data and
    as many columns as there are observations in the nested series. Requires
    series to be have the same index.

    Parameters
    ----------
    X : nested pd.DataFrame or nested pd.Series
    return_numpy : bool, default = False
        - If True, returns a NumPy array of the tabular data.
        - If False, returns a pandas DataFrame with row and column names.

    Returns
    -------
     Xt : pandas DataFrame
        Transformed DataFrame in tabular format
    """
    # TODO does not handle dataframes with nested series columns *and*
    #  standard columns containing only primitives

    # convert nested data into tabular data
    if isinstance(X, pd.Series):
        Xt = np.array(X.tolist())

    elif isinstance(X, pd.DataFrame):
        try:
            Xt = np.hstack([X.iloc[:, i].tolist() for i in range(X.shape[1])])

        # except strange key error for specific case
        except KeyError:
            if (X.shape == (1, 1)) and (X.iloc[0, 0].shape == (1,)):
                # in fact only breaks when an additional condition is met,
                # namely that the index of the time series of a single value
                # does not start with 0, e.g. pd.RangeIndex(9, 10) as is the
                # case in forecasting
                Xt = X.iloc[0, 0].values
            else:
                raise

        if Xt.ndim != 2:
            raise ValueError(
                "Tabularization failed, it's possible that not "
                "all series were of equal length"
            )

    else:
        raise ValueError(
            f"Expected input is pandas Series or pandas DataFrame, "
            f"but found: {type(X)}"
        )

    if return_numpy:
        return Xt

    Xt = pd.DataFrame(Xt)

    # create column names from time index
    if X.ndim == 1:
        time_index = (
            X.iloc[0].index
            if hasattr(X.iloc[0], "index")
            else np.arange(X.iloc[0].shape[0])
        )
        columns = [f"{X.name}__{i}" for i in time_index]

    else:
        columns = []
        for colname, col in X.items():
            time_index = (
                col.iloc[0].index
                if hasattr(col.iloc[0], "index")
                else np.arange(col.iloc[0].shape[0])
            )
            columns.extend([f"{colname}__{i}" for i in time_index])

    Xt.index = X.index
    Xt.columns = columns
    return Xt



######################### SAX #########################

cdef extern from "sax_converter.h":
    cdef cppclass SAX:
        SAX(int, int, int, int)        
        vector[string] timeseries2SAX(vector[double])
        vector[double] map_weighted_patterns(vector[double], vector[string], vector[double])

cdef class PySAX:
    '''
    Wrapper of SAX C++ implementation.
    '''
    cdef SAX * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a, int di = 1):
        self.thisptr = new SAX(N, w, a, di)

    def __dealloc__(self):
        del self.thisptr

    def timeseries2SAX(self, ts):
        return self.thisptr.timeseries2SAX(ts)
        

    def timeseries2SAXseq(self, ts):
        words = self.thisptr.timeseries2SAX(ts)
        seq = b''
        
        for w in words:
            seq = seq + b' ' + w
        if seq:  # remove extra space
            seq = seq[1:]
        return seq

    def map_weighted_patterns(self, ts, sequences, weights):
        return self.thisptr.map_weighted_patterns(ts, sequences, weights)

###########################################################################

cdef extern from "sfa/SFAWrapper.cpp":
    cdef cppclass SFAWrapper:
        SFAWrapper(int, int, int, bool, bool)        
        void fit(vector[vector[double]])
        vector[string] transform(vector[vector[double]])
    # cdef void printHello()

cdef class PySFA:
    '''
    Wrapper of SFA C++ implementation.
    '''
    cdef SFAWrapper * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a, bool norm, bool normTS):
        self.thisptr = new SFAWrapper(N, w, a, norm, normTS)

    def __dealloc__(self):
        del self.thisptr

    def fit(self, X):
        self.thisptr.fit(X)
        return self

    def transform(self, X):
        return self.thisptr.transform(X)



#########################SQM wrapper#########################


cdef extern from "strie.cpp":
    cdef cppclass SeqTrie:
        SeqTrie(vector[string])
        vector[int] search(string)


cdef extern from "sqminer.h":
    cdef cppclass SQMiner:
        SQMiner(double, double)
        vector[string] mine(vector[string] &, vector[int] &)

cdef class PyFeatureTrie:
    cdef SeqTrie *thisptr

    def __cinit__(self, vector[string] sequences):
        self.thisptr = new SeqTrie(sequences)
    def __dealloc__(self):
        del self.thisptr

    def search(self, string sequence):
        return self.thisptr.search(sequence)


cdef class PySQM:
    cdef SQMiner *thisptr

    def __cinit__(self, double selection, double threshold):
        self.thisptr = new SQMiner(selection,threshold)
    def __dealloc__(self):
        del self.thisptr

    def mine(self, vector[string] sequences, vector[int] labels):
        return self.thisptr.mine(sequences, labels)     


######################### MrSQM Classifier #########################

class MrSQMClassifier:    
    '''     
    Overview: MrSQM is an efficient time series classifier utilizing symbolic representations of time series. MrSQM implements four different feature selection strategies (R,S,RS,SR) that can quickly select subsequences from multiple symbolic representations of time series data.
    def __init__(self, strat = 'RS', features_per_rep = 500, selection_per_rep = 2000, nsax = 1, nsfa = 0, custom_config=None, random_state = None, sfa_norm = True):

    Parameters
    ----------
    
    strat               : str, feature selection strategy, either 'R','S','SR', or 'RS'. R and S are single-stage filters while RS and SR are two-stage filters. By default set to 'RS'.
    features_per_rep    : int, (maximum) number of features selected per representation. By deafault set to 500.
    selection_per_rep   : int, (maximum) number of candidate features selected per representation. Only applied in two stages strategies (RS and SR). By deafault set to 2000.
    nsax                : int, control the number of representations produced by sax transformation.
    nsfa                : int, control the number of representations produced by sfa transformation.
    custom_config       : dict, customized parameters for the symbolic transformation.
    random_state        : set random seed for classifier. By default 'none'.
    ts_norm             : time series normalisation (standardisation). By default set to 'True'.

    '''

    def __init__(self, callback, strat = 'RS', features_per_rep = 500, selection_per_rep = 2000, nsax = 1, nsfa = 0, custom_config=None, random_state = None, sfa_norm = True):

        self.callback = callback
        self.nsax = nsax
        self.nsfa = nsfa

        self.sfa_norm = sfa_norm
        if random_state is not None:
            np.random.seed(random_state)
        # self.random_state = (
        #     np.int32(random_state) if isinstance(random_state, int) else None
        # )

        if custom_config is None:
            self.config = [] # http://effbot.org/zone/default-values.htm
        else:
            self.config = custom_config

        self.strat = strat   

        # all the unique labels in the data
        # in case of binary data the first one is always the negative class
        self.classes_ = []
        self.clf = None # scikit-learn model       

        self.fpr = features_per_rep
        self.spr = selection_per_rep
        
        self.filters = [] # feature filters (one filter for a rep) for test data transformation

        debug_logging("Initialize MrSQM Classifier.")
        debug_logging("Feature Selection Strategy: " + strat)
        debug_logging("SAX Reps: " + str(self.nsax) + "x")
        debug_logging("SFA Reps: " + str(self.nsfa) + "x")
        debug_logging("Number of features per rep: " + str(self.fpr))
        debug_logging("Number of candidates per rep (only for SR and RS):" + str(self.spr))
        
     

    def create_pars(self, min_ws, max_ws, xrep, random_sampling=False, is_sfa=False):
        pars = []      
        if xrep > 0:      
            if random_sampling:    
                debug_logging("Sampling window size, word length, and alphabet size.")       
                ws_choices = [int(2**(w/xrep)) for w in range(3*xrep,xrep*int(np.log2(max_ws))+ 1)]            
                wl_choices = [6,8,10,12,14,16]
                if is_sfa:
                    wl_choices = [6,8,10,12,14] # can't handle 16x6 case
                alphabet_choices = [3,4,5,6]

                nrep = xrep*int(np.log2(max_ws))                
                for w in range(nrep):
                    pars.append([np.random.choice(ws_choices) , np.random.choice(wl_choices), np.random.choice(alphabet_choices)])
            else:
                #debug_logging("Doubling the window while fixing word length and alphabet size.")                   
                #pars = [[int(2**(w/xrep)),8,4] for w in range(3*xrep,xrep*int(np.log2(max_ws))+ 1)]     
                #if not is_sfa:
                pars = [[w,8,4] for w in range(8,max_ws,int(np.sqrt(max_ws)))]     

        debug_logging("Symbolic Parameters: " + str(pars))      
            
        
        return pars
  
    def sample_random_sequences(self, seqs, min_length, max_length, max_n_seq):  
                
        output = set()
        splitted_seqs = [s.split(b' ') for s in seqs]
        n_input = len(seqs)       

        # while len(output) < n_seq: #infinity loop if sequences are too alike
        for i in range(0, max_n_seq):
            did = randint(0,n_input)
            wid = randint(0,len(splitted_seqs[did]))
            word = splitted_seqs[did][wid]
            
            s_length = randint(min_length, min(len(word) + 1, max_length + 1))
            start = randint(0,len(word) - s_length + 1)        
            sampled = word[start:(start + s_length)]
            output.add(sampled)
        
        return list(output)



    def feature_selection_on_test(self, mr_seqs):
        debug_logging("Compute test data in subsequence space.")
        full_fm = []


        for rep, seq_features, fs in zip(mr_seqs, self.sequences, self.filters):            
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
            ft = PyFeatureTrie(seq_features)
            for i,s in enumerate(rep):
                fm[i,:] = ft.search(s)
            fm = fm > 0 # binary only

            # if self.strat == 'RS':
            #     fm = fs.transform(fm)        
            full_fm.append(fm)


        full_fm = np.hstack(full_fm)
        #return self.final_vt.transform(full_fm)
        return full_fm
    

    def mine(self,rep, int_y):        
        mined_subs = []
        #print("mine")
        if self.strat == 'S':
            debug_logging("Select " + str(self.fpr) + " discrimative subsequences with SQM.")
            miner = PySQM(self.fpr,0.0)
            mined_subs = miner.mine(rep, int_y)

        elif self.strat == 'SR':
            debug_logging("Select " + str(self.spr) + " discrimative subsequences with SQM.")
            miner = PySQM(self.spr,0.0)
            mined_subs = miner.mine(rep, int_y)
            debug_logging("Randomly pick " + str(self.fpr) + " subsequences from the list.")
            mined_subs = np.random.permutation(mined_subs)[:self.fpr].tolist()

        elif self.strat == 'R':
            debug_logging("Random sampling " + str(self.fpr) + " subsequences from this symbolic representation.")
            mined_subs = self.sample_random_sequences(rep,3,16,self.fpr)
        elif self.strat == 'RS':
            debug_logging("Random sampling " + str(self.spr) + " subsequences from this symbolic representation.")
            mined_subs = self.sample_random_sequences(rep,3,16,self.spr)

        debug_logging("Found " + str(len(mined_subs)) + " unique subsequences.")
        return mined_subs

    def feature_selection_on_train(self, mr_seqs, y):
        debug_logging("Compute train data in subsequence space.")
        full_fm = []
        self.filters = []

        for i in range(0, len(mr_seqs)):
            #for rep, seq_features in zip(mr_seqs, self.sequences):
            rep = mr_seqs[i]
            seq_features = self.sequences[i]
            fm = np.zeros((len(rep), len(seq_features)), dtype=np.int32)
            ft = PyFeatureTrie(seq_features)
            for ii, s in enumerate(rep):
                fm[ii, :] = ft.search(s)
            fm = fm > 0  # binary only

            fs = SelectKBest(chi2, k=min(self.fpr, fm.shape[1]))
            if self.strat == 'RS':
                debug_logging("Filter subsequences of this representation with chi2 (only with RS).")
                fm = fs.fit_transform(fm, y)
                self.sequences[i] = [seq_features[ii] for ii in fs.get_support(indices=True)]

            self.filters.append(fs)
            full_fm.append(fm)

        full_fm = np.hstack(full_fm)

        #self.final_vt = VarianceThreshold()
        #return self.final_vt.fit_transform(full_fm)
        return full_fm

    def fit(self, X, y):
        debug_logging("Fit training data.")
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        debug_logging("Search for subsequences.")        
        old_mr_seqs, mr_seqs = self.callback(self, X)#self.MY_transform_time_series(X)
        

        
        for rep in tqdm(mr_seqs, desc="mining sequences"):
            mined = self.mine(rep,int_y)
            self.sequences.append(mined)

        #if True == True:
        #    raise Exception(X, old_mr_seqs, mr_seqs, self.sequences)


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        debug_logging("Compute feature vectors.")
        train_x = self.feature_selection_on_train(mr_seqs, int_y)

        self.train_x = train_x
        self.int_y = int_y
        
        debug_logging("Fit logistic regression model.")
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)        
        # self.clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True).fit(train_x, y)        
        self.classes_ = self.clf.classes_ # shouldn't matter  

        return self


    def predict_proba(self, X): 
        old_mr_seqs, mr_seqs = self.callback(self, X)#self.MY_transform_time_series(X)
        test_x = self.feature_selection_on_test(mr_seqs)
        self.test_x = test_x
        return self.clf.predict_proba(test_x) 

    def predict(self, X):
        old_mr_seqs, mr_seqs = self.callback(self, X)#self.MY_transform_time_series(X)
        test_x = self.feature_selection_on_test(mr_seqs)
        self.test_x = test_x
        return self.clf.predict(test_x)

    def decision_function(self, X):
        old_mr_seqs, mr_seqs = self.callback(self, X)#self.MY_transform_time_series(X)
        test_x = self.feature_selection_on_test(mr_seqs)
        self.test_x = test_x
        return self.clf.decision_function(test_x)

    def get_saliency_map(self, ts):        

        is_multiclass = len(self.classes_) > 2
        weighted_ts = np.zeros((len(self.classes_), len(ts)))

        fi = 0
        for cfg, features in zip(self.config, self.sequences):
            if cfg['method'] == 'sax':
                ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                if is_multiclass:
                    for ci, cl in enumerate(self.classes_):
                        weighted_ts[ci, :] += ps.map_weighted_patterns(
                            ts, features, self.clf.coef_[ci, fi:(fi+len(features))])
                else:
                    # because classes_[1] is the positive class
                    weighted_ts[1, :] += ps.map_weighted_patterns(
                        ts, features, self.clf.coef_[0, fi:(fi+len(features))])

            fi += len(features)
        if not is_multiclass:
            weighted_ts[0, :] = -weighted_ts[1, :]
        return weighted_ts
        





 






