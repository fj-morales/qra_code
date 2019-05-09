
# coding: utf-8

# ## ADA: BM25+RM3 baseline

# In[23]:


import pickle
import json
import gzip
import os
import subprocess
import numpy as np
import multiprocessing
import re 
import csv
import torch
import sys
# sys.path.append('qra_cod')
from utils.meter import AUCMeter
import shutil
import random

import uuid
import datetime
import time

# In[24]:


def remove_sc(text):
###    text = re.sub('[.,?;*!%^&_+():-\[\]{}]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip())
##    text = re.sub('[\[\]{}.,?;*!%^&_+():-]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip()) # DeepPaper method
    text = re.sub(r'[^\w\s]',' ',text) # My method
###     text = text.rstrip('.?')
    return text


# In[25]:


def read_questions(filename):
    with gzip.open(filename, 'rt') as tsv_in:
        qreader = csv.reader(tsv_in, delimiter = '\t')
        questions = {}
#         q_dict = {}
        for q in qreader:
            question = {}
            if 'quora' in filename:
                print('quora')
#             elif 'sprint' in filename:
#                 print('print')
            else:
#                 question['id'] = q[0]
#                 q_dict[q[0]] = q[1] + ' ' + q[2]
                question['title'] = q[1]
                question['text'] = q[2]
                questions[q[0]]=(dict(question))
#         return [questions, q_dict]
        return questions


# In[26]:


def trectext_format(questions):
    trec_questions = {}
    for key, q in questions.items():
        doc = '<DOC>\n' +               '<DOCNO>' + key + '</DOCNO>\n' +               '<TITLE>' + q['title'] + '</TITLE>\n' +               '<TEXT>' + q['text'] + '</TEXT>\n' +               '</DOC>\n'
        trec_questions[key] = doc
    return trec_questions


# In[27]:


def to_trecfile(docs, filename, compression = 'yes'):
    # Pickle to Trectext converter
    doc_list = []
    if compression == 'yes':
        with gzip.open(filename,'wt') as f_out:
            docus = {}
            for key, value in docs.items():
                f_out.write(value)
    else:
        with open(filename,'wt') as f_out:
            docus = {}
            for key, value in docs.items():
                f_out.write(value)


# In[28]:


def save_trectext(trec_questions, filename, compression = True):
# Generate file to index
#     with gzip.open(filename,'wt', encoding='utf-8') as f_out:
    if compression == True:
        with gzip.open(filename,'wt') as f_out:
            for key, value in trec_questions.items():
                f_out.write(value)
    else:
        with open(filename,'wt') as f_out:
            for key, value in trec_questions.items():
                f_out.write(value)


# In[29]:


def build_index(index_input, index_loc):
    if build_index_flag == 'no':
        return
# Build corpus index 
    if os.path.exists(index_loc):
        shutil.rmtree(index_loc)
        os.makedirs(index_loc)
    else:
        os.makedirs(index_loc) 
#     index_loc_param = '--indexPath=' + index_loc

    anserini_index = anserini_loc + 'target/appassembler/bin/IndexCollection'
    anserini_parameters = [
#                            'nohup', 
                           'sh',
                           anserini_index,
                           '-collection',
                           'TrecCollection',
                           '-generator',
                           'JsoupGenerator',
                           '-threads',
                            '16',
                            '-input',
                           index_input,
                           '-index',
                           index_loc,
                           '-storePositions',
                            '-keepStopwords',
                            '-storeDocvectors',
                            '-storeRawDocs']
#                           ' >& ',
#                           log_file,
#                            '&']



#     anserini_parameters = ['ls',
#                           index_loc]


    print(anserini_parameters)

    index_proc = subprocess.Popen(anserini_parameters,
            stdout=subprocess.PIPE, shell=False)
    (out, err) = index_proc.communicate()
    print(out.decode("utf-8"))
    print('Index error: ', err)


# In[30]:


def read_dups(dups_file):
    with open(dups_file, 'rt') as dups_in:
        dup_reader = csv.reader(dups_in, delimiter = ' ')
        dup_list = []
        dup_dict = {}
        for dup in dup_reader:
            dup_dict['doc_id'] = dup[0]
            dup_dict['dup_id'] = dup[1]
            if 'pos' in dups_file:
                dup_dict['label'] = 1
            elif 'neg' in dups_file:
                dup_dict['label'] = 0
            dup_list.append(dict(dup_dict))
    return dup_list


# In[31]:


def read_dup_files(dups_file):
    with open(dups_file, 'rt') as dups_in:
        dup_reader = csv.reader(dups_in, delimiter = ' ')
        dup_list = []
        for dup in dup_reader:
#             print(dup)
            if dup[0] in dup_dict.keys():
                dup_dict[dup[0]].append(dup[1])
            else:
                dup_dict[dup[0]] = [dup[1]]
    return dup_list


# In[32]:


def generate_queries_file(questions, q_dup_pos, filename):
    queries_list = []
    queries_dict = {}
    query = {}
    id_num = 0
    ids_dict = {}
    q_trec = {}
    for query in q_dup_pos:
        str_id = str(id_num)
        id_new = str_id.rjust(15, '0')
        
        key = query['doc_id']
        q = questions[key]
#         print(key)
        text = remove_sc(q['title'] + ' ' + q['text']) #Join title and text 
        query['number'] = key
#         query['text'] = '#stopword(' + text + ')'
        query['text'] = '(' + text + ')'
        queries_list.append(dict(query))
        
        q_t = '<top>\n\n' +           '<num> Number: ' + id_new + '\n' +           '<title> ' + text + '\n\n' +           '<desc> Description:' + '\n\n' +           '<narr> Narrative:' + '\n\n' +           '</top>\n\n'
        q_trec[key] = q_t
#         print(q)
        ids_dict[str(id_num)] = key
        id_num += 1
        
    queries_dict['queries'] = queries_list
    # with open(filename, 'wt', encoding='utf-8') as q_file:
    with open(filename, 'wt') as q_file: #encoding option not working on python 2.7
        json.dump(queries_dict, q_file, indent = 4)
        
    return [q_trec, ids_dict]
        
        ########################
        ########################


# In[33]:


def retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, hits, b=0.2, k=0.8, N=10, M=10, Lambda=0.5):
    print(q_topics_file)
    #print(hits)
    anserini_search = anserini_loc + 'target/appassembler/bin/SearchCollection'
    command = [ 
               'sh',
               anserini_search,
               '-topicreader',
                'Trec',
                '-index',
                index_loc,
                '-topics',
                q_topics_file,
                '-output',
                retrieved_docs_file,
                '-bm25',
                '-b',
                str(b),
                '-k1',
                str(k),
                '-rm3',
                '-rm3.fbDocs',
                str(N),
                '-rm3.fbTerms',
                str(M),
                '-rm3.originalQueryWeight',
                str(Lambda),
                '-hits',
                str(hits), 
                '-threads',
                '10'
               ]
    print(command)
#     command = command.encode('utf-8')
    anserini_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    (out, err) = anserini_exec.communicate()
#     print(out)
    print('Searching error: ', err)


# In[34]:


# Return top 1 bm25 scored question = 'duplicated' question
# Return top 100 bm25 scored docs, given query and corpus indexed by anserini

def generate_preds_file(retrieved_docs_file, q_all, ids_dict, hits):
    
    with open(retrieved_docs_file, 'rt') as f_in:
        all_dict = {}
        for doc in f_in:
#             print(doc)
            id_aux = doc.split(' ')[0]
            current_key = ids_dict[id_aux]
            key_pair = current_key + '_' + doc.split(' ')[2]
            all_dict[key_pair] = doc.split(' ')[4]
        bm25_scores = [] 
        i = 0
        for query_dict in q_all:
            i += 1
            key_pair = query_dict['doc_id'] + '_' + query_dict['dup_id']
            try: 
                query_dict['score'] = all_dict[key_pair]
            except:
                query_dict['score'] = 0
#            if i % 10000 == 0:
#                 print('processed: ', i)
            bm25_scores.append(dict(query_dict))
        return bm25_scores
          


# In[35]:


def remove_work_dirs():
    if debug == 'yes':
        print('yes')
        # Execute remove sequence


# In[ ]:


def save_preds(file, preds):
    with open(file, 'wt') as f_out:
        json.dump(preds, f_out, indent=4)


# In[ ]:


def start_process():
    print( 'Starting', multiprocessing.current_process().name)


# In[43]:

def get_random_params(hyper_params, num_iter):
    random_h_params_list = []
    while len(random_h_params_list) < num_iter:
        random_h_params_set = []
        for h_param_list in hyper_params:
            sampled_h_param = random.sample(list(h_param_list), k=1)
#             print(type(sampled_h_param[0]))
#             print(sampled_h_param[0])
            random_h_params_set.append(round(sampled_h_param[0], 3))
        if not random_h_params_set in random_h_params_list:
            random_h_params_list.append(random_h_params_set)
#             print('Non repeated')
        else:
            print('repeated')
    return random_h_params_list


def evaluate(baseline_preds):

#     print(baseline_preds[0:1])
    
    scores = [doc['score'] for doc in baseline_preds]
    scores = np.asarray(scores)
    scores = scores.astype(np.float)
    labels = [doc['label'] for doc in baseline_preds]
    labels = np.asarray(labels)
    labels = labels.astype(np.int)
    
    auc_meter = AUCMeter()
    auc_meter.add(scores, labels)
    auc05_score = auc_meter.value(0.05)
    #print('AUC(0.05) = ', auc05_score)
    return auc05_score


# In[36]:


def baseline_computing(params):
    b = params[0]
    k = params[1]
    N = int(params[2])
    M = int(params[3])
    Lambda = params[4]

    params_suffix = 'b' + str(b) + 'k' + str(k) + 'N' + str(N) + 'M' + str(M) + 'Lambda' + str(Lambda) + 'n_rand_iter' + str(num_random_iter) + 'hits' + str(hits)
    retrieved_docs_file = workdir + 'run_bm25_rm3_preds_' + dataset_name[0] + '_' + data_split + '_' + params_suffix + '.txt'

    retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, hits, b, k, N, M, Lambda)
    baseline_preds = generate_preds_file(retrieved_docs_file, q_all, ids_dict, hits)
#     save_preds(baseline_preds_file, baseline_preds)  
    auc05_score = evaluate(baseline_preds)
    
    results = [
        b,
        k,
        N,
        M,
        Lambda,
        float(auc05_score)
    ]
    os.remove(retrieved_docs_file)
#     os.remove(baseline_preds_file)

    return results


# In[ ]:


def find_best_dev_model(best_model_params_file, random_iterations, pool_size):
#     random_search = 'yes'
    
    if random_search == 'yes':
        ## Heavy random search
        brange = np.arange(0.1,1,0.05)
        krange = np.arange(0.1,4,0.1)
        N_range = np.arange(5,500,1) # num of docs
        M_range = np.arange(5,500,1) # num of terms
        lamb_range = np.arange(0,1,0.1) # weights of original query

        ## Light random search
#         brange = [0.2]
#         krange = [0.8]
#         N_range = np.arange(1,50,2)
#         M_range = np.arange(1,50,2)
#         lamb_range = np.arange(0,1,0.2)
        
        h_param_ranges = [brange, krange, N_range, M_range, lamb_range]
        params = get_random_params(h_param_ranges, random_iterations)

    else:
        brange = [0.2]
        krange = [0.8]
        N_range = [11]
        M_range = [10]
        lamb_range = [0.5]
       
        params = [[round(b,3), round(k,3), round(N,3), round(M,3), round(Lambda,3)] 
                  for b in brange for k in krange for N in N_range for M in M_range for Lambda in lamb_range]
    
#     print(len(params))
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

#     pool_outputs = pool.map(baseline_computing, params)
    

    pool_outputs = pool.map_async(baseline_computing, params)
    print(pool_outputs.get())
    ###

    
    ##
    
    
    pool.close() # no more tasks
    while (True):
        if (pool_outputs.ready()): break
        remaining = pool_outputs._number_left
#         remaining2 = remaining1
#         remaining1 = pool_outputs._number_left
        if remaining%10 == 0:
            print("Waiting for", remaining, "tasks to complete...")
            time.sleep(2)
        
      
    pool.join()  # wrap up current tasks
    pool_outputs.get()
    params_file = './baselines/best_ir_model/' + dataset_name[0] + '_' + 'bm25_rm3_' + data_split + '_hparams.pickle'
    pickle.dump(pool_outputs.get(), open(params_file, "wb" ) )
    print('Total parameters tested: ' + str(len(pool_outputs.get())))
    best_model_params = max(pool_outputs.get(), key=lambda x: x[5])
    
    best_model_dict = {
        'b': best_model_params[0],
        'k': best_model_params[1],
        'N': best_model_params[2],
        'M': best_model_params[3],
        'Lambda': best_model_params[4],
        'random_iterations': random_iterations,
        'hits': hits,
        'auc05_score': best_model_params[5],
    }
    best_model_dict = {k:str(v) for k, v in best_model_dict.items()} # everything to string
    
    print(best_model_dict)
    with open(best_model_params_file, 'wt') as best_model_f:
        json.dump(best_model_dict, best_model_f)


# def get_test_metrics(best_params):
    
    
#     try:
#         with open(best_model_params_file, 'rt') as best_model_in:
#             best_dev_params = json.load(best_model_in)
#     #         print(best_dev_model_params)
#     #         best_dev_params = {k:float(v) for k, v in best_dev_params.items()}
#         params = [best_dev_params['b'],
#                   best_dev_params['k'],
#                   best_dev_params['N'],
#                   best_dev_params['M'],
#                   best_dev_params['Lambda']
#                   best_dev_params['hits']
#                  ]
        
#     except:
#         print('No dev model file. Run Dev model first!')
        
        
# In[37]:


if __name__ == '__main__':
    
    print('Starting: ', datetime.datetime.now())
    
    try:
        qloc = sys.argv[1] 
        if not qloc.endswith('/'):
            qloc = qloc + '/'
        print(qloc)
        data_split = sys.argv[2]
        
        
    except:
        sys.exit("Provide data location, split, number of retrieved documents (hits), and number of random iterations")
    
    try:
        num_random_iter = sys.argv[4]
        pool_size = int(sys.argv[5])
    except: 
        pool_size = 10
        if 'dev' in data_split:
            num_random_iter = 5000
        elif 'test' in data_split:
            print('No need for random iterations in test mode.')
            num_random_iter = 1
    
    ## Options

    build_index_flag = 'yes'
    
    random_search = 'yes'
#     data_split = 'test'
    
    
    workdir = './baselines/workdir/'
   # hits = 500
    
    
    
#     qloc = './qra_data/apple/' # Must be a commandline parameter
    
    
    
    galago_loc='./baselines/galago-3.10-bin/bin/'
    trec_storage = '/ssd/francisco/trec_datasets/qra/'
    
    to_index_files ='./baselines/to_index_files/'
    anserini_loc = '../anserini/'

    dataset_name = qloc.split('/')[-2:]
    
    if not os.path.exists(workdir): 
        os.makedirs(workdir)
    
    if os.path.exists(to_index_files): 
        shutil.rmtree(to_index_files)
        os.makedirs(to_index_files)
    else:
        os.makedirs(to_index_files)
    
    loc_prefix = workdir + dataset_name[0]
    index_loc = loc_prefix + '_anserini_index'
    questions_file = loc_prefix + '_questions' + '.gz'
    queries_file = loc_prefix + '_queries'
    trectext_file = to_index_files + dataset_name[0] + '_trectext.gz' # needs to be alone, no other files in the same directory when indexing
    trectext_doc_file = trec_storage + dataset_name[0] + '_trectext'
    
    index_input = to_index_files
    dups_file_pos = qloc + data_split + '.pos.txt'
    dups_file_neg = qloc + data_split + '.neg.txt'
    corpus_file = qloc + 'corpus.tsv.gz' 
    
    questions = read_questions(corpus_file)
    
    q_dup_pos = read_dups(dups_file_pos)
    #print(len(q_dup_pos))
    q_dup_neg = read_dups(dups_file_neg)
    #print(len(q_dup_neg))
    q_all = q_dup_pos + q_dup_neg 
    trec_questions = trectext_format(questions)
    save_trectext(trec_questions, trectext_file)
    save_trectext(trec_questions, trectext_doc_file, compression = False)
    
    build_index(index_input, index_loc)
    
    q_dup_pos[0:2]
    
    
    #print(queries_file)
    q_topics_file = loc_prefix + '_query'
    [q_trec, ids_dict] = generate_queries_file(questions, q_dup_pos, queries_file)
    to_trecfile(q_trec, q_topics_file, compression = 'no')
    ids_equivalence_filename = dataset_name[0] + '_' + 'ids_equivalence'  + '_' + data_split + '.txt'
    ids_equivalence_file = trec_storage + ids_equivalence_filename
    with open(ids_equivalence_file, 'wt') as outfile:
        json.dump(ids_dict, outfile)
    
    best_model_params_file = './baselines/best_ir_model/' + dataset_name[0] + '_bm25_rm3_best_model_dev.json'
    
    if 'dev' in data_split:
        print('Dev Mode')
        
        try:
            hits = sys.argv[3]
        except: 
            print('No number of documents (hits) provided. Using default = 5000.')
            hits = 5000
        find_best_dev_model(best_model_params_file, int(num_random_iter), pool_size)
    if 'test' in data_split:
        print('Test Mode')
        if os.path.exists(best_model_params_file):
            print("Best model exits. Loading...")
#         try:
            
        with open(best_model_params_file, 'rt') as best_model_in:
            best_dev_params = json.load(best_model_in)
    #         print(best_dev_model_params)
    #         best_dev_params = {k:float(v) for k, v in best_dev_params.items()}
            best_dev_params_loaded = [
                        best_dev_params['b'],
                        best_dev_params['k'],
                        best_dev_params['N'],
                        best_dev_params['M'],
                        best_dev_params['Lambda'],
                        best_dev_params['hits']
                     ]
            params = best_dev_params_loaded[0:5]
            hits = best_dev_params_loaded[5]
            print(params)
            test_results = baseline_computing(params)
            print(test_results)

#         except:
#             print('No dev model file. Run Dev model first!')
        
    print('Ending: ', datetime.datetime.now())

# In[38]:





# In[39]:





# In[40]:


# b=0.75
# k=1.2
# N=10
# M=10
# Lambda=0.5

# params_suffix = 'b' + str(b) + 'k' + str(k) + 'N' + str(N) + 'M' + str(M) + 'Lambda' + str(Lambda)
# retrieved_docs_file = workdir + 'run_bm25_rm3_preds_' + dataset_name[0] + '_' + data_split + '_' + params_suffix + '.txt'
# print(retrieved_docs_file)
# retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, hits, b, k, N, M, Lambda)


# In[42]:


# baseline_docs = generate_preds_file(retrieved_docs_file, q_all, ids_dict, hits)
# save_preds(baseline_preds_file, baseline_preds) 

