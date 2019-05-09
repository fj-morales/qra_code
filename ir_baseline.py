
# coding: utf-8

# In[25]:


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


# In[26]:


## Options

build_index_flag = 'yes'
data_split = 'test'
workdir = './baselines/workdir/'
qloc = './qra_data/superuser/'
galago_loc='./baselines/galago-3.10-bin/bin/'


# In[27]:


def remove_sc(text):
###    text = re.sub('[.,?;*!%^&_+():-\[\]{}]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip())
##    text = re.sub('[\[\]{}.,?;*!%^&_+():-]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip()) # DeepPaper method
    text = re.sub(r'[^\w\s]',' ',text) # My method
###     text = text.rstrip('.?')
    return text


# In[28]:


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


# In[29]:


def trectext_format(questions):
    trec_questions = {}
    for key, q in questions.items():
        doc = '<DOC>\n' +               '<DOCNO>' + key + '</DOCNO>\n' +               '<TITLE>' + q['title'] + '</TITLE>\n' +               '<TEXT>' + q['text'] + '</TEXT>\n' +               '</DOC>\n'
        trec_questions[key] = doc
    return trec_questions


# In[30]:


def save_trectext(trec_questions, filename):
# Generate file to index
#     with gzip.open(filename,'wt', encoding='utf-8') as f_out:
    with gzip.open(filename,'wt') as f_out:
        for key, value in trec_questions.items():
            f_out.write(value)


# In[31]:


def build_index(index_input, index_loc):
    if build_index_flag == 'no':
        return
# Build corpus index 
    if not os.path.exists(index_loc):
            os.makedirs(index_loc) 
    index_loc_param = '--indexPath=' + index_loc
    galago_parameters = [galago_loc + 'galago', 'build', '--stemmer+krovetz']
    galago_parameters.append('--inputPath+' + index_input)
    galago_parameters.append(index_loc_param)
    print(galago_parameters)

    index_proc = subprocess.Popen(galago_parameters,
            stdout=subprocess.PIPE, shell=False)
    (out, err) = index_proc.communicate()
    print(out.decode("utf-8"))
    print(err)


# In[32]:


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


# In[33]:


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


# In[34]:


def generate_queries_file(questions, q_dup_pos, filename):
    queries_list = []
    queries_dict = {}
    query = {}
    for query in q_dup_pos:
        key = query['doc_id']
        q = questions[key]
        text = remove_sc(q['title'] + ' ' + q['text']) #Join title and text 
        query['number'] = key
#         query['text'] = '#stopword(' + text + ')'
        query['text'] = '(' + text + ')'
        queries_list.append(dict(query))
    queries_dict['queries'] = queries_list
    # with open(filename, 'wt', encoding='utf-8') as q_file:
    with open(filename, 'wt') as q_file: #encoding option not working on python 2.7
        json.dump(queries_dict, q_file, indent = 4)


# In[35]:


# Return top 1 bm25 scored question = 'duplicated' question
def get_bm25_docs(queries_file, q_all, index_loc, b_val=0.75, k_val=1.2):
    index_loc_param = '--index=' + index_loc  
    b=' --b=' + str(b_val)
    k=' --k=' + str(k_val)
    
    command = galago_loc + 'galago threaded-batch-search --threadCount=50 --verbose=false          --casefold=true --requested=50000 ' +          index_loc_param + ' --scorer=bm25' +          b +          k +          '   ' +          queries_file + ' | cut -d" " -f1,3,5 '
#          queries_file + ' | cut -d" " -f1,3,5 > all_results.txt'
        # cut -d" " -f1,3' # for the document 
        
    print(command)
#     command = command.encode('utf-8')
#     galago_bm25_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
    galago_bm25_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = galago_bm25_exec.communicate()
    all_scores = out.splitlines()
#     print(ids_docs)
    
#     return ids_docs
    all_dict = {}
    print(len(all_scores))
    for doc in all_scores:
        key_pair = doc.split(' ')[0] + '_' + doc.split(' ')[1]
        all_dict[key_pair] = doc.split(' ')[2]
    bm25_scores = [] 
    i = 0
    for query_dict in q_all:
        i += 1
        key_pair = query_dict['doc_id'] + '_' + query_dict['dup_id']
        try: 
            query_dict['score'] = all_dict[key_pair]
        except:
            query_dict['score'] = 0
        if i % 10000 == 0:
            print('processed: ', i)
        bm25_scores.append(dict(query_dict))
    return bm25_scores   


# In[36]:


def remove_work_dirs():
    if debug == 'yes':
        print('yes')
        # Execute remove sequence


# In[49]:


#len(q_all)


# In[37]:


dataset_name = qloc.split('/')[-2:]


# In[38]:


if not os.path.exists(workdir):
    os.makedirs(workdir)


# In[39]:


loc_prefix = workdir + dataset_name[0]
index_loc = loc_prefix + '_index'
questions_file = loc_prefix + '_questions' + '.gz'
queries_file = loc_prefix + '_queries'
trectext_file = loc_prefix + '_trectext.gz'
index_input = trectext_file
dups_file_pos = qloc + data_split + '.pos.txt'
dups_file_neg = qloc + data_split + '.neg.txt'
corpus_file = qloc + 'corpus.tsv.gz'


# In[40]:


questions = read_questions(corpus_file)


# In[41]:


q_dup_pos = read_dups(dups_file_pos)
print(len(q_dup_pos))
q_dup_neg = read_dups(dups_file_neg)
print(len(q_dup_neg))
q_all = q_dup_pos + q_dup_neg 
trec_questions = trectext_format(questions)
save_trectext(trec_questions, trectext_file)


# In[42]:


q_dup_pos[0:2]


# In[43]:


generate_queries_file(questions, q_dup_pos, queries_file)


# In[44]:


build_index(index_input, index_loc)


# In[45]:


bm25_docs = get_bm25_docs(queries_file, q_all, index_loc, b_val=0.75, k_val=1.2)


# In[46]:


bm25_docs[0:1]


# In[47]:


scores = [doc['score'] for doc in bm25_docs]
scores = np.asarray(scores)
scores = scores.astype(np.float)
labels = [doc['label'] for doc in bm25_docs]
labels = np.asarray(labels)
labels = labels.astype(np.int)


# In[48]:


auc_meter = AUCMeter()
auc_meter.add(scores, labels)
auc05_score = auc_meter.value(0.05)
print('AUC(0.05) = ', auc05_score)

