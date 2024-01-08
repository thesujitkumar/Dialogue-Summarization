#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import pandas as pd
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from datasets import load_dataset


# In[135]:


dataset = load_dataset('samsum')


# In[136]:


dataset


# In[137]:


import re
def striphtml(data):
    p = re.compile(r'<(.*)>.*?|<(.*) />')
    return p.sub('', data)

def preprocess(sentence):
    #print("setence before parsing",sentence)
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence =  striphtml(sentence)
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 1 ]
#     filtered_words = [w for w in filtered_words if w not in stopwords.words('english')]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
#     lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    #print("filtered word"," ".join(lemma_words))
    return " ".join(filtered_words)


# In[ ]:





# In[138]:


dialogues = []
dia_len=[]
summaries =[] 
sum_len=[]
print(len(dataset['train']))
# Extract dialogue and summary information from the dataset
for i in range(len(dataset['train'])):
    a = preprocess(dataset['train'][i]['dialogue'])
    b = preprocess(dataset['train'][i]['summary'])
    if len(a.split()) ==0 or len(b.split()) == 0 :
        continue
    else:
        dialogues.append(a)
        dia_len.append(len(a.split()))
        summaries.append(b)
        sum_len.append(len(b.split()))
    
    
    
print("the length of dia_len list ", len(dia_len))
print("the length of sum_len list ", len(sum_len))
    


# Create a DataFrame
df = pd.DataFrame({'dialogue': dialogues, 'summary': summaries})

# Display the DataFrame
print(df.head())
df = df.reset_index(drop=True)
df.to_csv("clean_train.csv")
df.columns


# In[139]:


df_count = pd.DataFrame({'dial_word_count': dia_len, 'sum_word_count': sum_len})


# In[140]:


x=0
for a in df_count["dial_word_count"]:
    if a== 0:
        x=x+1
print(x)
x=0
for a in df_count["sum_word_count"]:
    if a== 0:
        x=x+1    
print(x)


# In[141]:


x


# In[142]:


df_count.columns


# In[143]:


df_count["sum_word_count"].describe()


# In[144]:


df_count["dial_word_count"].describe()


# In[145]:


dialogues = []
dia_len=[]
summaries =[] 
sum_len=[]
print(len(dataset['test']))
# Extract dialogue and summary information from the dataset
for i in range(len(dataset['test'])):
    a = preprocess(dataset['test'][i]['dialogue'])
    b = preprocess(dataset['test'][i]['summary'])
    if len(a.split()) ==0 or len(b.split()) == 0 :
        continue
    else:
        dialogues.append(a)
        dia_len.append(len(a.split()))
        summaries.append(b)
        sum_len.append(len(b.split()))
    
    
    
print("the length of dia_len list ", len(dia_len))
print("the length of sum_len list ", len(sum_len))
    


# Create a DataFrame
df = pd.DataFrame({'dialogue': dialogues, 'summary': summaries})

# Display the DataFrame
print(df.head())
df = df.reset_index(drop=True)
df.to_csv("clean_test.csv")
df.columns


# In[146]:


df_count = pd.DataFrame({'dial_word_count': dia_len, 'sum_word_count': sum_len})


# In[147]:


x=0
for a in df_count["dial_word_count"]:
    if a== 0:
        x=x+1
print(x)
x=0
for a in df_count["sum_word_count"]:
    if a== 0:
        x=x+1    
print(x)


# In[148]:


df_count["dial_word_count"].describe()


# In[149]:


df_count["sum_word_count"].describe()


# In[150]:


dialogues = []
dia_len=[]
summaries =[] 
sum_len=[]
print(len(dataset['validation']))
# Extract dialogue and summary information from the dataset
for i in range(len(dataset['validation'])):
    a = preprocess(dataset['validation'][i]['dialogue'])
    b = preprocess(dataset['validation'][i]['summary'])
    if len(a.split()) ==0 or len(b.split()) == 0 :
        continue
    else:
        dialogues.append(a)
        dia_len.append(len(a.split()))
        summaries.append(b)
        sum_len.append(len(b.split()))
    
    
    
print("the length of dia_len list ", len(dia_len))
print("the length of sum_len list ", len(sum_len))
    


# Create a DataFrame
df = pd.DataFrame({'dialogue': dialogues, 'summary': summaries})

# Display the DataFrame
print(df.head())
df = df.reset_index(drop=True)
df.to_csv("clean_valid.csv")
df.columns


# In[151]:


df_count = pd.DataFrame({'dial_word_count': dia_len, 'sum_word_count': sum_len})


# In[152]:


x=0
for a in df_count["dial_word_count"]:
    if a== 0:
        x=x+1
print(x)
x=0
for a in df_count["sum_word_count"]:
    if a== 0:
        x=x+1    
print(x)


# In[153]:


df_count["dial_word_count"].describe()


# In[154]:


df_count["sum_word_count"].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




