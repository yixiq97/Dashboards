#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px 
import matplotlib.pyplot as plt


# In[2]:


import re
import spacy
nlp = spacy.load('en_core_web_sm')
STOPWORDS = nlp.Defaults.stop_words
stopwords = STOPWORDS # customize (TBD)


# In[3]:


from dash_holoniq_wordcloud import DashWordcloud


# In[4]:


import dash
#import dash_core_components as dcc
#import dash_html_components as html
import dash.dependencies as dd
from dash import dcc
from dash import html


# In[521]:


from wordcloud import WordCloud
from io import BytesIO
import base64


# In[5]:


import requests
from bs4 import BeautifulSoup


# # GET DATA

# In[6]:


def get_CREC_packages(pageSize, startDate= '2018-07-01T00:00:00'): #from collections
    apiKey = 'RjhPNKjGtGp1AbN4Ilb1jbrhCO9IBvqqLvRKsA1e'
    api_url = f'https://api.govinfo.gov/collections/CREC/{startDate}Z?offset=0&pageSize={pageSize}&api_key={apiKey}'
    data = requests.get(api_url).json()
    data = data['packages']
    return data


# In[7]:


def get_CREC_summary(packageLink):   #packageLink retrieved from package table
    apiKey = 'RjhPNKjGtGp1AbN4Ilb1jbrhCO9IBvqqLvRKsA1e'
    url = packageLink + f'?api_key={apiKey}'
    record = requests.get(url).json()
    return record #one record each time
    


# In[8]:


def get_CREC_granules(granulesLink):  # granulesLink retrieved from the summary table, get_CREC_summary(packageLink)['granulesLink']
    apiKey = 'RjhPNKjGtGp1AbN4Ilb1jbrhCO9IBvqqLvRKsA1e'
    url = granulesLink + f'&api_key={apiKey}'
    data = requests.get(url).json()
    count = data['count']
    granules = data['granules']
    return count,granules


# In[9]:


# General Version: 
def get_CREC_granuleInfo(granuleLink): # granuleLink retrieved from the granule table
    apiKey = 'RjhPNKjGtGp1AbN4Ilb1jbrhCO9IBvqqLvRKsA1e'
    url = granuleLink + f'?api_key={apiKey}'
    data = requests.get(url).json()
    return data

def txt_download(granule_json):
    apiKey = 'RjhPNKjGtGp1AbN4Ilb1jbrhCO9IBvqqLvRKsA1e'
    url = granule_json['download']['txtLink'] + f'?api_key={apiKey}'
    data = requests.get(url)
    raw_text = data.text
    
    soup = BeautifulSoup(raw_text)
    text_body = soup.find('body').get_text()
    return text_body


# ### Endpoints (TBD)

# In[10]:


class DailyDigest():
    def __init__(self,granules_data):
        self.all_granules = granules_data
        self.all_digests = [i for i in granules_data if i['granuleClass'] == 'DAILYDIGEST']
        self.today_digests = self.all_digests[2:]
        
    #def get_titles():
    #def get_class():

    def today_digest_text(self):
        today_text = ''
        for record in self.today_digests:
            link = record['granuleLink']
            dailyDigest_json = get_CREC_granuleInfo(link)
            text_record = txt_download(dailyDigest_json)
            today_text += text_record
        return today_text
    
    def today_digest_df(self):
        today_df = pd.DataFrame()
        today_df['title'] = [i['title'] for i in self.today_digests]
        content = []
        for record in self.today_digests:
            link = record['granuleLink']
            dailyDigest_json = get_CREC_granuleInfo(link)
            text_record = txt_download(dailyDigest_json)
            content.append(text_record)
        today_df['content'] = content
        return today_df


# # PROCESS DATATABLE

# In[12]:


# get packagelink list from CREC collection data
collection_json = get_CREC_packages(10)
collection_attr = list(collection_json[0].keys())
packageLinks = [r['packageLink'] for r in collection_json]
collection_df = pd.DataFrame(collection_json)[['packageId','dateIssued','title','lastModified']]
    
    
# get summary of each CERC record, in 1:1 relationship
summary_data = [get_CREC_summary(packageLink) for packageLink in packageLinks]
summary_attr = list(summary_data[0].keys())
granulesLinks = [r['granulesLink'] for r in summary_data]
record_info_df = pd.DataFrame(summary_data)[['packageId','detailsLink','pages','governmentAuthor1','category']]
    
    
# get granule contents of each CERC record, in 1:M relationship
granules_relation = [(granulesLink,get_CREC_granules(granulesLink)[0],get_CREC_granules(granulesLink)[1]) 
                    for granulesLink in granulesLinks]    
granules_backward_mapping = dict([(r['granulesLink'],r['packageId']) for r in summary_data])
names = globals()
num_of_granules_table = 0
names_of_granules_table = []
for granules in granules_relation:
    gran_link = granules[0]
    pack_id = granules_backward_mapping[gran_link]
    pack_id_formatted = pack_id.replace('-','')
    names['granules_%s_df' % pack_id_formatted] = pd.DataFrame(granules[2])[['granuleId','title','granuleClass']]
    eval('granules_%s_df' % pack_id_formatted)['packageId'] = pack_id
    names_of_granules_table.append('granules_%s_df' % pack_id)
    # return to table i: eval(names_of_granules_table[i])
    num_of_granules_table += 1
granules_count_df=pd.DataFrame(granules_relation,columns = ['granulesLink','granulesCount','json_file'])[['granulesLink','granulesCount']]
package_granule_df = pd.DataFrame.from_dict(granules_backward_mapping,orient = 'index',columns = ['packageId'])
package_granule_df = package_granule_df.reset_index().rename(columns = {'index':'granulesLink'})
    
    
# get daily digest today；all dayas + the latest day
dailydigest_data = []
for granules in granules_relation:
    gran_link = granules[0]
    pack_id = granules_backward_mapping[gran_link]
    dailydigest_text = DailyDigest(granules[2]).today_digest_text()
    dailydigest_data.append((pack_id,dailydigest_text))
dailydigest_df = pd.DataFrame(dailydigest_data,columns = ['packageId','dailyDigest'])
    
gran_link_today = granules_relation[0][0]
pack_id_today = granules_backward_mapping[gran_link_today]
dailydigest_today_df = DailyDigest(granules_relation[0][2]).today_digest_df()


# In[13]:


# Join(merge) and get original datatables
CREC_summary = pd.merge(collection_df,record_info_df,on = 'packageId')
CREC_summary['pages'] = CREC_summary.pages.astype('int64')
CREC_granules =  pd.merge(pd.merge(collection_df,package_granule_df,on = 'packageId'),
                          granules_count_df,on = 'granulesLink')
CREC_dailydigest = pd.merge(collection_df,dailydigest_df,on = 'packageId')


# # DATA ANALYTICS

# ## Analyze Daily Active Levels

# In[14]:


num_pages_df = CREC_summary.groupby('dateIssued').agg({'pages': np.sum})
num_granules_df = CREC_granules.groupby('dateIssued').agg({'granulesCount': np.sum})


# ## Analyze Titles of Daily Granules

# #### Text Processing Functions

# In[15]:


# Basic text processing method using built-in functions
def docParse(doclist):
    doclist_parsed = []
    for doc in doclist:
        doclist_parsed.append(doc.split())
    return doclist_parsed

def wordCount(doclist_parsed):  # assume inputs are in a form of documentList(wordList())
    word_ct = dict()
    for wordlist in doclist_parsed:
        for word in wordlist:
            word = word.lower()
            if word not in stopwords:    
                if word not in list(word_ct.keys()):
                    word_ct[word] = 1
                else:
                    word_ct[word] += 1
    word_ct_sorted = sorted(word_ct.items(), key=lambda x:x[1],reverse=True)
    return word_ct_sorted


# In[16]:


# Fancier text processing method using SpaCy
def get_one_package(package_id):
    #package_id = CREC_summary.packageId[i]
    granules_df = eval('granules_%s_df' % package_id.replace('-',''))
    S_H_df = granules_df[(granules_df['granuleClass'] == 'SENATE')|(granules_df['granuleClass'] == 'HOUSE')]
    S_H_titles = list(S_H_df.title)
    titles_parsed = []
    for title in S_H_titles:
        titles_parsed.append(nlp(title))
    return titles_parsed


def get_parsed_titles(titles_parsed):
    daily_tokens = []
    daily_tokens_lemma = []
    daily_tokens_pos_ent = []
    for title_parsed in titles_parsed:
        title_tokens = []
        title_tokens_lemma = []
        title_tokens_pos_ent = []
        for token in title_parsed:
            title_tokens.append(token)
            title_tokens_lemma.append(token.lemma_.lower())
            title_tokens_pos_ent.append((token.lemma_.lower(),token.pos_,token.ent_type_))

        daily_tokens.append(title_tokens)
        daily_tokens_lemma.append(title_tokens_lemma)
        daily_tokens_pos_ent.append(title_tokens_pos_ent)
    return daily_tokens_pos_ent


# In[17]:


# Fancier text processing method using SpaCy (especially for analyzing titles as they're better formatted)
important_nouns = ['petition','resolution','bill']  #https://www.congress.gov/help/congressional-record
important_pos = ['ADJ','PROPN','VERB'] #TBD
important_ents = ['PERSON','NORP','FAC','GPE','LOC'] #TBD

def wordCount_pro(doclist_parsed_spacy,pos_list,ent_list):  # assume inputs are in a form of documentList(wordList())
    word_ct = dict()
    for wordlist in doclist_parsed_spacy:
        for wordzip in wordlist:
            word = wordzip[0]
            word_pos = wordzip[1]
            word_ent = wordzip[2]
            if word not in stopwords and (word_pos in pos_list or word_ent in ent_list):    
                if word not in list(word_ct.keys()):
                    word_ct[word] = 1
                else:
                    word_ct[word] += 1
    word_ct_sorted = sorted(word_ct.items(), key=lambda x:x[1],reverse=True)
    return word_ct_sorted


def wordFilter(doclist_parsed_spacy,pos_list,ent_list): 
    words_filtered = ''
    for wordlist in doclist_parsed_spacy:
        for wordzip in wordlist:
            word = wordzip[0]
            word_pos = wordzip[1]
            word_ent = wordzip[2]
            if word not in stopwords and (word_pos in pos_list or word_ent in ent_list):
                words_filtered += (word+' ')
    return words_filtered


# #### Text Processing & Formatting

# In[18]:


# Get concatenated text content of titles, filtered based on certain nouns, pos, and entity type 
titles_filtered_agg = []
for i in range(len(CREC_summary)):
    parsed_titles = get_parsed_titles(
        get_one_package(CREC_summary.packageId[i])
    )
    title_filtered_i = wordFilter(parsed_titles,important_pos,important_ents)
    titles_filtered_agg.append(title_filtered_i)
titles_filtered_agg  #len(titles_filtered_agg) = len(CREC_summary)


# In[19]:


# Get word count lists of titles, filtered based on certain nouns, pos, and entity type 
count_filtered_agg = []
for i in range(len(CREC_summary)):
    parsed_titles = get_parsed_titles(
        get_one_package(CREC_summary.packageId[i])
    )
    count_filtered_i = wordCount_pro(parsed_titles,important_pos,important_ents)
    count_filtered_i = [list(i) for i in count_filtered_i]
    count_filtered_agg.append(count_filtered_i)
count_filtered_agg   #len(count_filtered_agg) = len(CREC_summary), but len(count_filtered_agg[i]) differs across different i


# In[20]:


# Build new datatable joining concatenated titles and wordCount list, with packageId as PK
titles_filtered_df = pd.DataFrame(zip(CREC_summary.packageId,CREC_summary.dateIssued,titles_filtered_agg,count_filtered_agg),columns = ['packageId','dateIssued','granuleTitles_filtered_agg','wordCount_filtered'])

# Group concatenated filtered titles by date
date_titles_filtered_df = titles_filtered_df.groupby('dateIssued').agg({'granuleTitles_filtered_agg': np.sum}) 

# Group word count lists of titles by date
date_filtered_wc = []
date_idx_ls = [list(titles_filtered_df.dateIssued).index(x) for x in set(list(titles_filtered_df.dateIssued))]
for i in date_idx_ls:
    date_i = titles_filtered_df['dateIssued'][i]
    id_i = titles_filtered_df['packageId'][i]
    wc_i = titles_filtered_df['wordCount_filtered'][i]
    for j in range(len(titles_filtered_df)):
        date_j = titles_filtered_df['dateIssued'][j]
        id_j = titles_filtered_df['packageId'][j]
        wc_j = titles_filtered_df['wordCount_filtered'][j]
        if id_j != id_i and date_j == date_i:
            wc_i.extend(wc_j)
    date_filtered_wc.append((date_i,wc_i))
        
date_wcount_filtered_df = pd.DataFrame(date_filtered_wc,columns = ['dateIssued','wordCount_filtered'])
date_wcount_filtered_df['dateIssued'] = pd.to_datetime(date_wcount_filtered_df['dateIssued'])
date_wcount_filtered_df = date_wcount_filtered_df.sort_values(by = 'dateIssued',ascending=False) 


# In[21]:


# Format words in wordCount lists (for wordcloud presentation)
for j in range(len(date_wcount_filtered_df)):
    date_wcount_filtered_df['wordCount_filtered'][j] = [[i[0].capitalize(),i[1]] for i in date_wcount_filtered_df['wordCount_filtered'][j]]
date_wcount_filtered_df


# In[22]:


# [!DEPRECATED] Word counting based on concatenated filtered titles
date_titles_filtered = date_titles_filtered_df.granuleTitles_filtered_agg
date_titles_filtered_df

titles_filtered_ct_agg = []
title_filtered_ct = dict()
for title in date_titles_filtered:
    for word in title.split():
        if word not in list(title_filtered_ct.keys()):
            title_filtered_ct[word] = 1
        else:
            title_filtered_ct[word] += 1
    title_filtered_ct_sorted = sorted(title_filtered_ct.items(), key=lambda x:x[1],reverse=True)
    title_filtered_ct_sorted = [list(i) for i in title_filtered_ct_sorted]
    titles_filtered_ct_agg.append(title_filtered_ct_sorted)

#date_titles_filtered_df['wordCount_filtered_agg'] = titles_filtered_ct_agg


# ### Analyze titles of daily digest

# #### Text Processing Functions

# In[23]:


def dailyDigest_processing(dailydigest_record): #remove headlines and space, return pure texts
    raw_lines = dailydigest_record.split('\n')
    processed_text = ''
    noise_pattern = re.compile('(.*)\[(.+)\](.*)')
    for i in raw_lines:
        if i != '' and noise_pattern.match(i) is None:
            #processed_lines.append(i)
            processed_text += i
    return processed_text


# #### Text Processing & Formatting

# In[24]:


# Join(merge) a new datateble with pure text daily digest, using packageId as PK [not grouped by days]
dailyDigest_processed = []
for i in CREC_dailydigest.dailyDigest:
    dailyDigest_processed.append(dailyDigest_processing(i))
CREC_dailydigest['dailyDigest_processed'] = dailyDigest_processed
CREC_dailydigest


# In[25]:


# Format dailydigest_today table (only include daily digests for the latest date)
dailydigest_today_df['title_processed'] = [re.sub('Daily Digest/','',i) for i in dailydigest_today_df.title ]
dailydigest_today_df['content_processed'] = [re.sub('(.*)\[(.+)\](.*)','',i) for i in dailydigest_today_df.content]
dailydigest_today_df['content_processed'] = [re.sub(r'\n\n\n','',i) for i in dailydigest_today_df.content_processed]
dailydigest_today_df['content_processed'] = [i.strip() for i in dailydigest_today_df.content_processed]
dailydigest_today_df


# In[825]:


# Future ideas: topic model → analyze contents?


# # DEPLOYMENT

# In[26]:


app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
    html.H1('Congressional Records Daily Briefing'),
    html.P('Congressional Record Tracking Dashboard: Help to identifying emerging political directions based on CREC data from govInfo.api.')
    ],style={'width': '100%',  'display': 'inline-block'}),
    
    html.Div([
        html.H2('Daily Digest'),
        html.H4(dailydigest_today_df['title_processed'][0]),
        html.P(dailydigest_today_df['content_processed'][0]),
        html.H4(dailydigest_today_df['title_processed'][1]),
        html.P(dailydigest_today_df['content_processed'][1]),
        html.H4(dailydigest_today_df['title_processed'][2]),
        html.P(dailydigest_today_df['content_processed'][2]),
        html.H4(dailydigest_today_df['title_processed'][3]),
        html.P(dailydigest_today_df['content_processed'][3]),
    ], style={'width': '30%', 'whiteSpace': 'pre-wrap', 'display': 'inline-block'}),      
    
    html.Div([
        html.H2("How active the Congress has been?"),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": CREC_summary['dateIssued'],
                        "y": num_granules_df['granulesCount'],
                        "type": "bar",
                    },
                ],
                "layout": {"title": 'Number of Record Granules per Day'},
            },
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": CREC_summary['dateIssued'],
                        "y": num_pages_df['pages'],
                        "type": "bar",
                    },
                ],
                "layout": {"title": 'Number of Record Pages per Day'},
            },
        )
    ], style={'width': '40%','float': 'left', 'display': 'inline-block'}),
    
    html.Div([
        html.H2("What's discussed by the House and Senate?"),
        #html.Img(id="image_wc"),
        html.P(['From CREC of ',date_wcount_filtered_df['dateIssued'][0].strftime('%D'),': ' ]),
        DashWordcloud(
            id='wc_1',
            list=date_wcount_filtered_df['wordCount_filtered'][0],
            width=350, height=350,
            #color = 'random-dark',
            backgroundColor='#F5F5F5', #'#001f00'
            weightFactor = 10,
            shuffle=False,
            rotateRatio=0.5,
            shrinkToFit=True,
            hover=True
            ),
        html.P(['From CREC of ',date_wcount_filtered_df['dateIssued'][1].strftime('%D'),': '  ]),
        DashWordcloud(
            id='wc_2',
            list=date_wcount_filtered_df['wordCount_filtered'][1],
            width=350, height=350,
            gridSize=10,
            #color='random-dark',
            backgroundColor='#F5F5F5', #'#001f00'
            weightFactor = 10,
            shuffle=False,
            rotateRatio=0.5,
            shrinkToFit=True,
            hover=True
            ),
    ], style={'width': '30%', 'float': 'left', 'display': 'inline-block'}),


])


# In[ ]:


if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




