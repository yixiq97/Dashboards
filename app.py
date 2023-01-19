#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px 


# In[2]:


import re
import spacy
try:
    nlp = spacy.load("en_core_web_md")
except:  # If not present, we download
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
STOPWORDS = nlp.Defaults.stop_words
stopwords = STOPWORDS # customize (TBD)


# In[431]:

import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import Dash, dcc, html, Input, Output
from dash_holoniq_wordcloud import DashWordcloud
import dash_bootstrap_components as dbc


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



# # PROCESS DATATABLE

# In[15]:
collection_json = get_CREC_packages(10)
collection_attr = list(collection_json[0].keys())
packageLinks = [r['packageLink'] for r in collection_json]
collection_df = pd.DataFrame(collection_json)[['packageId', 'dateIssued', 'title', 'lastModified']]

# get summary of each CERC record, in 1:1 relationship
summary_data = [get_CREC_summary(packageLink) for packageLink in packageLinks]
summary_attr = list(summary_data[0].keys())
granulesLinks = [r['granulesLink'] for r in summary_data]
record_info_df = pd.DataFrame(summary_data)[['packageId', 'detailsLink', 'pages', 'governmentAuthor1', 'category']]

# get granule contents of each CERC record, in 1:M relationship
granules_relation = [(granulesLink, get_CREC_granules(granulesLink)[0], get_CREC_granules(granulesLink)[1])
                     for granulesLink in granulesLinks]
granules_backward_mapping = dict([(r['granulesLink'], r['packageId']) for r in summary_data])
names = globals()
num_of_granules_table = 0
names_of_granules_table = []
for granules in granules_relation:
    gran_link = granules[0]
    pack_id = granules_backward_mapping[gran_link]
    pack_id_formatted = pack_id.replace('-', '')
    names['granules_%s_df' % pack_id_formatted] = pd.DataFrame(granules[2])[['granuleId', 'title', 'granuleClass']]
    eval('granules_%s_df' % pack_id_formatted)['packageId'] = pack_id
    names_of_granules_table.append('granules_%s_df' % pack_id_formatted)
    # return to table i: eval(names_of_granules_table[i])
    num_of_granules_table += 1
granules_count_df = pd.DataFrame(granules_relation, columns=['granulesLink', 'granulesCount', 'json_file'])[
    ['granulesLink', 'granulesCount']]
package_granule_df = pd.DataFrame.from_dict(granules_backward_mapping, orient='index', columns=['packageId'])
package_granule_df = package_granule_df.reset_index().rename(columns={'index': 'granulesLink'})

# In[79]:



# In[18]:


# Join(merge) and get original datatables
CREC_summary = pd.merge(collection_df,record_info_df,on = 'packageId')
CREC_summary['pages'] = CREC_summary.pages.astype('int64')
CREC_granules =  pd.merge(pd.merge(collection_df,package_granule_df,on = 'packageId'),
                          granules_count_df,on = 'granulesLink')


# # DATA ANALYTICS

# ## Analyze Daily Active Levels

# In[70]:


num_pages_df = CREC_summary.groupby('dateIssued').agg({'pages': np.sum})
num_granules_df = CREC_granules.groupby('dateIssued').agg({'granulesCount': np.sum})
num_pages_df['dateIssued'] = num_pages_df.index
num_granules_df['dateIssued'] = num_pages_df.index


# In[80]:


# ## Analyze Titles of Daily Granules

# #### Text Processing Functions

# In[112]:


# Fancier text processing method using SpaCy
def get_one_package(package_id,granuleClass):
    #package_id = CREC_summary.packageId[i]
    granules_df = eval('granules_%s_df' % package_id.replace('-',''))
    #S_H_df = granules_df[(granules_df['granuleClass'] == 'SENATE')|(granules_df['granuleClass'] == 'HOUSE')]
    S_H_df = granules_df[granules_df['granuleClass'] == str(granuleClass).upper()]
    S_H_titles = list(S_H_df.title)
    S_H_titles_parsed = []
    for title in S_H_titles:
        S_H_titles_parsed.append(nlp(title))
    return S_H_titles_parsed


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


# In[108]:


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

# In[226]:


# Format words in wordCount lists (for wordcloud presentation)
def wc_formatted(count_filtered_agg):
    count_filtered_agg_fmt = []
    for record in count_filtered_agg:
        count_filtered_agg_fmt.append([[i[0].capitalize(),i[1]] for i in record])
    return count_filtered_agg_fmt


# In[244]:


# Normalize the text data
def wc_percent(count_filtered_agg_fmt):
    count_filtered_agg_pct = []
    record_wc_sum = []

    for record in count_filtered_agg_fmt:
        count_sum = 0
        for word in record:
            count_sum += word[1]
        record_wc_sum.append(count_sum)
    
        count_filtered_record_pct = []
        for word in record:
            word_pct = word[0]
            wc_pct = round(word[1]/count_sum * 100,2)
            count_filtered_record_pct.append([word_pct,wc_pct])
        count_filtered_agg_pct.append(count_filtered_record_pct)
        
    return count_filtered_agg_pct


# In[217]:


# Get word count lists of titles, filtered based on certain nouns, pos, and entity type 
count_filtered_agg_H = []
count_filtered_agg_S = []
count_filtered_agg = []
for i in range(len(CREC_summary)):
    parsed_titles_H = get_parsed_titles(
        get_one_package(CREC_summary.packageId[i],'house')
    )
    parsed_titles_S = get_parsed_titles(
        get_one_package(CREC_summary.packageId[i],'senate')
    )
    parsed_titles_H_S = parsed_titles_H.copy()
    parsed_titles_H_S.extend(parsed_titles_S)
    
    count_filtered_i_H = wordCount_pro(parsed_titles_H,important_pos,important_ents)
    count_filtered_i_S = wordCount_pro(parsed_titles_S,important_pos,important_ents)
    count_filtered_i_H_S = wordCount_pro(parsed_titles_H_S,important_pos,important_ents)
       
    count_filtered_i_H = [list(i) for i in count_filtered_i_H]
    count_filtered_i_S = [list(i) for i in count_filtered_i_S]
    count_filtered_i_H_S = [list(i) for i in count_filtered_i_H_S]

    count_filtered_agg_H.append(count_filtered_i_H)
    count_filtered_agg_S.append(count_filtered_i_S)
    count_filtered_agg.append(count_filtered_i_H_S)
#count_filtered_agg_H   #len(count_filtered_agg) = len(CREC_summary), but len(count_filtered_agg[i]) differs across different i
#count_filtered_agg_S
#count_filtered_agg


# In[245]:


count_filtered_agg_formatted = wc_formatted(count_filtered_agg)
count_filtered_agg_pct = wc_percent(count_filtered_agg_formatted)

count_filtered_agg_H_formatted = wc_formatted(count_filtered_agg_H)
count_filtered_agg_H_pct = wc_percent(count_filtered_agg_H_formatted)

count_filtered_agg_S_formatted = wc_formatted(count_filtered_agg_S)
count_filtered_agg_S_pct = wc_percent(count_filtered_agg_S_formatted)


# In[326]:


# Build new datatable joining concatenated titles and wordCount list, with packageId as PK
titles_filtered_df = pd.DataFrame(zip(CREC_summary.packageId,CREC_summary.dateIssued,count_filtered_agg,count_filtered_agg_formatted,count_filtered_agg_pct,count_filtered_agg_H_formatted,count_filtered_agg_H_pct,count_filtered_agg_S_formatted,count_filtered_agg_S_pct),
                                  columns = ['packageId','dateIssued','wordCount_filtered','wordCount_filtered_formatted','wordCount_filtered_formatted_pct','wordCount_filtered_H_formatted','wordCount_filtered_H_formatted_pct','wordCount_filtered_S_formatted','wordCount_filtered_S_formatted_pct'])

# Group word count lists of titles by date
date_filtered_wc = []
date_idx_ls = [list(titles_filtered_df.dateIssued).index(x) for x in set(list(titles_filtered_df.dateIssued))]
for i in date_idx_ls:
    date_i = titles_filtered_df['dateIssued'][i]
    id_i = titles_filtered_df['packageId'][i]
    wc_fmt_i = titles_filtered_df['wordCount_filtered_formatted'][i]
    wc_fmt_pct_i = titles_filtered_df['wordCount_filtered_formatted_pct'][i]
    wc_H_fmt_i = titles_filtered_df['wordCount_filtered_H_formatted'][i]
    wc_H_fmt_pct_i = titles_filtered_df['wordCount_filtered_H_formatted_pct'][i]
    wc_S_fmt_i = titles_filtered_df['wordCount_filtered_S_formatted'][i]
    wc_S_fmt_pct_i = titles_filtered_df['wordCount_filtered_S_formatted_pct'][i]
    for j in range(len(titles_filtered_df)):
        date_j = titles_filtered_df['dateIssued'][j]
        id_j = titles_filtered_df['packageId'][j]
        wc_fmt_j = titles_filtered_df['wordCount_filtered_formatted'][j]
        wc_fmt_pct_j = titles_filtered_df['wordCount_filtered_formatted_pct'][j]
        wc_H_fmt_j = titles_filtered_df['wordCount_filtered_H_formatted'][j]
        wc_H_fmt_pct_j = titles_filtered_df['wordCount_filtered_H_formatted_pct'][j]
        wc_S_fmt_j = titles_filtered_df['wordCount_filtered_S_formatted'][j]
        wc_S_fmt_pct_j = titles_filtered_df['wordCount_filtered_S_formatted_pct'][j]
        if id_j != id_i and date_j == date_i:
            wc_fmt_i.extend(wc_fmt_j)
            wc_fmt_pct_i.extend(wc_fmt_pct_j)
            wc_H_fmt_i.extend(wc_H_fmt_j)
            wc_H_fmt_pct_i.extend(wc_H_fmt_pct_j)
            wc_S_fmt_i.extend(wc_S_fmt_j)
            wc_S_fmt_pct_i.extend(wc_S_fmt_pct_j)
    date_filtered_wc.append((date_i,wc_fmt_i,wc_fmt_pct_i,wc_H_fmt_i,wc_H_fmt_pct_i,wc_S_fmt_i,wc_S_fmt_pct_i))
        
date_wcount_filtered_df = pd.DataFrame(date_filtered_wc,columns = ['dateIssued','wordCount_filtered_formatted','wordCount_filtered_formatted_pct','wordCount_filtered_H_formatted','wordCount_filtered_H_formatted_pct','wordCount_filtered_S_formatted','wordCount_filtered_S_formatted_pct'])
date_wcount_filtered_df['dateIssued'] = pd.to_datetime(date_wcount_filtered_df['dateIssued'])
date_wcount_filtered_df = date_wcount_filtered_df.sort_values(by = 'dateIssued',ascending=False) 


# In[334]:


date_wcount_filtered_df['dateIssued'] = pd.to_datetime(date_wcount_filtered_df['dateIssued']).dt.date
date_wcount_filtered_sorted_df = date_wcount_filtered_df.sort_values(by = 'dateIssued',ascending=False)
date_wcount_filtered_sorted_df = date_wcount_filtered_sorted_df.reset_index(drop = True)
date_wcount_filtered_sorted_df['dateIssued'] = date_wcount_filtered_sorted_df.dateIssued.astype(str)
date_wcount_filtered_sorted_df


# #### Plotting

# In[359]:


def get_lemma_count(wc_records,date,num_top_lemmas_to_plot):
    idx = date_wcount_filtered_sorted_df[date_wcount_filtered_sorted_df['dateIssued']==date].index[0]
    wc_record = wc_records[idx]
    top_lemmas = []
    top_counts = []
    for r in wc_record[:num_top_lemmas_to_plot]:
        top_lemmas.append(r[0])
        top_counts.append(r[1])
    return top_lemmas,top_counts
#top_lemmas,top_counts = get_lemma_count(date_wcount_filtered_sorted_df['wordCount_filtered_formatted'],'2023-01-17',20)


# In[353]:


def get_count_S_H(wc_record_S_H, date, all_top_lemmas):
    idx = date_wcount_filtered_sorted_df[date_wcount_filtered_sorted_df['dateIssued'] == date].index[0]
    wc_record_S_H = wc_record_S_H[idx]
    counts_S_H = np.zeros(len(all_top_lemmas))
    for i in range(len(all_top_lemmas)):
        top_lemma = all_top_lemmas[i]
        for r in wc_record_S_H:
            lemma, count = r[0], r[1]
            if lemma == top_lemma:
                counts_S_H[i] = count

    return list(counts_S_H)

#top_counts_H = get_count_S_H(date_wcount_filtered_sorted_df['wordCount_filtered_H_formatted'],'2023-01-17',top_lemmas)
#top_counts_S = get_count_S_H(date_wcount_filtered_sorted_df['wordCount_filtered_S_formatted'],'2023-01-17',top_lemmas)


# # DEPLOYMENT

# In[423]:

app = dash.Dash(external_stylesheets=[dbc.themes.SKETCHY])
server = app.server

# In[455]:


app.layout = html.Div([
    html.Div([
        html.H1('Congressional Records Daily Briefing'),
        html.Div(children=[
            # html.H3('More details information can refer to daily digest in '),
            html.P(
                'Congressional record tracking dashboard helps to show emerging political potentials based on CREC data from GovInfo API. You can also check out the daily digests released by the congress record.',
                style={'fontSize': '24px'}),
            html.A(href="https://www.congress.gov/congressional-record",
                   children=[html.Button('Latest Daily Digest', className="btn btn-primary disabled")],
                   style={'float': 'right', 'margin-right': '50px', 'margin-top': '0px', 'margin-bottom': '20px',
                          'display': 'inline-block'}),
        ], style={'float': 'left', 'display': 'inline-block'}),
    ]),

    html.Div([
        html.H2('Top words in Senate and House', className='card-header'),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.P("Issued date", style={'fontSize': '18px'}, className='card-text'),
                        dcc.Dropdown(
                            id="date-filter",
                            options=[
                                {"label": date, "value": date}
                                for date in date_wcount_filtered_sorted_df.dateIssued
                            ],
                            value=date_wcount_filtered_sorted_df.dateIssued[0],
                            clearable=False,
                            className="form-label mt-4",
                        ),
                    ]),
                dcc.Graph(
                    id="word-freq-comparison-chart", config={"displayModeBar": False},
                ),
            ], className='card-body'
        ),
        # html.H5('Recent top words in granules discussed in Senate and House in different days'),
    ], className="card border-primary mb-3",
        style={'width': '93%',
               'float': 'left',
               'margin-left': '50px',
               'display': 'inline-block',
               }),  # style={'width': '60%', 'whiteSpace': 'pre-wrap', 'display': 'inline-block'}

    html.Div(
        children=[
            html.H2("What's discussed by the House and Senate?", className='card-header'),
            html.Div(
                children=[
                    html.Div(children=[
                        html.H5(['Word Cloud of ', date_wcount_filtered_sorted_df['dateIssued'][0]]),
                        DashWordcloud(
                            id='wordcloud1',
                            list=date_wcount_filtered_sorted_df.wordCount_filtered_formatted_pct[0],
                            width=350, height=350,
                            # color = 'random-dark',
                            # backgroundColor='#F5F5F5', #'#001f00'
                            weightFactor=10,
                            shuffle=False,
                            rotateRatio=0.5,
                            shrinkToFit=True,
                            hover=True
                        )], style={'width': '30%', 'textAlign': 'center', 'float': 'left', 'display': 'inline-block'}
                    ),
                    html.Div(children=[
                        html.H5(['Word Cloud of ', date_wcount_filtered_sorted_df['dateIssued'][1]]),
                        DashWordcloud(
                            id='wordcloud2',
                            list=date_wcount_filtered_sorted_df.wordCount_filtered_formatted_pct[1],
                            width=350, height=350,
                            # color = 'random-dark',
                            # backgroundColor='#F5F5F5', #'#001f00'
                            weightFactor=10,
                            shuffle=False,
                            rotateRatio=0.5,
                            shrinkToFit=True,
                            hover=True
                        )],
                        style={'width': '30%', 'textAlign': 'center', 'margin-left': '60px', 'display': 'inline-block'}
                    ),
                    html.Div(children=[
                        html.H5(['Word Cloud of ', date_wcount_filtered_sorted_df['dateIssued'][2]]),
                        DashWordcloud(
                            id='wordcloud3',
                            list=date_wcount_filtered_sorted_df.wordCount_filtered_formatted_pct[2],
                            width=350, height=350,
                            # color = 'random-dark',
                            # backgroundColor='#F5F5F5', #'#001f00'
                            weightFactor=10,
                            shuffle=False,
                            rotateRatio=0.5,
                            shrinkToFit=True,
                            hover=True
                        )], style={'width': '30%', 'textAlign': 'center', 'float': 'right', 'display': 'inline-block'}
                    ),
                    html.P('* Notice: word clouds above are generated based on a relative term', className='lead'),
                ], className='card-body'
            ),
        ], className="card border-primary mb-3",
        style={'width': '93%',
               'float': 'left',
               'margin-left': '50px',
               'margin-right': '50px',
               'display': 'inline-block',
               }),  # style={'width': '100%', 'display': 'inline-block'}

    html.Div([
        html.H2("How active the Congress has been?", className='card-header'),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.P("Index type", style={'fontSize': '18px'}, className='card-text'),
                        dcc.Dropdown(
                            id="activity-type-filter",
                            options=[
                                {"label": activity_type, "value": activity_type}
                                for activity_type in ['Record Granules', 'Record Pages']
                            ],
                            value="Record Granules",
                            clearable=False,
                            className="form-label mt-4",
                        ),
                    ]),
                dcc.Graph(
                    id="activity-chart", config={"displayModeBar": False}
                ),
                # html.H5('Different indexes help to identify the active level of congress in recent days'),
            ], className='card-body'
        ),
    ], className="card border-primary mb-3",
        style={'width': '93%',
               'float': 'left',
               'margin-left': '50px',
               'display': 'inline-block',
               }),  # style={'width': '40%','display': 'inline-block'}

],
)


# In[458]:
@app.callback(
    [Output("word-freq-comparison-chart", "figure"), Output("activity-chart", "figure")],
    [Input("date-filter", "value"), Input("activity-type-filter", "value"), ],
)
def update_charts(date, activity_type):
    top_lemmas = get_lemma_count(date_wcount_filtered_sorted_df['wordCount_filtered_formatted'], date, 20)[0]
    wordfreq_comparison_chart = {
        "data": [
            {
                "x": top_lemmas,
                "y": get_count_S_H(date_wcount_filtered_sorted_df['wordCount_filtered_H_formatted'], date, top_lemmas),
                "type": "bar", 'name': 'House'
            },
            {
                "x": top_lemmas,
                "y": get_count_S_H(date_wcount_filtered_sorted_df['wordCount_filtered_S_formatted'], date, top_lemmas),
                "type": "bar", 'name': 'Senate'
            },
        ],
        "layout": {"title": 'Senate vs House: Top words distibution pattern'},
    }

    if activity_type == 'Record Granules':
        activity_chart = {
            "data": [
                {
                    "x": num_granules_df['dateIssued'],
                    "y": num_granules_df['granulesCount'],
                    "type": "bar",
                },
            ],
            "layout": {"title": 'Number of Record Granules per Day'},
        }
    else:
        activity_chart = {
            "data": [
                {
                    "x": num_pages_df['dateIssued'],
                    "y": num_pages_df['pages'],
                    "type": "bar",
                },
            ],
            "layout": {"title": 'Number of Record Pages per Day'},
        }

    return wordfreq_comparison_chart, activity_chart

# In[459]:


if __name__ == '__main__':
    app.run_server(debug=True)




