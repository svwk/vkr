#! python3
# -*- coding: UTF-8 -*-

# %% Импорт
import pandas as pd
import plotly.express as px
import warnings
from config import settings

import scripts.utils.files as files


warnings.simplefilter("ignore")

def plot_clusters(dataset, m_key, model_dict):
    fig = px.scatter(dataset,f'{m_key}_umap_0', f'{m_key}_umap_1', hover_data = ['canonical'], title = f"Pre-trained Model:{model_dict[m_key]}",color = f'{m_key}_dbscan_1')
    fig.update_traces(marker=dict(size=2.5), opacity=0.3)
    # fig.update_layout(margin=go.layout.Margin(l=1, r=0,b=0, t=45 ),
    #                      plot_bgcolor = 'rgb(240,245,255)',
    #                      xaxis_title="",
    #                      yaxis_title="",
    #                      font=dict(size=20),
    #                      showlegend = False
    #                      #yaxis_tickformat = '%'
    #                      )
    fig.show()
    return fig

# %%
model_subdir = settings.get_fresh('BLOCK_MODELING_SUBDIR')
filename_emb_suffix = settings.get_fresh('FILENAME_EMB_SUFFIX')
filename_umap_suffix = settings.get_fresh('FILENAME_UMAP_SUFFIX')
filename_dbscan_suffix = settings.get_fresh('FILENAME_DBSCAN_SUFFIX')

filename = settings.get_fresh('REQUIREMENTS_TOKENIZED')
requirements_dict=files.load_data_dump(filename, model_subdir)
df = pd.DataFrame.from_dict(requirements_dict, "index")

model1_name='paraphrase_minilm_l12'
model2_name= 'distiluse_cased_v2'
model3_name='paraphrase_mpnet_v2'
model4_name='stsb_xlm_r'

models = {model1_name:
              'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
       model2_name:
             'sentence-transformers/distiluse-base-multilingual-cased-v2',
         model3_name:
              'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
         model4_name:
              'sentence-transformers/stsb-xlm-r-multilingual'}

embeddings={}
for m_key in models:
    filename = f'{m_key}_{filename_emb_suffix}'
    embeddings[m_key]=files.load_data_dump(filename, model_subdir)

umap_embeddings = {}
for m_key in models:
    filename = f'{m_key}_{filename_umap_suffix}'
    umap_embeddings[m_key] = files.load_data_dump(filename, model_subdir)

for m_key in models:
    df[f'{m_key}_umap_0']= [i[0] for i in umap_embeddings[m_key]]
    df[f'{m_key}_umap_1']= [i[1] for i in umap_embeddings[m_key]]

dbscan_embeddings={}
for m_key in models:
    filename = f'{m_key}_{filename_dbscan_suffix}'
    dbscan_embeddings[m_key]=files.load_data_dump(filename, model_subdir)
    df[f'{m_key}_dbscan_0'] = dbscan_embeddings[m_key].labels_
    df[f'{m_key}_dbscan_1'] = dbscan_embeddings[m_key].labels_.astype(str)
# %%
# plot_clusters(df, model2_name, models)
# fig.show()

fig = px.scatter(df,f'{m_key}_umap_0', f'{m_key}_umap_1', hover_data = ['canonical'], title = f"Pre-trained Model:{models[m_key]}",color = f'{m_key}_dbscan_1')
# fig.update_traces(marker=dict(size=2.5), opacity=0.3)
# fig.update_layout(margin=go.layout.Margin(l=1, r=0,b=0, t=45 ),
#                      plot_bgcolor = 'rgb(240,245,255)',
#                      xaxis_title="",
#                      yaxis_title="",
#                      font=dict(size=20),
#                      showlegend = False
#                      #yaxis_tickformat = '%'
#                      )
fig.show()


