import numpy as np
import nltk
import torch
from sentence_transformers import SentenceTransformer, util
from nltk.cluster import KMeansClusterer
from collections import Counter

from config import settings
import scripts.utils.files as files
import scripts.model_scripts.text_process as proc
from scripts.data_scripts import clear_text as cl

MODEL_SUBDIR = settings.get_fresh('BLOCK_MODELING_SUBDIR')
PROCESSED_SUBDIR = settings.get_fresh('PROCESSED_LOCAL_SUBDIR')
NLTK_DIR = settings.get_fresh('NLTK_DIR')
FILENAME_BLOCK_REDUCER_MODEL = settings.get_fresh('FILENAME_BLOCK_REDUCER_MODEL')
FILENAME_REQUIREMENT_VECTORIZATION_MODEL = settings.get_fresh('FILENAME_REQUIREMENT_VECTORIZATION_MODEL')


MODEL1_NAME='paraphrase_minilm_l12'
MODEL2_NAME= 'distiluse_cased_v2'
MODEL3_NAME='paraphrase_mpnet_v2'
MODEL4_NAME='stsb_xlm_r'

MODEL_FULL_NAMES = {
    MODEL1_NAME:
              'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
   MODEL2_NAME:
         'sentence-transformers/distiluse-base-multilingual-cased-v2',
     MODEL3_NAME:
          'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
     MODEL4_NAME:
          'sentence-transformers/stsb-xlm-r-multilingual'}

MODELS = {
    MODEL1_NAME: None,
    MODEL2_NAME: None,
    MODEL3_NAME: None,
    MODEL4_NAME: None,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kclusterer = None

COL_CODE = 0
COL_COUNT = 1
COL_CANONICAL = 2
COL_UNIGRAMS = 3
COL_BIGRAMS = 4
COL_TRIGRAMS = 5
COL_CLUSTER = 6
COL_U1 = 7
COL_U2 = 8


def get_model(m_key):
    if m_key not in MODELS.keys():
        return None
    if MODELS[m_key] is None:
        MODELS[m_key] = SentenceTransformer(MODELS[m_key])
    return MODELS[m_key]


def get_embeddings(m_key, sentences):
    model = get_model(m_key)
    if model is None:
        return None
    return model.encode(sentences, device=DEVICE, convert_to_tensor=True)


def get_kclusterer(n_clusters):
    kclusterer = KMeansClusterer(n_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
    return kclusterer


def get_cluster_top_n(cluster_model, cluster_label, embeddings, top_count, dataframe, col):
    '''
    Выводит первые top_count и последние top_count элементы кластера
    :param cluster_model: Обученная модель
    :param cluster_label:  Метка кластера, для которого необходимо вывести данные
    :param embeddings: Векторное представление набора данных, для которого определены метки кластеров в модели dbscan_model
    :param top_count: Количество выводимых элементов кластера
    :return: None
    '''
    # embeddings_f = embeddings.astype(float)
    top_k = min(top_count, len(embeddings))
    ds = {}

    # Вычисление среднего значения для кластера
    indexes = np.where(cluster_model.labels_ == cluster_label)
    _list = []
    index_list = []
    for index in indexes[0]:
        index_list.append(index)
        _list.append(embeddings[index])
    arr = np.array(_list)
    mean = arr.mean(axis=0)

    ds['count'] = len(indexes[0])
    _score_list = [[np.linalg.norm(mean - e), i] for i, e in enumerate(embeddings) if i in index_list]
    score_list = sorted(_score_list, key=lambda x: x[0])

    ds['Ближайшие'] = '\n'.join(
        [f"{dataframe.at[result[1], col]} ({result[0]:2.4f})" for result in score_list[:top_k]])
    ds['Самые удаленные'] = '\n'.join(
        [f"{dataframe.at[result[1], col]} ({result[0]:2.4f})" for result in score_list[-top_k:]])
    # print("Ближайшие значения к центру кластера:")
    # for result in score_list[:top_k]:
    #     print("\""+df.at[result[1],'canonical']+"\" Расстояние: "+str(result[0]))
    # print("Самые удаленные от центра кластера значения:")
    # for result in score_list[-top_k:]:
    #     print("\""+df.at[result[1],'canonical']+"\" Расстояние: "+str(result[0]))
    return ds

def get_most_similar(query, n, embeddings, m_key, dataframe, col):
    top_k = min(n, len(embeddings))
    query_embedding = get_embeddings(m_key, query)
    if query_embedding is None:
        return None
    # vect_model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    for score, idx in zip(top_results[0], top_results[1]):
        print(dataframe.at[idx.item(), col] + " " + str(score.item()))


def get_cluster_costop_n(cluster, n, embeddings, dataframe, col):
    if kclusterer is None:
        return None
    embeddings_f = embeddings.astype(float)
    top_k = min(n, len(embeddings))

    cos_scores = util.pytorch_cos_sim(kclusterer.means()[cluster], embeddings_f)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    for score, idx in zip(top_results[0], top_results[1]):
        print(dataframe.at[idx.item(), col] + " Score: " + str(score.item()))


def get_cluster_mean(dataframe):
    arr_umap_1 = np.array(dataframe.iloc[:, COL_U1])
    arr_umap_2 = np.array(dataframe.iloc[:, COL_U2])
    cluster_mean_u1 = arr_umap_1.mean()
    cluster_mean_u2 = arr_umap_2.mean()
    cluster_mean = np.array([cluster_mean_u1, cluster_mean_u2])
    return cluster_mean_u1, cluster_mean_u2, cluster_mean


def get_all_skills(requirements_dict):
    unigrams_counter = Counter()
    bigrams_counter = Counter()
    trigrams_counter = Counter()
    for key in requirements_dict:
        for token in requirements_dict[key]['unigrams']:
            unigrams_counter.update({token: requirements_dict[key]['count']})
        for token in requirements_dict[key]['bigrams']:
            item = proc.join_unigrams(token)
            if item != '':
                bigrams_counter.update({item: requirements_dict[key]['count']})
        for token in requirements_dict[key]['trigrams']:
            item = proc.join_unigrams(token)
            if item != '':
                trigrams_counter.update({item: requirements_dict[key]['count']})
    return unigrams_counter, bigrams_counter, trigrams_counter


def update_keywords(keywords_counter, key, count, only_termins):
    if keywords_counter[key] != 0 or len(key) < 2:
        return
    if only_termins and cl.has_any_russian_symbol(key):
        return
    # for word in keywords_counter:
    #     if util.pytorch_cos_sim(key, keywords_counter)
    keywords_counter.update({key: count})


def get_cluster_skills(requirements_dict, dataframe):
    unigrams_counter, bigrams_counter, trigrams_counter = get_all_skills(requirements_dict)

    all_keywords = Counter()
    keywords = {}

    n_df = len(dataframe)
    n_df_max = int(n_df / 6)
    k_u_add = .05
    k_incl = 0.3
    k_excl = 0.7
    stopwords = [x for x in unigrams_counter if unigrams_counter[x] > n_df_max]
    cluster_indexes = dataframe['cluster'].unique()

    for cluster_id in cluster_indexes:
        print(f'Начинается обработка кластера {cluster_id}')
        df_cl = dataframe[dataframe.cluster == cluster_id]
        n_cl = len(df_cl)
        n_u_min = int(n_cl * k_u_add)

        codes = df_cl.code.tolist()
        unigrams_counter_cl, bigrams_counter_cl, trigrams_counter_cl = (
            get_cluster_counters(codes, requirements_dict, stopwords))

        keywords[cluster_id] = Counter()
        for unigram in unigrams_counter_cl:
            if unigram.isdigit():
                continue

            only_termins = unigrams_counter_cl[unigram] <= n_u_min and cl.has_any_russian_symbol(unigram) == False

            if unigrams_counter_cl[unigram] > n_u_min or only_termins:
                to_add_unigram = unigrams_counter_cl[unigram] > n_u_min
                n_b_min = int(unigrams_counter_cl[unigram] * k_incl)
                n_b_max = int(unigrams_counter_cl[unigram] * k_excl)

                bigrams = [bigram for bigram in bigrams_counter_cl if
                           unigram in bigram and bigrams_counter_cl[bigram] > n_b_min]
                for bigram in bigrams:
                    to_add_bigram = True
                    n_t_min = int(bigrams_counter_cl[bigram] * k_incl)
                    n_t_max = int(bigrams_counter_cl[bigram] * k_excl)

                    trigrams = [trigram for trigram in trigrams_counter_cl if
                                bigram in trigram and trigrams_counter_cl[trigram] > n_t_min]
                    for trigram in trigrams:
                        if only_termins and not cl.has_any_russian_symbol(trigram):
                            update_keywords(all_keywords, trigram, trigrams_counter_cl[trigram], only_termins)
                        elif not only_termins or not cl.has_any_russian_symbol(trigram):
                            update_keywords(keywords[cluster_id], trigram, trigrams_counter_cl[trigram],
                                               only_termins)
                        if trigrams_counter_cl[trigram] > n_t_max:
                            to_add_bigram = False

                    if to_add_bigram:
                        if only_termins and not cl.has_any_russian_symbol(bigram):
                            update_keywords(all_keywords, bigram, bigrams_counter_cl[bigram], only_termins)
                        elif not only_termins or not cl.has_any_russian_symbol(bigram):
                            update_keywords(keywords[cluster_id], bigram, bigrams_counter_cl[bigram], only_termins)
                        if bigrams_counter_cl[bigram] > n_b_max:
                            to_add_unigram = False

                if to_add_unigram:
                    update_keywords(keywords[cluster_id], unigram, unigrams_counter_cl[unigram], only_termins)
                elif only_termins:
                    update_keywords(all_keywords, unigram, unigrams_counter_cl[unigram], only_termins)

        all_keywords.update(keywords[cluster_id])
        print(f'Закончилась обработка кластера {cluster_id}')

    return keywords, all_keywords


def get_cluster_labels(model_name, keywords, dataframe):
    filename = f'{model_name}_{FILENAME_REQUIREMENT_VECTORIZATION_MODEL}'
    model = files.load_model(filename)
    filename = f'{model_name}_{2}_{FILENAME_BLOCK_REDUCER_MODEL}'
    reducer = files.load_model(filename)
    cluster_mean_u1, cluster_mean_u2, cluster_mean = get_cluster_mean(dataframe)

    keywords_list = []
    for cluster_id in keywords:
        print(f'Начинается обработка кластера {cluster_id}')
        cluster_dict = {'cluster_id': cluster_id, 'mean_u1': cluster_mean[0], 'mean_u2': cluster_mean[1]}
        index = 0
        min_distance = 1000
        max_distance = 0
        closest_sentence = ''

        for key in keywords[cluster_id].most_common(10):
            key_embedding = model.encode(key, device=DEVICE)
            key_umap = reducer.transform(key_embedding)
            distance = np.linalg.norm(cluster_mean - key_umap)
            cluster_dict[f'key_{index}'] = key
            cluster_dict[f'count_{index}'] = keywords[cluster_id][key]
            cluster_dict[f'distance_{index}'] = distance
            cluster_dict[f'u1_{index}'] = key_umap[0]
            cluster_dict[f'u2_{index}'] = key_umap[1]
            index += 1

            if (distance < min_distance):
                min_distance = distance
                closest_sentence = key
            if distance > max_distance:
                max_distance = distance

        cluster_dict['min_distance'] = min_distance
        cluster_dict['max_distance'] = max_distance
        cluster_dict['label'] = closest_sentence

        keywords_list.append(cluster_dict)
        print(f'Закончилась обработка кластера {cluster_id}')

    return keywords_list


def get_cluster_counters(codes, requirements_dict, stopwords):
    unigrams_counter_cl = Counter()
    bigrams_counter_cl = Counter()
    trigrams_counter_cl = Counter()

    for code in codes:
        for token in requirements_dict[code]['unigrams']:
            if token in stopwords:
                continue
            unigrams_counter_cl.update([token])

        for token in requirements_dict[code]['bigrams']:
            if any(part_token in stopwords for part_token in token):
                continue
            item = proc.join_unigrams(token)
            if item != '':
                bigrams_counter_cl.update([item])

        for token in requirements_dict[code]['trigrams']:
            if any(part_token in stopwords for part_token in token):
                continue
            item = proc.join_unigrams(token)
            if item != '':
                trigrams_counter_cl.update([item])
    return unigrams_counter_cl, bigrams_counter_cl, trigrams_counter_cl
