import re

data_path = ""

import scipy
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

# show all columns
pd.set_option('display.max_columns', None)


def get_rec(my_profile, num_of_rec):
    # load csv file
    df = pd.read_csv(data_path + "data/pre_processed.csv")

    df.rename(columns={
        'orderedBy': 'user_id',
        'age': 'age',
        'food_cuisine': 'cuisine',
        'food': 'food_id',
        'food_name': 'food_name',
        'food_type': 'food_type',
        'ingredients': 'Ingredients',
        'feedback': 'food_rating'
    }, inplace=True)

    # # unique food_id
    # print("\nUnique Foods:")
    # print(df['food_name'].unique())
    # print(df['food_name'].value_counts().tail())

    # # unique cuisine
    # print("\nUnique Cuisine:")
    # print(df['cuisine'].unique())
    # print(df['cuisine'].value_counts().tail())

    # # print number of rows
    # print("\nNumber of Rows:")
    # print(df.shape[0])
    # remove duplicates, null values
    df = df.drop_duplicates()
    df = df.dropna()
    # # print number of rows
    # print("\n:Number of rows after duplicates are removed")
    # print(df.shape[0])

    #############################################################################################

    # keep only user_id, food_name, food_rating columns in new df
    new_df = df[['user_id', 'food_id', 'food_rating']]

    # By taking user_id, food_id as index, we can easily get the food_rating for a given user_id and food_id
    new_df = new_df.set_index(['user_id', 'food_id'])

    # print("\n:After Rename")
    # print(new_df.head(5))
    # print()

    # # find if more than one food_rating exists for a given user_id and food_id
    # print("Duplicate Count for each user for each food")
    # print(new_df.index.duplicated().sum())
    #
    # # print duplicate rows sorted by user_id
    # print("\n:Print Duplicates")
    # print(new_df[new_df.index.duplicated()].sort_index())

    # remove duplicate rows with sum of food_rating
    new_df = new_df.groupby(level=[0, 1]).sum()

    # print(new_df[new_df.index.duplicated()].sort_index())
    #
    # print("\n:After Duplicate Removed")
    # print(new_df.head())

    # separate user_id and food_id from index
    new_df = new_df.reset_index(level=[0, 1])

    # print("\n:Separate user and food")
    # print(new_df.head())

    # randomly mix the rows
    new_df = new_df.sample(frac=1).reset_index(drop=True)

    # rename to personId	contentId	eventStrength
    new_df = new_df.rename(columns={'user_id': 'personId', 'food_id': 'contentId', 'food_rating': 'eventStrength'})

    # print("\n:Final")
    # print(new_df.head(20))

    # rename to personId	contentId	eventStrength
    df = df.rename(
        columns={'user_id': 'personId', 'food_id': 'contentId', 'food_rating': 'eventStrength', 'food_name': 'title',
                 'Ingredients': 'text', 'food_rating': 'eventStrength'})

    # print(df.head())
    articles_df = df

    interactions_full_df = new_df
    # print(interactions_full_df)

    # # create a sparse matrix
    # sparse_matrix = pd.pivot_table(new_df, values='eventStrength', index='personId', columns='contentId', fill_value=0)
    #
    # print("\n:Sparse Matrix")
    # print(sparse_matrix.head())

    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)

    # print('# interactions on Train set: %d' % len(interactions_train_df))
    # print('# interactions on Test set: %d' % len(interactions_test_df))

    # Indexing by personId to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index('personId')
    interactions_train_indexed_df = interactions_train_df.set_index('personId')
    interactions_test_indexed_df = interactions_test_df.set_index('personId')

    def get_items_interacted(person_id, interactions_df):
        # Get the user's data and merge in the movie information.
        interacted_items = interactions_df.loc[person_id]['contentId']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    # Top-N accuracy metrics consts
    EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 10

    class ModelEvaluator:

        def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
            interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
            all_items = set(articles_df['contentId'])
            non_interacted_items = all_items - interacted_items

            random.seed(seed)
            non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
            return set(non_interacted_items_sample)

        def _verify_hit_top_n(self, item_id, recommended_items, topn):
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

        def evaluate_model_for_user(self, model, person_id):
            # Getting the items in test set
            interacted_values_testset = interactions_test_indexed_df.loc[person_id]
            if type(interacted_values_testset['contentId']) == pd.Series:
                person_interacted_items_testset = set(interacted_values_testset['contentId'])
            else:
                person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])
            interacted_items_count_testset = len(person_interacted_items_testset)

            # Getting a ranked recommendation list from a model for a given user
            person_recs_df = model.recommend_items(person_id,
                                                   items_to_ignore=get_items_interacted(person_id,
                                                                                        interactions_train_indexed_df),
                                                   topn=10000000000)

            hits_at_5_count = 0
            hits_at_10_count = 0
            # For each item the user has interacted in test set
            for item_id in person_interacted_items_testset:
                # Getting a random sample (100) items the user has not interacted
                # (to represent items that are assumed to be no relevant to the user)
                # keep only numbers in item_id
                # if item_id is not a number
                # print("item_id: ", item_id)
                if not str(item_id).isdigit():
                    num_item_id = re.sub("[^0-9]", "", item_id)
                    if not num_item_id:
                        # random number
                        num_item_id = random.randint(0, 1000000)
                    # keep only 5 digits
                    num_item_id = int(num_item_id[-5:])
                else:
                    num_item_id = item_id

                non_interacted_items_sample = self.get_not_interacted_items_sample(person_id,
                                                                                   sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                                   seed=num_item_id % (2 ** 32))

                # Combining the current interacted item with the 100 random items
                items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

                # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
                valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]
                valid_recs = valid_recs_df['contentId'].values
                # Verifying if the current interacted item is among the Top-N recommended items
                hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
                hits_at_5_count += hit_at_5
                hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
                hits_at_10_count += hit_at_10

            # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
            # when mixed with a set of non-relevant items
            recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
            recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

            person_metrics = {'hits@5_count': hits_at_5_count,
                              'hits@10_count': hits_at_10_count,
                              'interacted_count': interacted_items_count_testset,
                              'recall@5': recall_at_5,
                              'recall@10': recall_at_10}
            return person_metrics

        def evaluate_model(self, model):
            # print('Running evaluation for users')
            people_metrics = []
            for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
                # if idx % 100 == 0 and idx > 0:
                #    print('%d users processed' % idx)
                person_metrics = self.evaluate_model_for_user(model, person_id)
                person_metrics['_person_id'] = person_id
                people_metrics.append(person_metrics)
            print('%d users processed' % idx)

            detailed_results_df = pd.DataFrame(people_metrics) \
                .sort_values('interacted_count', ascending=False)

            global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
                detailed_results_df['interacted_count'].sum())
            global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
                detailed_results_df['interacted_count'].sum())

            global_metrics = {'modelName': model.get_model_name(),
                              'recall@5': global_recall_at_5,
                              'recall@10': global_recall_at_10}
            return global_metrics, detailed_results_df

    model_evaluator = ModelEvaluator()

    # Computes the most popular items
    item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(
        ascending=False).reset_index()
    print("Most popular items: ")
    print(item_popularity_df.head(10))

    class PopularityRecommender:
        MODEL_NAME = 'Popularity'

        def __init__(self, popularity_df, items_df=None):
            self.popularity_df = popularity_df
            self.items_df = items_df

        def get_model_name(self):
            return self.MODEL_NAME

        def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
            # Recommend the more popular items that the user hasn't seen yet.
            recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
                .sort_values('eventStrength', ascending=False) \
                .head(topn)

            if verbose:
                if self.items_df is None:
                    raise Exception('"items_df" is required in verbose mode')

                recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                              left_on='contentId',
                                                              right_on='contentId')[
                    ['eventStrength', 'contentId', 'title', 'veg', 'cuisine']]

            return recommendations_df

    popularity_model = PopularityRecommender(item_popularity_df, articles_df)

    # print(item_popularity_df)
    # print("\n\n\n\n")
    # print(df)

    print('Evaluating Popularity recommendation model...')
    pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
    print('\nGlobal metrics:\n%s' % pop_global_metrics)
    print(pop_detailed_results_df.head(10))

    ####################################################################### Content-Based Filtering model

    import nltk
    nltk.download('stopwords')

    # Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
    stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

    # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1, 2),
                                 min_df=0.003,
                                 max_df=0.5,
                                 max_features=5000,
                                 stop_words=stopwords_list)

    item_ids = articles_df['contentId'].tolist()
    tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
    tfidf_feature_names = vectorizer.get_feature_names_out()
    print(tfidf_matrix)

    def get_item_profile(item_id):
        idx = item_ids.index(item_id)
        item_profile = tfidf_matrix[idx:idx + 1]
        return item_profile

    def get_item_profiles(ids):
        item_profiles_list = [get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_users_profile(person_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[person_id]
        user_item_profiles = get_item_profiles(interactions_person_df['contentId'])

        user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
        # Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
            user_item_strengths)

        user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
        # Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
            user_item_strengths)
        # print(user_item_strengths_weighted_avg.shape)
        user_item_strengths_weighted_avg = np.asarray(user_item_strengths_weighted_avg)
        # print(type(user_item_strengths_weighted_avg))
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
        return user_profile_norm

    def build_users_profiles():
        interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
            .isin(articles_df['contentId'])].set_index('personId')
        user_profiles = {}
        for person_id in interactions_indexed_df.index.unique():
            user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
        return user_profiles

    user_profiles = build_users_profiles()

    # print(len(user_profiles))

    class ContentBasedRecommender:
        MODEL_NAME = 'Content-Based'

        def __init__(self, items_df=None):
            self.item_ids = item_ids
            self.items_df = items_df

        def get_model_name(self):
            return self.MODEL_NAME

        def _get_similar_items_to_user_profile(self, person_id, topn=1000):
            # Computes the cosine similarity between the user profile and all item profiles
            cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
            # Gets the top similar items
            similar_indices = cosine_similarities.argsort().flatten()[-topn:]
            # Sort the similar items by similarity
            similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                                   key=lambda x: -x[1])
            return similar_items

        def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
            similar_items = self._get_similar_items_to_user_profile(user_id)
            # Ignores items the user has already interacted
            similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

            recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                .head(topn)

            if verbose:
                if self.items_df is None:
                    raise Exception('"items_df" is required in verbose mode')

                recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                              left_on='contentId',
                                                              right_on='contentId')[
                    ['recStrength', 'contentId', 'title', 'cuisine']]

            return recommendations_df

    content_based_recommender_model = ContentBasedRecommender(articles_df)

    print('Evaluating Content-Based Filtering model...')
    cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)

    print('\nGlobal metrics:\n%s' % cb_global_metrics)
    cb_detailed_results_df.head(10)

    ####################################################################### Collaborative Filtering model

    # Creating a sparse pivot table with users in rows and items in columns
    users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId',
                                                              columns='contentId',
                                                              values='eventStrength').fillna(0)

    print(users_items_pivot_matrix_df.head(10))

    users_items_pivot_matrix = users_items_pivot_matrix_df.values
    users_items_pivot_matrix[:10]

    users_ids = list(users_items_pivot_matrix_df.index)
    users_ids[:10]

    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
    users_items_pivot_sparse_matrix

    # The number of factors to factor the user-item matrix.
    NUMBER_OF_FACTORS_MF = 15
    # Performs matrix factorization of the original user item matrix
    # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
    U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

    print(U.shape)

    print(Vt.shape)

    sigma = np.diag(sigma)
    print(sigma.shape)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    print(all_user_predicted_ratings)

    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
            all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

    # Converting the reconstructed matrix back to a Pandas dataframe
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                               index=users_ids).transpose()
    cf_preds_df.head(10)

    len(cf_preds_df.columns)

    class CFRecommender:
        MODEL_NAME = 'Collaborative Filtering'

        def __init__(self, cf_predictions_df, items_df=None):
            self.cf_predictions_df = cf_predictions_df
            self.items_df = items_df

        def get_model_name(self):
            return self.MODEL_NAME

        def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
            # Get and sort the user's predictions
            sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                .reset_index().rename(columns={user_id: 'recStrength'})

            # Recommend the highest predicted rating movies that the user hasn't seen yet.
            recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                .sort_values('recStrength', ascending=False) \
                .head(topn)

            if verbose:
                if self.items_df is None:
                    raise Exception('"items_df" is required in verbose mode')

                recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                              left_on='contentId',
                                                              right_on='contentId')[
                    ['recStrength', 'contentId', 'title', 'cuisine']]

            return recommendations_df

    cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

    print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
    cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
    print('\nGlobal metrics:\n%s' % cf_global_metrics)
    cf_detailed_results_df.head(10)

    ####################################################################### Hybrid model

    class HybridRecommender:
        MODEL_NAME = 'Hybrid'

        def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
            self.cb_rec_model = cb_rec_model
            self.cf_rec_model = cf_rec_model
            self.cb_ensemble_weight = cb_ensemble_weight
            self.cf_ensemble_weight = cf_ensemble_weight
            self.items_df = items_df

        def get_model_name(self):
            return self.MODEL_NAME

        def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
            # Getting the top-1000 Content-based filtering recommendations
            cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})

            # Getting the top-1000 Collaborative filtering recommendations
            cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})

            # Combining the results by contentId
            recs_df = cb_recs_df.merge(cf_recs_df,
                                       how='outer',
                                       left_on='contentId',
                                       right_on='contentId').fillna(0.0)

            # Computing a hybrid recommendation score based on CF and CB scores
            # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
            recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) \
                                           + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)

            # Sorting recommendations by hybrid score
            recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

            if verbose:
                if self.items_df is None:
                    raise Exception('"items_df" is required in verbose mode')

                recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                              left_on='contentId',
                                                              right_on='contentId')[
                    ['recStrengthHybrid', 'contentId', 'title', 'cuisine']]

            return recommendations_df

    hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df,
                                                 cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)

    print('Evaluating Hybrid model...')
    hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
    print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
    print(hybrid_detailed_results_df.head(10))

    global_metrics_df = pd.DataFrame([cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics]) \
        .set_index('modelName')
    print(global_metrics_df)

    ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15, 8))
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')

    def inspect_interactions(person_id, test_set=True):
        if test_set:
            interactions_df = interactions_test_indexed_df
        else:
            interactions_df = interactions_train_indexed_df
        return interactions_df.loc[person_id].merge(articles_df, how='left',
                                                    left_on='contentId',
                                                    right_on='contentId') \
            .sort_values('eventStrength', ascending=False)[['eventStrength',

                                                            'contentId',
                                                            'title', 'veg', 'cuisine']]

    # inspect_interactions(4, test_set=False).head(20)

    if my_profile in user_profiles:
        print("my profile is in user_profiles")
        myprofile = user_profiles[my_profile]
        print(myprofile.shape)
        pd.DataFrame(sorted(zip(tfidf_feature_names,
                                myprofile.flatten().tolist()), key=lambda x: -x[1])[:20],
                     columns=['token', 'relevance'])

        print("\n\nHybrid")
        print("Hybrid")
        print("Hybrid")
        Hybrid = hybrid_recommender_model.recommend_items(my_profile, topn=50, verbose=True)
        Hybrid = Hybrid.drop_duplicates(subset='contentId', keep="first")
        print(Hybrid)

        # print(cf_recommender_model.recommend_items(my_profile, topn=20, verbose=True))
        # remove duplicates
        recommadation = cf_recommender_model.recommend_items(my_profile, topn=50, verbose=True)
        recommadation = recommadation.drop_duplicates(subset='contentId', keep="first")
        print("\n\nCF")
        print("CF")
        print("CF")
        print("CF")
        print("CF")
        print(recommadation)

        print("\n\nCB")
        print("CB")
        print("CB")
        CB = content_based_recommender_model.recommend_items(my_profile, topn=50, verbose=True)
        CB = CB.drop_duplicates(subset='contentId', keep="first")
        print(CB)

        return recommadation.head(num_of_rec)['contentId'].tolist(), 1

    else:
        print("my profile is not in user_profiles")

        # return top 10 popular items
        item_popularity_df = interactions_train_df.groupby('contentId')['eventStrength'].sum().sort_values(
            ascending=False).reset_index()
        print(item_popularity_df.head(10))

        x = item_popularity_df.head(num_of_rec)['contentId'].tolist()

        return x, 0
