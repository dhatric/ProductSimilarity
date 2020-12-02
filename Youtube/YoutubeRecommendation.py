import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Read csv
ds = pd.read_csv("youtube-small.csv")

#Create TF-IDF Vectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['tags'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['video_id'][i]) for i in similar_indices]

    results[row['video_id']] = similar_items[1:]

print('done!')


def item(id):
    return ds.loc[ds['video_id'] == id]['tags'].tolist()[0].split(' - ')[0]


# Just reads the results out of the dictionary.
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item_id + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + rec[1] + " (score:" + str(rec[0]) + ")")


recommend(item_id='kzwfHumJyYc', num=5)