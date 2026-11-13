"""Wine taste-alike finder (Gradio).

Describe a wine you like in your own words → its taste style and the most similar wines from 120k Wine Enthusiast
reviews, optionally under a price cap and from other grapes. The model is rebuilt exactly as in the notebook:
blind the notes (grape / country words removed) → TF-IDF → LSA (100) → k-means (8 styles).

Data (not stored here): WINE_REVIEWS_CSV, else ./data/winemag-data-130k-v2.csv
(Kaggle dataset zynicide/wine-reviews). The fitted index is cached in ./cache after the first start (about 2 minutes).

    pip install -r requirements.txt
    python app.py

Optional: WINE_DEMO=1 runs the example note on page load (used for the README screenshot).
"""
import os
import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
SEED, K, MAX_FEATURES, N_COMPONENTS = 42, 8, 20000, 100
CSV = Path(os.environ.get('WINE_REVIEWS_CSV', HERE / 'data' / 'winemag-data-130k-v2.csv'))
CACHE = HERE / 'cache' / 'wine_index.joblib'
GENERIC = {'wine', 'wines', 'drink', 'drinks', 'flavors', 'flavor', 'aromas', 'aroma', 'finish', 'palate', 'nose', 'notes',
           'note', 'offers', 'shows', 'bottle', 'bottling', 'vintage', 'years', 'year', 'now', 'just', 'like', 'made', 'make',
           'feels', 'gives', 'hint', 'hints', 'touch', 'bit', 'quite', 'good', 'nice', 'well', 'mouth', 'mouthfeel', 'character'}
TOKEN = re.compile(r'[a-z]+')
EXAMPLE = 'Crisp and zesty with lime, green apple and a flinty, mineral finish.'


def name_tokens(series):
    words = set()
    for name in series.dropna().unique():
        words |= {w for w in re.findall(r'[a-zà-ÿ]+', name.lower()) if len(w) > 2}
    return words


def build():
    if not CSV.exists():
        raise FileNotFoundError(f'(!) {CSV} not found: download Kaggle dataset zynicide/wine-reviews into ./data '
                                'or set WINE_REVIEWS_CSV')
    raw = pd.read_csv(CSV, index_col=0)
    wine = raw[~raw.duplicated(['title', 'description'])].dropna(subset=['description', 'variety']).reset_index(drop=True)
    stop = set(ENGLISH_STOP_WORDS) | GENERIC | name_tokens(wine['variety']) | name_tokens(wine['country']) | name_tokens(wine['province'])
    clean = lambda t: ' '.join(w for w in TOKEN.findall(t.lower()) if w not in stop and len(w) > 2)
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    T = tfidf.fit_transform(wine['description'].map(clean))
    svd = TruncatedSVD(N_COMPONENTS, random_state=SEED)
    Z = normalize(svd.fit_transform(T))
    kmeans = KMeans(K, n_init=10, random_state=SEED).fit(Z)
    terms = np.array(tfidf.get_feature_names_out())
    centre_terms = svd.inverse_transform(kmeans.cluster_centers_)
    style = {g: 'S{}: {}'.format(g + 1, ', '.join([t for t in terms[np.argsort(centre_terms[g])[::-1]] if ' ' not in t][:3])) for g in range(K)}
    table = wine[['title', 'variety', 'country', 'points', 'price']]
    return {'stop': stop, 'tfidf': tfidf, 'svd': svd, 'kmeans': kmeans, 'Z': Z.astype(np.float32), 'style': style,
            'labels': kmeans.labels_, 'table': table}


if CACHE.exists():
    IDX = joblib.load(CACHE)
else:
    IDX = build()
    CACHE.parent.mkdir(exist_ok=True)
    joblib.dump(IDX, CACHE)
NN = NearestNeighbors(n_neighbors=200, metric='cosine').fit(IDX['Z'])


def find(note, max_price, other_grapes_only):
    words = ' '.join(w for w in TOKEN.findall(note.lower()) if w not in IDX['stop'] and len(w) > 2)
    if not words:
        return 'Describe the taste (fruit, acidity, oak, body …).', pd.DataFrame()
    z = normalize(IDX['svd'].transform(IDX['tfidf'].transform([words])))
    g = int(IDX['kmeans'].predict(z)[0])
    dist, idx = NN.kneighbors(z)
    hits = IDX['table'].iloc[idx[0]].assign(similarity=(1 - dist[0]).round(3), style=[IDX['style'][l].split(':')[0] for l in IDX['labels'][idx[0]]])
    hits = hits[hits['price'].fillna(1e9) <= max_price]
    if other_grapes_only and len(hits):
        hits = hits[hits['variety'] != hits['variety'].iloc[0]]
    share = (IDX['labels'] == g).mean()
    text = f'### {IDX["style"][g]}\nThis note reads like **{share:.0%}** of the 120k reviewed wines. Words used: *{words}*'
    hits = hits.assign(wine=hits['title'].str.slice(0, 48))[['points', 'price', 'similarity', 'style', 'variety', 'country', 'wine']]
    return text, hits.head(8).reset_index(drop=True)


with gr.Blocks(title='Wine taste-alike finder') as demo:
    gr.Markdown('# Wine Taste-alike Finder\nDescribe a wine you like → its taste style and similar wines, from 120k Wine Enthusiast reviews.')
    with gr.Row():
        with gr.Column(scale=2):
            note = gr.Textbox(lines=4, value=EXAMPLE, label='Describe a wine you like')
            price = gr.Slider(5, 200, value=30, step=1, label='Maximum price ($)')
            other = gr.Checkbox(label='Show other grapes only (discover something new)')
            btn = gr.Button('Find wines', variant='primary')
        with gr.Column(scale=3):
            style = gr.Markdown()
            table = gr.Dataframe(label='Wines that taste alike')
    btn.click(find, inputs=[note, price, other], outputs=[style, table])
    if os.environ.get('WINE_DEMO') == '1':
        cached = find(EXAMPLE, 30, False)
        demo.load(lambda: cached, outputs=[style, table])

if __name__ == '__main__':
    demo.launch()
