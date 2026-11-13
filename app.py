"""Wine style finder (Gradio).

Move the 13 lab-measurement sliders → the app returns the wine's style (with probabilities), how it differs from the
style centre, and the 5 most similar wines. The model is rebuilt at start-up exactly as in the notebook:
Yeo-Johnson transform → k-means (k = 3) for the styles, Gaussian mixture for the probabilities.
The data ships with scikit-learn (`load_wine`, the same 178 wines as the Kaggle CSV), so nothing is downloaded.

    pip install -r requirements.txt
    python app.py

Optional: WINE_DEMO=<row> fills the sliders with that wine and runs it on page load (used for the README screenshot).
"""
import os

import gradio as gr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler

SEED, K = 42, 3
FEATURES = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
            'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
wine = pd.DataFrame(load_wine().data, columns=FEATURES)

# ---------------------------------------------------------------- model (same steps as the notebook)
transformer = PowerTransformer(method='yeo-johnson').fit(wine)
X = transformer.transform(wine)
groups = KMeans(K, n_init=50, random_state=SEED).fit_predict(X)
scaler = StandardScaler().fit(wine)
z = pd.DataFrame(scaler.transform(wine), columns=FEATURES)
centres_z = z.groupby(groups).mean()


def style_name(row):
    top = row.abs().sort_values(ascending=False).index[:2]
    return ' · '.join(f'{"high" if row[c] > 0 else "low"} {c.lower().replace("_", " ")}' for c in top)


STYLE = {g: f'Style {chr(65 + i)}: {style_name(centres_z.loc[g])}' for i, g in enumerate(centres_z.index)}
gmm = GaussianMixture(K, covariance_type='diag', n_init=10, random_state=SEED).fit(X)
gmm_to_style = {g: pd.Series(groups[gmm.predict(X) == g]).mode().iloc[0] for g in range(K)}
nn = NearestNeighbors(n_neighbors=5).fit(X)


# ---------------------------------------------------------------- app logic
def find_style(*values):
    row = pd.DataFrame([values], columns=FEATURES)
    x = transformer.transform(row)
    proba = pd.Series(gmm.predict_proba(x)[0], index=[STYLE[gmm_to_style[g]] for g in range(K)]).groupby(level=0).sum()
    best = proba.idxmax()
    g = next(k for k, v in STYLE.items() if v == best)
    wz = pd.Series(scaler.transform(row)[0], index=FEATURES)
    diff = (wz - centres_z.loc[g]).sort_values(key=abs, ascending=False)
    unusual = gmm.score_samples(x)[0] < np.quantile(gmm.score_samples(X), 0.05)
    text = (f'### {best}\n'
            f'Probability {proba.max():.0%}. Most different from its style centre: '
            + ', '.join(f'{c.lower().replace("_", " ")} ({"higher" if d > 0 else "lower"})' for c, d in diff.iloc[:3].items())
            + ('\n\n(!) Unusual wine: less likely than 95 % of the known wines → worth a lab re-check.' if unusual else ''))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = centres_z.loc[g].index
    ax.barh(order, centres_z.loc[g], color='lightgrey', label=f'{best.split(":")[0]} centre')
    ax.scatter(wz[order], order, color='black', zorder=3, label='this wine')
    ax.axvline(0, color='grey', linewidth=0.8)
    ax.set(xlabel='standard deviations from the average wine', title='This wine vs its style')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    dist, idx = nn.kneighbors(x)
    similar = wine.iloc[idx[0]][['Alcohol', 'Flavanoids', 'Color_Intensity', 'Hue', 'Proline']].assign(
        style=[STYLE[groups[i]].split(':')[0] for i in idx[0]], distance=dist[0].round(2)).reset_index(names='wine')
    return proba.sort_values(ascending=False).to_dict(), text, fig, similar.round(2)


sliders = [gr.Slider(float(wine[c].min()), float(wine[c].max()), value=float(wine[c].median()), label=c.replace('_', ' '))
           for c in FEATURES]
with gr.Blocks(title='Wine style finder') as demo:
    gr.Markdown('# Wine Style Finder\nSet the 13 lab measurements of a wine → its style, how it differs, and the most similar wines.')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    for s in sliders[:7]:
                        s.render()
                with gr.Column():
                    for s in sliders[7:]:
                        s.render()
            btn = gr.Button('Find style', variant='primary')
        with gr.Column(scale=3):
            label = gr.Label(num_top_classes=3, label='Style')
            text = gr.Markdown()
            plot = gr.Plot(show_label=False)
            table = gr.Dataframe(label='5 most similar wines')
    btn.click(find_style, inputs=sliders, outputs=[label, text, plot, table])
    if os.environ.get('WINE_DEMO'):
        demo_values = [float(v) for v in wine.iloc[int(os.environ['WINE_DEMO'])]]
        cached = (*demo_values, *find_style(*demo_values))
        demo.load(lambda: cached, outputs=[*sliders, label, text, plot, table])

if __name__ == '__main__':
    demo.launch()
