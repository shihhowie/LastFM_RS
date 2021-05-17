import json

from flask import Flask, render_template
from plotly import graph_objects as go
import numpy as np
import plotly
import matplotlib.pyplot as plt
import umap

from lastfm.tracker import track_user


app = Flask(__name__)


def visualize_user_tracking_umap(uid, user_embeddings):
    reducer = umap.UMAP()
    active_ind = np.where(np.all(user_embeddings, axis=1))[0]
    embeddings_active = user_embeddings[active_ind]
    reduced_embedding = reducer.fit_transform(embeddings_active)
    # fig, _ = plt.subplots(figsize=(12, 8))
    # plt.plot(reduced_embedding[:, 0], reduced_embedding[:, 1], "o-", alpha=0.8)
    # for i in range(len(active_ind)):
    #     plt.annotate(active_ind[i], (reduced_embedding[i, 0], reduced_embedding[i, 1]))
    # plt.title(f"user tracking for uid {uid}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reduced_embedding[:, 0],
            y=reduced_embedding[:, 1],
            text=[str(x) for x in active_ind],
            mode="lines+markers+text",
            textposition="top center",
        )
    )
    fig.update_layout(
        title=f"User {uid} tracking with UMAP",
        xaxis_title="Vector 1",
        yaxis_title="Vector 2",
    )

    return fig


def visualize_user_tracking(uid, user_embeddings):
    active_ind = np.where(np.all(user_embeddings, axis=1))[0]
    embeddings_active = user_embeddings[active_ind]
    embedding_diff = []
    embedding_diff_to_first = []
    base = embeddings_active[0]
    for e1, e2 in zip(embeddings_active[:-1], embeddings_active[1:]):
        diff = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        embedding_diff.append(diff)
        diff_to_first = np.dot(base, e2) / (np.linalg.norm(base) * np.linalg.norm(e2))
        embedding_diff_to_first.append(diff_to_first)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=active_ind,
            y=embedding_diff,
            mode="lines+markers",
            name="cosine similarity to previous window",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=active_ind,
            y=embedding_diff_to_first,
            mode="lines+markers",
            name="cosine similarity to first window",
        )
    )

    fig.update_layout(
        title=f"User {uid} tracking",
        xaxis_title="Window",
        yaxis_title="Cosine Similarity",
    )
    return fig


@app.route("/")
def main():
    return "Lastfm Data Analysis"


@app.route("/user/<uid>")
def hello_world(uid):
    e = track_user(uid, "baseline")
    fig = visualize_user_tracking(uid, e)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("plotly.html", graphJSON=graphJSON)


@app.route("/user/<uid>/umap")
def hello_umap(uid):
    e = track_user(uid, "baseline")
    fig = visualize_user_tracking_umap(uid, e)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("plotly.html", graphJSON=graphJSON)
