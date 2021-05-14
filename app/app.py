import json

from flask import Flask, render_template
from plotly import graph_objects as go
import plotly
import matplotlib.pyplot as plt
import umap

from lastfm.tracker import track_user, visualize_user_tracking


app = Flask(__name__)


def visualize_user_tracking(uid, user_embeddings):
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
