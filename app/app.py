import json

from flask import Flask, render_template
import plotly

from lastfm.tracker import track_user, visualize_user_tracking


app = Flask(__name__)


@app.route("/")
def main():
    return "Lastfm Data Analysis"


@app.route("/user/<uid>")
def hello_world(uid):
    e = track_user(uid, "baseline")
    fig = visualize_user_tracking(uid, e)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("plotly.html", graphJSON=graphJSON)
