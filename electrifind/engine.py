from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from src.pipeline import SearchEngine

bp = Blueprint('engine', __name__)
engine = SearchEngine()

DEFAULT_USER = 1
DEFAULT_PROMPT = "fast"
DEFAULT_LNG = "-83.0703"
DEFAULT_LAT = "42.3317"


def get_results_all(lat=DEFAULT_LAT, lng=DEFAULT_LNG, prompt=DEFAULT_PROMPT, top_n=10, user_id=DEFAULT_USER):
    query = str(lat) + ", " + str(lng) + ", " + str(prompt)
    # + str(prompt)
    param = {
        "user_id": user_id,
    }
    print(query)
    results = engine.search(query, **param)
    results = results[:top_n]
    return results


@bp.route('/')
def index():
    result, table_html = [], []
    # result = get_results_all(user_id=1)
    if result:
        if type(result[0]) is list:
            result = [rel[0] for rel in result]

        result_df = engine.get_station_info([i.docid for i in result])
        table_html = result_df.to_html(
            classes="table table-striped", index=False, justify="left")

    return render_template("engine/index.html", result=result, table_html=table_html)


@bp.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        lat = request.form.get("lat")
        lng = request.form.get("lng")
        prompt = request.form.get("prompt")
        user_id = request.form.get("user_id")
        error = None

        if error is not None:
            flash(error)
        else:
            result = get_results_all(
                lat=lat, lng=lng, prompt=prompt, user_id=int(user_id))
            if type(result[0]) is list:
                result = [rel[0] for rel in result]
            result_df = engine.get_station_info([i.docid for i in result])
            table_html = result_df.to_html(
                classes="table table-striped", index=False, justify="left")

        return render_template(
            "search.html",
            result=result,
            lat=lat,
            lng=lng,
            prompt=prompt,
            user_id=user_id,
            table_html=table_html,
        )
    return redirect(url_for('engine.index'))
