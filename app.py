from flask import Flask, redirect, render_template, request, url_for

import pandas as pd
import sys
sys.path.append("src")
from pipeline import SearchEngine

app = Flask(__name__)

engine = SearchEngine(reranker="l2r")

DEFAULT_USER = 1
DEFAULT_PROMPT = "fast"
DEFAULT_LNG = "-83.0703"
DEFAULT_LAT = "42.3317"


def get_results_all(lat = DEFAULT_LAT, lng = DEFAULT_LNG, prompt = DEFAULT_PROMPT, top_n = 10, user_id = DEFAULT_USER):
    query = str(lat) + ", " + str(lng) + ", " + str(prompt)
    # + str(prompt)
    param = {
        "user_id": user_id,
    }
    print(query)
    results = engine.search(query, **param)
    results = results[:top_n]
    return results

@app.route("/")
def home():
    result = get_results_all(user_id=1)

    if type(result[0]) is list:
        result = [rel[0] for rel in result]
    
    result_df = engine.get_station_info([i.docid for i in result])
    table_html = result_df.to_html(classes="table table-striped", index=False, justify="left")
    
    return render_template("index.html", result=result, table_html=table_html)


@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        lat = request.form.get("lat")
        lng = request.form.get("lng")
        prompt = request.form.get("prompt")
        user_id = request.form.get("user_id")

        print(lat)
        print(lng)
        print(prompt)
        print(user_id)

        result = get_results_all(lat=lat, lng=lng, prompt=prompt, user_id=int(user_id))
        if type(result[0]) is list:
            result = [rel[0] for rel in result]
        result_df = engine.get_station_info([i.docid for i in result])
        table_html = result_df.to_html(classes="table table-striped", index=False, justify="left")

        return render_template(
            "search.html",
            result=result,
            lat=lat,
            lng=lng,
            prompt=prompt,
            user_id=user_id, 
            table_html=table_html, 
        )
    return redirect(url_for("home"))

if __name__ == "__main__":
    # engine = initialize_all()
    app.run(debug=True)
