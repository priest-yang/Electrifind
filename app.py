from flask import Flask, redirect, render_template, request, url_for

import pandas as pd
import sys
sys.path.append("src")
from pipeline import SearchEngine

app = Flask(__name__)

engine = SearchEngine(cf=False, l2r=False)

params = {
    "user-id": 1, 
}

default_prompt = "fast"
default_lng = "-73.985"
default_lat = "40.758"


def get_results_all(lat = default_lat, lng = default_lng, prompt = default_prompt, top_n = 10, userid = None):
    query = str(lat) + ", " + str(lng) + ", " + str(prompt)
    results = engine.search(query, userid = userid)
    results = results[:top_n]
    return results

@app.route("/")
def home():
    result = get_results_all()
    return render_template("index.html", result=result)


@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        lat = request.form.get("lat")
        lng = request.form.get("lng")
        query = request.form.get("query")
        query = request.form.get("query")
        # if not query:
        #     query = "A mountain in spring"
        style = request.form.get("style")
        scene = request.form.get("scene")
        medium = request.form.get("medium")
        light = request.form.get("light")
        quality = request.form.get("quality")
        print(query)
        print(style)
        print(scene)
        print(medium)
        print(light)
        print(quality)
        args = [style, scene, medium, light, quality]
        prompts, urls = get_results_all(engine, query, 200, args)
        result = list(zip(prompts, urls))
        return render_template(
            "search.html",
            result=result,
            query=query,
            style=style,
            scene=scene,
            medium=medium,
            light=light,
            quality=quality,
        )
    return redirect(url_for("home"))


if __name__ == "__main__":
    # engine = initialize_all()
    app.run(debug=True)
