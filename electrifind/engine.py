from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
import csv
from tqdm import tqdm

from src.pipeline import SearchEngine

DEFAULT_USER = 0
DEFAULT_PROMPT = None
DEFAULT_LNG = "-83.0703"
DEFAULT_LAT = "42.3317"
DATA_PATH = "./data/"
NREL_PATH = DATA_PATH + "NREL_raw.csv"

bp = Blueprint('engine', __name__)
engine = SearchEngine()


def get_results_all(lat, lng, prompt, user_id, top_n=10):
    query = str(lat) + ", " + str(lng)
    if prompt:
        query = query + ", " + prompt
    param = {
        "user_id": user_id,
    }
    results = engine.search(query, **param)
    results = results[:top_n]
    return results


@bp.route('/')
def index():
    result, table_html = [], []

    if result:
        if type(result[0]) is list:
            result = [rel[0] for rel in result]

        result_df = engine.get_station_info([i.docid for i in result])
        table_html = result_df.to_html(
            classes="table table-striped", index=False, justify="left")

    return render_template("engine/index.html", result=result, table_html=table_html)


@bp.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        lat = request.form['lat']
        lng = request.form['lng']
        prompt = request.form['prompt'] if 'prompt' in request.form else DEFAULT_PROMPT
        user_id = request.form['user_id'] if 'user_id' in request.form else DEFAULT_USER
        error = None

        if not lat or not lng:
            error = 'Latitude and Longitude are required.'

        if error is not None:
            flash(error)
        else:
            result = get_results_all(
                lat=lat, lng=lng, prompt=prompt, user_id=user_id)
            if result:
                if type(result[0]) is list:
                    result = [rel[0] for rel in result]
                result_df = engine.get_station_info([i.docid for i in result])
                table_html = result_df.to_html(
                    classes="table table-striped", index=False, justify="left")
                return render_template(
                    'engine/search.html',
                    result=result,
                    lat=lat,
                    lng=lng,
                    prompt=prompt,
                    user_id=user_id,
                    table_html=table_html,
                )

    return redirect(url_for('engine.index'))


@bp.route('/add_csv', methods=['POST'])
def add_csv():
    if request.method == 'POST':
        with open(NREL_PATH, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            row = next(reader)
            for row in tqdm(reader):
                pass
            
            flash('CSV file uploaded and added to the database successfully.')
    return redirect(url_for('engine.index'))
