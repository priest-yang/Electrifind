from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
import csv
from tqdm import tqdm

from src.pipeline import SearchEngine

DATA_PATH = "./data/"
NREL_PATH = DATA_PATH + "NREL_raw.csv"
DEFAULT_LAT = "42.30136771768067"
DEFAULT_LNG = "-83.71907280246434"
DEFAULT_USER = 0
DEFAULT_PROMPT = None

bp = Blueprint('engine', __name__)
engine = SearchEngine()


@bp.route('/')
def index():
    return render_template("engine/index.html")


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
            result = engine.get_results_all(lat, lng, prompt, user_id)
            if result:
                print(result)
                result_df = engine.get_station_info(result)
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
