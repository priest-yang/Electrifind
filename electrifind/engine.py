from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from flask_googlemaps import Map, icons
from werkzeug.exceptions import abort
import csv
from tqdm import tqdm

from src.pipeline import SearchEngine

DATA_PATH = "./data/"
NREL_PATH = DATA_PATH + "NREL_raw.csv"
DEFAULT_LAT = 42.30136771768067
DEFAULT_LNG = -83.71907280246434
DEFAULT_USER = 0
DEFAULT_PROMPT = None
RADIUS_DICT = {'small': 0.01, 'med': 0.03, 'large': 0.05}

bp = Blueprint('engine', __name__)
engine = SearchEngine()


@bp.route('/')
def index():
    gmap = Map(
        identifier="gmap",
        varname="gmap",
        lat=DEFAULT_LAT,
        lng=DEFAULT_LNG,
        style="height:50vmax;width:80vmax;margin:50px;",
    )
    return render_template("engine/index.html", gmap=gmap)


@bp.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        lat = request.form['lat']
        lng = request.form['lng']
        prompt = request.form['prompt'] if 'prompt' in request.form else DEFAULT_PROMPT
        user_id = request.form['user_id'] if 'user_id' in request.form else DEFAULT_USER
        sort_by = request.form['sort']
        radius = request.form['radius']
        error = None

        if not lat or not lng:
            error = 'Latitude and Longitude are required.'

        if sort_by == 'distance':
            engine.set_reranker()
        elif sort_by == 'base':
            engine.set_reranker('vector')
        elif sort_by == 'cf':
            if user_id == DEFAULT_USER or user_id == None: 
                error = 'User ID is required for collaborative filtering.'
            engine.set_reranker('cf')
        else:
            error = 'Invalid sort_by parameter.'

        if radius in RADIUS_DICT:
            radius = RADIUS_DICT[radius]
        else:
            error = 'Invalid radius parameter.'

        if error is not None:
            flash(error)
        else:
            result = engine.get_results_all(lat, lng, prompt, user_id, radius)
            if result:
                print(result)
                result_df = engine.get_station_info(result)
                table_html = result_df.to_html(
                    classes="table table-striped", index=False, justify="left")
                marker_t = {
                    "icon": icons.dots.blue,
                    "lat": None,
                    "lng": None,
                    "infobox": None
                }
                markers = []
                for i in range(len(result_df)):
                    marker_t["lat"] = result_df.iloc[i]['latitude']
                    marker_t["lng"] = result_df.iloc[i]['longitude']
                    marker_t["infobox"] = f"{result_df.iloc[i]['station_name']}<br>{result_df.iloc[i]['street_address']}"
                    markers.append(marker_t.copy())
                gmap = Map(
                    identifier="gmap",
                    varname="gmap",
                    lat=float(lat),
                    lng=float(lng),
                    markers=markers,
                    style="height:50vmax;width:80vmax;margin:50px;",
                )
                return render_template(
                    'engine/index.html',
                    result=result,
                    lat=lat,
                    lng=lng,
                    prompt=prompt,
                    user_id=user_id,
                    table_html=table_html,
                    gmap=gmap
                )

    return redirect(url_for('engine.index'))
