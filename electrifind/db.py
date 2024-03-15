import sqlite3
import csv

import click
from flask import current_app, g

from src.utils import DATA_PATH

NREL_PATH = DATA_PATH + "NREL_raw.csv"


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

    def mod_item(item):
        if item == '':
            return 'NULL'
        item = item.replace('"', '\'')
        return f'"{item}"'

    with open(NREL_PATH, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        columns = next(reader)
        for row in reader:
            row = [mod_item(item) for item in row]
            db.execute(f'INSERT INTO nrel (' + ', '.join(columns) +
                       ') VALUES (' + ', '.join(row) + ')')


@click.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
