"""Store and load detection results in an SQLite database."""

import pandas as pd
from sqlalchemy import create_engine

RESULTS_TABLE_NAME = "results"


def save_results(results, database_path):
    """Save the results table to an SQLite database file.

    An existing table with the same name is replaced.
    """
    engine = create_engine(f"sqlite:///{database_path}")
    results.to_sql(
        RESULTS_TABLE_NAME, engine, if_exists="replace", index=False
    )
    engine.dispose()


def load_results(database_path):
    """Load the results table from an SQLite database file."""
    engine = create_engine(f"sqlite:///{database_path}")
    results = pd.read_sql_table(RESULTS_TABLE_NAME, engine)
    engine.dispose()
    return results
