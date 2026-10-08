"""
load.py
-------
Load layer for the CompetitionClimbing ETL pipeline.
Inserts transformed DataFrames into the Supabase PostgreSQL database.

⚠️  LOAD ORDER IS CRITICAL — foreign key constraints require:
    1. paises          (no dependencies)
    2. athletes        (no dependencies)
    3. competitions    (no dependencies)
    4. results         (depends on athletes.id + competitions.id)
    5. route_results   (depends on results.id)

Upsert strategy:
    All inserts use on_conflict="ignore" so that re-running the pipeline
    on already-loaded data is safe. Supabase will skip duplicate rows
    rather than raising an error or creating duplicates.
    ⚠️  This requires a UNIQUE constraint on each table's natural key:
        paises          → unique(nombre)
        athletes        → unique(nombre_completo)   ← NOT added per design decision;
                          instead we rely on the source sheet being pre-cleaned.
        competitions    → unique(nombre)
        results         → unique(athlete_id, competition_id, etapa)
        route_results   → unique(result_id, ruta_num)

Batch size:
    Large DataFrames are split into chunks of BATCH_SIZE rows before
    inserting. Supabase's REST API has a practical limit of ~1 000–2 000
    rows per request; batching avoids payload-size errors.
"""

import os
import math
import datetime
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Rows per Supabase REST API insert request.
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_supabase_client() -> Client:
    """
    Initialise and return a Supabase client using the SUPABASE_URL and
    SUPABASE_KEY environment variables loaded from the .env file.

    Returns:
        Client: An authenticated Supabase client.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY are missing from .env.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "Missing Supabase credentials. "
            "Ensure SUPABASE_URL and SUPABASE_KEY are set in your .env file."
        )

    client = create_client(url, key)
    print("[load] ✅ Supabase client initialised.")
    return client


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Convert a DataFrame to a list of dicts safe for the Supabase JSON payload.

    pandas .to_dict('records') does NOT convert float NaN to None — it leaves
    them as float('nan'), which is not valid JSON and causes Supabase insert
    errors. This function fixes that with an explicit per-cell check using
    math.isnan(), which is the only 100% reliable approach.

    Args:
        df (pd.DataFrame): Any clean transform-layer DataFrame.

    Returns:
        list[dict]: Row records with all NaN replaced by None, ready for
                    supabase .insert().
    """
    raw = df.to_dict(orient="records")
    clean = []
    for row in raw:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                clean_row[k] = None
            elif isinstance(v, (datetime.date, datetime.datetime)):
                clean_row[k] = v.isoformat()
            else:
                clean_row[k] = v
        clean.append(clean_row)
    return clean



def _insert_batched(
    client: Client,
    table: str,
    records: list[dict],
    label: str,
    on_conflict: str | None = None,
) -> list[dict]:
    """
    Insert or upsert records into a Supabase table in batches of BATCH_SIZE rows,
    returning all inserted rows (with server-generated ids) as a flat list.

    Args:
        client (Client):        Authenticated Supabase client.
        table (str):            Supabase table name (e.g. "athletes").
        records (list[dict]):   Rows to insert, already serialized to dicts.
        label (str):            Human-readable label used in log output.
        on_conflict (str|None): Column name(s) for ON CONFLICT DO UPDATE (upsert).

    Returns:
        list[dict]: All rows as returned by Supabase (includes generated ids).

    Raises:
        Exception: Re-raises any Supabase API error with contextual info.
    """
    all_returned = []
    total = len(records)
    batches = range(0, total, BATCH_SIZE)

    for i, start in enumerate(batches, 1):
        batch = records[start : start + BATCH_SIZE]
        try:
            if on_conflict:
                response = client.table(table).upsert(batch, on_conflict=on_conflict).execute()
            else:
                response = client.table(table).insert(batch).execute()
            all_returned.extend(response.data)
            action = "upserted" if on_conflict else "inserted"
            print(
                f"[load]   ✔ {label} batch {i}/{len(batches)} "
                f"({len(batch)} rows) → {action}"
            )
        except Exception as exc:
            action = "upserting" if on_conflict else "inserting"
            raise RuntimeError(
                f"[load] ❌ Failed {action} batch {i} into '{table}': {exc}"
            ) from exc

    return all_returned


# ---------------------------------------------------------------------------
# Public load functions
# ---------------------------------------------------------------------------

def load_paises(client: Client, df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert the paises DataFrame into the 'paises' Supabase table and return
    the inserted rows (including any server-generated fields).

    Seeded first — no FK dependencies.

    Args:
        client (Client): Authenticated Supabase client.
        df (pd.DataFrame): Clean paises DataFrame from transform_paises().

    Returns:
        pd.DataFrame: Inserted rows as returned by Supabase.
    """
    records = _df_to_records(df)
    print(f"[load] 🌎 Upserting {len(records)} paises rows...")
    try:
        response = client.table("paises").upsert(records).execute()
        df_out = pd.DataFrame(response.data)
    except Exception:
        response = client.table("paises").select("*").execute()
        df_out = pd.DataFrame(response.data)
    print(f"[load] ✅ paises done — {len(df_out)} rows in DB.")
    return df_out



def load_athletes(client: Client, df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert the athletes DataFrame into the 'athletes' Supabase table
    and return the inserted rows (including the Supabase-generated 'id').

    The returned DataFrame is passed to build_id_map() in main.py to create:
        athlete_id_map = { nombre_completo: supabase_uuid }

    Seeded second — no FK dependencies.

    Args:
        client (Client): Authenticated Supabase client.
        df (pd.DataFrame): Clean athletes DataFrame from transform_athletes().

    Returns:
        pd.DataFrame: Inserted rows including server-generated 'id' column.
    """
    records = _df_to_records(df)
    print(f"[load] 🧍 Inserting {len(records)} athlete rows...")
    inserted = _insert_batched(client, table="athletes", records=records, label="athletes")
    df_out = pd.DataFrame(inserted)
    print(f"[load] ✅ athletes done — {len(df_out)} rows in DB.")
    return df_out


def load_competitions(client: Client, df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert the competitions DataFrame into the 'competitions' Supabase table
    and return the inserted rows (including the Supabase-generated 'id').

    The returned DataFrame is passed to build_id_map() in main.py to create:
        competition_id_map = { competition_nombre: supabase_uuid }

    Seeded third — no FK dependencies.

    Args:
        client (Client): Authenticated Supabase client.
        df (pd.DataFrame): Clean competitions DataFrame from transform_competitions().

    Returns:
        pd.DataFrame: Inserted rows including server-generated 'id' column.
    """
    records = _df_to_records(df)
    print(f"[load] 🏆 Upserting {len(records)} competition rows...")
    inserted = _insert_batched(
        client,
        table="competitions",
        records=records,
        label="competitions",
        on_conflict="nombre",
    )
    df_out = pd.DataFrame(inserted)
    print(f"[load] ✅ competitions done — {len(df_out)} rows in DB.")
    return df_out


def load_results(client: Client, df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert the results DataFrame into the 'results' Supabase table and return
    the inserted rows (including the Supabase-generated 'id').

    The returned DataFrame is used in main.py to build result_id_map:
        athlete_id_map and competition_id_map are reversed to recover
        (nombre_completo, comp_nombre, etapa) → result_id.

    Seeded fourth — depends on athletes.id + competitions.id.

    Args:
        client (Client): Authenticated Supabase client.
        df (pd.DataFrame): Clean results DataFrame from transform_results().

    Returns:
        pd.DataFrame: Inserted rows including server-generated 'id' column.
    """
    records = _df_to_records(df)
    print(f"[load] 📊 Inserting {len(records)} result rows...")
    inserted = _insert_batched(client, table="results", records=records, label="results")
    df_out = pd.DataFrame(inserted)
    print(f"[load] ✅ results done — {len(df_out)} rows in DB.")
    return df_out


def load_route_results(client: Client, df: pd.DataFrame) -> None:
    """
    Insert the route_results DataFrame into the 'route_results' Supabase table.

    This is the largest load step (~324 source rows × up to 5 routes = up to
    1 620 rows). No return value is needed — route_results has no downstream
    FK dependencies.

    Seeded last — depends on results.id.

    Args:
        client (Client): Authenticated Supabase client.
        df (pd.DataFrame): Clean route_results DataFrame from transform_route_results().

    Returns:
        None
    """
    records = _df_to_records(df)
    print(f"[load] 🧗 Inserting {len(records)} route_result rows...")
    _insert_batched(client, table="route_results", records=records, label="route_results")
    print(f"[load] ✅ route_results done — {len(records)} rows in DB.")
