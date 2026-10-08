"""
main.py
-------
ETL Orchestrator for the CompetitionClimbing pipeline.

Run this file to execute the full ETL process:
    python main.py

Load order (FK constraints require this exact sequence):
    1. paises          (no dependencies)
    2. athletes        (no dependencies)
    3. competitions    (no dependencies)
    4. results         ← needs athletes.id + competitions.id
    5. route_results   ← needs results.id
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from extract import get_gspread_client, extract_atletas, extract_cb_results
from transform import (
    build_id_map,
    build_result_id_map,
    transform_athletes,
    transform_paises,
    transform_competitions,
    transform_results,
    transform_route_results,
)
from load import (
    get_supabase_client,
    load_paises,
    load_athletes,
    load_competitions,
    load_results,
    load_route_results,
)


def run_etl() -> None:
    """
    Orchestrate the full ETL pipeline in the correct dependency order.

    Steps:
        1. Initialise clients (Google Sheets + Supabase).
        2. Extract raw data from both Google Sheet tabs.
        3. Transform & load paises (country lookup table).
        4. Transform & load athletes → build in-memory athlete_id_map.
        5. Transform & load competitions → build in-memory competition_id_map.
        6. Transform & load results (uses athlete + comp maps) → build in-memory result_id_map.
        7. Transform & load route_results (uses result_id_map).
        8. Print completion summary.
    """
    print("=" * 60)
    print("🚀 Starting CompetitionClimbing ETL Pipeline")
    print("=" * 60)

    # 1. Initialize API Clients
    gspread_client = get_gspread_client()
    supabase_client = get_supabase_client()

    # 2. Extract Data
    print("\n--- [1/5] EXTRACT PHASE ---")
    df_atletas_raw = extract_atletas(gspread_client)
    df_cb_raw = extract_cb_results(gspread_client)

    # 3. Reference Data (Athletes & Competitions)
    print("\n--- [2/5] REFERENCE DATA (Athletes, Competitions) ---")
    # Clean up any dummy numerical codes (1..7) from earlier testing
    try:
        supabase_client.table("paises").delete().in_("codigo", [str(i) for i in range(1, 8)]).execute()
    except Exception:
        pass

    df_athletes = transform_athletes(df_atletas_raw)
    df_athletes_inserted = load_athletes(supabase_client, df_athletes)
    athlete_id_map = build_id_map(df_athletes_inserted, key_col="nombre_completo", id_col="id")
    # Also map raw sheet's nombre_completo as a safeguard against any formatting variations
    for raw_name, ath_id in zip(df_atletas_raw["nombre_completo"].astype(str).str.strip(), df_athletes_inserted["id"]):
        athlete_id_map[raw_name] = ath_id
    print(f"[main] 🗺️  Mapped {len(athlete_id_map)} athletes to IDs in memory.")

    df_competitions = transform_competitions(df_cb_raw)
    df_competitions_inserted = load_competitions(supabase_client, df_competitions)
    competition_id_map = build_id_map(df_competitions_inserted, key_col="nombre", id_col="id")
    print(f"[main] 🗺️  Mapped {len(competition_id_map)} competitions to IDs in memory.")

    # 4. Transform & Load Results
    print("\n--- [3/5] RESULTS TABLE ---")
    df_results = transform_results(df_cb_raw, athlete_id_map, competition_id_map)
    df_results_inserted = load_results(supabase_client, df_results)
    result_id_map = build_result_id_map(df_results_inserted, athlete_id_map, competition_id_map)
    print(f"[main] 🗺️  Mapped {len(result_id_map)} results to IDs in memory.")

    # 5. Transform & Load Route Results
    print("\n--- [4/5] ROUTE RESULTS TABLE ---")
    df_route_results = transform_route_results(df_cb_raw, result_id_map)
    load_route_results(supabase_client, df_route_results)

    print("\n" + "=" * 60)
    print("🎉 ETL Pipeline Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_etl()
