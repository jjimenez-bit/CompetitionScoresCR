"""
transform.py
------------
Transformation layer for the CompetitionClimbing ETL pipeline.
Takes raw DataFrames from the extract layer and returns clean,
normalized DataFrames ready to be loaded into Supabase.

ID resolution strategy — "Select-then-Insert memory maps":
  After each load_*() call, the returned rows (with Supabase-generated UUIDs)
  are converted to a plain dict via build_id_map(). Downstream transforms
  use these dicts for FK lookups instead of re-querying the database.

  athlete_id_map    : { nombre_completo  → athletes.id }
  competition_id_map: { comp_nombre      → competitions.id }
  result_id_map     : { (nombre_completo, comp_nombre, etapa) → results.id }
  (result_id_map is built in main.py by reverse-mapping the first two dicts)
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maps ruta number (1–5) → the 4 source column names for that route.
_ROUTE_COLS: dict[int, dict[str, str]] = {
    n: {
        "top":         f"Ruta{n}_Top",
        "intento_top": f"Ruta{n}_IntentoTop",
        "zona":        f"Ruta{n}_Zona",
        "intento_zona":f"Ruta{n}_IntentoZona",
    }
    for n in range(1, 6)
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from all object (string) columns
    and replace empty strings with None (SQL NULL).

    Called at the start of every transform function. NaN values that remain
    after this step are handled at serialization time in load._df_to_records(),
    which performs the authoritative float-NaN → None conversion before the
    Supabase JSON payload is built.

    Args:
        df (pd.DataFrame): Any DataFrame.

    Returns:
        pd.DataFrame: Same DataFrame with trimmed strings and no empty strings.
    """
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    df = df.replace("", None)
    return df


def _build_comp_nombre(df: pd.DataFrame) -> pd.Series:
    """
    Build the competition 'nombre' key from CB_22A25 columns.

    Formula: "Campeonato " + Categoria_evento + " " + Evento + " " + Año
    Example: "Campeonato Mayor Interno Boulder 1 2022"

    Assumes the DataFrame has already been cleaned (no leading/trailing
    whitespace, no empty strings). Año is cast to str to handle integer dtype.

    Args:
        df (pd.DataFrame): DataFrame containing Categoria_evento, Evento, Año.

    Returns:
        pd.Series: One competition nombre string per row.
    """
    return (
        "Campeonato "
        + df["Categoria_evento"].astype(str)
        + " "
        + df["Evento"].astype(str)
        + " "
        + df["Año"].astype(str)
    )


def _is_route_empty(val_top, val_zona, val_intento_top, val_intento_zona) -> bool:
    """
    Return True if ALL four route column values indicate the route was
    not attempted (i.e., the stage had fewer than 5 routes and these
    columns are just structural placeholders in the sheet).

    A value is considered 'empty' if it is: None, NaN, "", or "0" / 0.

    Args:
        val_top, val_zona (str | None): "TOP"/"ZONA" or "0"/None.
        val_intento_top, val_intento_zona (int | str | None): numeric or "".

    Returns:
        bool: True if the route should be skipped.
    """
    def _empty(v) -> bool:
        if v is None:
            return True
        try:
            if pd.isna(v):
                return True
        except (TypeError, ValueError):
            pass
        return str(v).strip() in ("", "0")

    return all(_empty(v) for v in [val_top, val_zona, val_intento_top, val_intento_zona])


def _to_bool(val) -> bool:
    """
    Normalise a TOP/ZONA string flag to a Python bool.

    "TOP" or "ZONA"  → True
    "0", "", None    → False

    Args:
        val: Raw cell value from the Google Sheet.

    Returns:
        bool
    """
    if isinstance(val, str):
        return val.strip().upper() in ("TOP", "ZONA")
    return False


def _to_int(val) -> int:
    """
    Safely coerce a raw cell value to int, defaulting to 0 on any failure.
    Handles empty strings, None, NaN, and float representations.

    Args:
        val: Raw cell value from the Google Sheet.

    Returns:
        int: Parsed integer, or 0 if parsing fails.
    """
    try:
        if pd.isna(val) or str(val).strip() == "":
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_id_map(
    df_inserted: pd.DataFrame,
    key_col: str,
    id_col: str = "id",
) -> dict:
    """
    Convert a DataFrame of Supabase-inserted rows into a plain Python
    dictionary used for FK lookups throughout the transform layer.

    This is the core of the "Select-then-Insert memory mapping" strategy:
    after each .insert() call we capture the returned rows (which contain
    Supabase-generated ids) and build a dict so that downstream transform
    functions never need to query the database again.

    Example output for athletes:
        { "García López Juan": "uuid-aaa", "Martínez Ruiz Ana": "uuid-bbb" }

    Args:
        df_inserted (pd.DataFrame): Rows returned by Supabase after insert,
                                    must contain both key_col and id_col.
        key_col (str): Column whose values become the dict keys
                       (e.g. 'nombre_completo' for athletes,
                             'nombre' for competitions).
        id_col (str): Column whose values become the dict values.
                      Defaults to 'id' (Supabase auto-generated UUID).

    Returns:
        dict: { key_col_value: id_col_value } mapping for FK resolution.
    """
    return dict(zip(df_inserted[key_col], df_inserted[id_col]))


def build_result_id_map(
    df_results_inserted: pd.DataFrame,
    athlete_id_map: dict,
    competition_id_map: dict,
) -> dict:
    """
    Construct the composite lookup map:
        (nombre_completo, competition_nombre, etapa) → result_id (UUID)

    By inverting athlete_id_map and competition_id_map, we can map the
    FK UUIDs in df_results_inserted back to the natural business keys
    present in the raw CB_22A25 sheet without making any database queries.

    Args:
        df_results_inserted (pd.DataFrame): Inserted rows returned from Supabase
                                            with columns: id, athlete_id, competition_id, etapa.
        athlete_id_map (dict): { nombre_completo: athlete_id }
        competition_id_map (dict): { comp_nombre: competition_id }

    Returns:
        dict: { (nombre_completo, comp_nombre, etapa): result_id }
    """
    id_to_athlete = {v: k for k, v in athlete_id_map.items()}
    id_to_comp = {v: k for k, v in competition_id_map.items()}

    result_map = {}
    for _, row in df_results_inserted.iterrows():
        nombre = id_to_athlete.get(row["athlete_id"])
        comp_nombre = id_to_comp.get(row["competition_id"])
        etapa = row["etapa"]
        if nombre and comp_nombre and etapa:
            result_map[(nombre, comp_nombre, etapa)] = row["id"]

    return result_map


def transform_athletes(df_atletas: pd.DataFrame) -> pd.DataFrame:
    """
    Map the raw 'atletas' DataFrame to the 'athletes' Supabase table shape.

    Rules:
    - Drop 'id_atleta', 'edad', 'ciudad', 'primera_competicion'
      (not in destination schema).
    - Rename: apellido_1 → primer_apellido, apellido_2 → segundo_apellido.
    - Add 'gender' column as None (populated later via athlete self-service form).
    - Parse 'fecha_nacimiento' to ISO date; coerce unparseable values to None.
    - Strip whitespace and replace empty strings with None across all columns.

    Args:
        df_atletas (pd.DataFrame): Raw DataFrame from extract_atletas().

    Returns:
        pd.DataFrame: Clean athletes DataFrame matching the destination schema.
    """
    df = df_atletas.rename(columns={
        "apellido_1": "primer_apellido",
        "apellido_2": "segundo_apellido",
    }).copy()

    # Select only insertable destination columns (drops id_atleta, edad, ciudad, primera_competicion).
    # NOTE: 'nombre_completo' is omitted because in Supabase it is a GENERATED ALWAYS column.
    df = df[["nombre", "primer_apellido", "segundo_apellido",
             "fecha_nacimiento", "nacionalidad"]]

    df = _clean_strings(df)

    # 'primer_apellido' is NOT NULL in Supabase schema.
    # Single-name athletes (e.g. 'Courtland') have no surname, so default None to ""
    df["primer_apellido"] = df["primer_apellido"].fillna("")

    # Map country names to ISO codes expected by paises.codigo foreign key
    country_to_iso = {
        "Costa Rica": "CR",
        "Chile": "CL",
        "El Salvador": "SV",
        "España": "ES",
        "Estados Unidos": "US",
        "Honduras": "HN",
        "Reino Unido": "UK",
    }
    df["nacionalidad"] = df["nacionalidad"].map(lambda x: country_to_iso.get(x, x) if x else None)

    # Add gender placeholder
    df["gender"] = None

    # Parse dates — dayfirst=False because the sheet uses MM/DD/YYYY (US format).
    # errors='coerce' silently converts unparseable/empty values to NaT → None.
    # Convert to ISO 8601 string ("YYYY-MM-DD") which is JSON serializable for Postgres DATE columns.
    parsed_dates = pd.to_datetime(
        df["fecha_nacimiento"], dayfirst=False, errors="coerce"
    )
    df["fecha_nacimiento"] = [
        d.strftime("%Y-%m-%d") if pd.notna(d) else None for d in parsed_dates
    ]



    print(f"[transform] 🧍 Athletes ready: {len(df)} rows")
    return df.reset_index(drop=True)


def transform_paises(df_atletas: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique country values from 'atletas[nacionalidad]' and build
    the 'paises' reference table.

    NOTE: The source data contains full country names (e.g. "Costa Rica"),
    not ISO codes. A sequential integer is used for 'codigo' since no
    ISO mapping is available in the source. Update manually if ISO codes
    are required in the future.

    Args:
        df_atletas (pd.DataFrame): Raw DataFrame from extract_atletas().

    Returns:
        pd.DataFrame: Clean paises DataFrame with columns [codigo, nombre].
    """
    nombres = (
        df_atletas["nacionalidad"]
        .dropna()
        .str.strip()
        .replace("", None)
        .dropna()
        .unique()
    )

    df = pd.DataFrame({"nombre": sorted(set(nombres))})
    df.insert(0, "codigo", range(1, len(df) + 1))

    print(f"[transform] 🌎 Paises ready: {len(df)} unique countries")
    return df.reset_index(drop=True)


def transform_competitions(df_cb: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the unique competitions from the raw CB_22A25 DataFrame.

    Rules:
    - Build the 'nombre' key as:
        "Campeonato " + Categoria_evento + " " + Evento + " " + str(Año)
      Example: "Campeonato Mayor Interno Boulder 1 2022"
    - Deduplicate on (Categoria_evento, Evento, Año) — each unique
      combination becomes one competition row.
    - Map remaining columns:
        Modalidad        → modalidad
        Sede             → sede
        Categoria_evento → categoria
        Año              → año
    - Strip whitespace and replace empty strings with None.

    Args:
        df_cb (pd.DataFrame): Raw DataFrame from extract_cb_results().

    Returns:
        pd.DataFrame: Clean competitions DataFrame matching the destination schema.
    """
    df = df_cb[["Categoria_evento", "Evento", "Año", "Modalidad", "Sede"]].copy()
    df = _clean_strings(df)

    # Drop rows missing any key column (defensive — should not happen on clean data)
    df = df.dropna(subset=["Categoria_evento", "Evento", "Año"])

    df = df.drop_duplicates(
        subset=["Categoria_evento", "Evento", "Año"]
    ).reset_index(drop=True)

    df["nombre"] = _build_comp_nombre(df)

    df = df.rename(columns={
        "Modalidad":        "modalidad",
        "Sede":             "sede",
        "Categoria_evento": "categoria",
        "Año":              "año",
    })

    df = df[["nombre", "modalidad", "sede", "categoria", "año"]]
    print(f"[transform] 🏆 Competitions ready: {len(df)} unique competitions")
    return df


def transform_results(
    df_cb: pd.DataFrame,
    athlete_id_map: dict,
    competition_id_map: dict,
) -> pd.DataFrame:
    """
    Build the 'results' rows by linking athletes and competitions via
    pre-built in-memory id maps (Select-then-Insert strategy).

    Rules:
    - Deduplicate CB_22A25 on (Nombre, Categoria_evento, Evento, Año, Categoria, Etapa).
    - Resolve athlete_id  : athlete_id_map[Nombre]          → athletes.id
    - Resolve competition_id: competition_id_map[comp_nombre] → competitions.id
      where comp_nombre = "Campeonato " + Categoria_evento + " " + Evento + " " + Año
    - Output columns: athlete_id, competition_id, categoria, etapa.
    - Warn (but do not crash) if any key is missing from either map —
      print the unmatched value and skip that row.

    Args:
        df_cb (pd.DataFrame): Raw DataFrame from extract_cb_results().
        athlete_id_map (dict): { nombre_completo: supabase_uuid }
                               built from build_id_map() after load_athletes().
        competition_id_map (dict): { competition_nombre: supabase_uuid }
                                   built from build_id_map() after load_competitions().

    Returns:
        pd.DataFrame: Clean results DataFrame matching the destination schema.
    """
    df = df_cb[["Nombre", "Categoria_evento", "Evento", "Año", "Categoria", "Etapa"]].copy()
    df = _clean_strings(df)
    df = df.drop_duplicates(
        subset=["Nombre", "Categoria_evento", "Evento", "Año", "Categoria", "Etapa"]
    ).reset_index(drop=True)

    # Build competition nombre key for lookup
    df["comp_nombre"] = _build_comp_nombre(df)

    # Resolve FK ids via in-memory maps
    df["athlete_id"]     = df["Nombre"].map(athlete_id_map)
    df["competition_id"] = df["comp_nombre"].map(competition_id_map)

    # Warn on any unmatched keys (should be 0 on clean data)
    missing_athletes = df[df["athlete_id"].isna()]["Nombre"].unique()
    missing_comps    = df[df["competition_id"].isna()]["comp_nombre"].unique()

    if len(missing_athletes) > 0:
        print(f"[transform] ⚠️  Unmatched athletes  ({len(missing_athletes)}): {missing_athletes.tolist()}")
    if len(missing_comps) > 0:
        print(f"[transform] ⚠️  Unmatched competitions ({len(missing_comps)}): {missing_comps.tolist()}")

    # Drop unresolved rows so we never insert a NULL FK
    df = df[df["athlete_id"].notna() & df["competition_id"].notna()].copy()

    df = df.rename(columns={"Categoria": "categoria", "Etapa": "etapa"})
    df = df[["athlete_id", "competition_id", "categoria", "etapa"]]

    print(f"[transform] 📊 Results ready: {len(df)} rows")
    return df.reset_index(drop=True)


def transform_route_results(
    df_cb: pd.DataFrame,
    result_id_map: dict,
) -> pd.DataFrame:
    """
    Unpivot (melt) the wide route columns in CB_22A25 into one row per
    (result_id, ruta_num), matching the 'route_results' table schema.

    Source columns (up to 5 routes × 4 columns = 20 columns):
        RutaN_Top, RutaN_IntentoTop, RutaN_Zona, RutaN_IntentoZona  (N = 1..5)

    Target columns per row:
        result_id, ruta_num (1–5), top (bool), intento_top (int),
        zona (bool), intento_zona (int)

    Rules:
    - Iterate routes 1–5; for each route check whether ALL four columns are
      null/zero — if so, skip that route (stage had fewer than 5 routes).
    - Normalise 'top' and 'zona' columns:
        "TOP" / "ZONA" → True, "0" / NaN / empty → False.
    - Resolve result_id via result_id_map using a composite key:
        (Nombre, competition_nombre, Etapa) → results.id
      where competition_nombre = "Campeonato " + Categoria_evento + " " + Evento + " " + Año

    NOTE: result_id_map is built in main.py by reverse-mapping athlete_id_map
    and competition_id_map against the rows returned by load_results(). See
    main.py for the exact construction pattern.

    Args:
        df_cb (pd.DataFrame): Raw DataFrame from extract_cb_results().
        result_id_map (dict): { (nombre_completo, competition_nombre, etapa): supabase_uuid }
                              built in main.py after load_results().

    Returns:
        pd.DataFrame: Clean route_results DataFrame matching the destination schema.
    """
    df = df_cb.copy()
    df = _clean_strings(df)
    df["_comp_nombre"] = _build_comp_nombre(df)

    rows = []
    unmatched_keys: set = set()

    for _, row in df.iterrows():
        lookup_key = (row["Nombre"], row["_comp_nombre"], row["Etapa"])
        result_id = result_id_map.get(lookup_key)

        if result_id is None:
            unmatched_keys.add(lookup_key)
            continue

        for ruta_num in range(1, 6):
            cols = _ROUTE_COLS[ruta_num]
            val_top         = row.get(cols["top"],         "")
            val_intento_top = row.get(cols["intento_top"], 0)
            val_zona        = row.get(cols["zona"],        "")
            val_intento_zona= row.get(cols["intento_zona"],0)

            # Skip routes that were not part of this stage
            if _is_route_empty(val_top, val_zona, val_intento_top, val_intento_zona):
                continue

            rows.append({
                "result_id":    result_id,
                "ruta_num":     ruta_num,
                "top":          _to_bool(val_top),
                "intento_top":  _to_int(val_intento_top),
                "zona":         _to_bool(val_zona),
                "intento_zona": _to_int(val_intento_zona),
            })

    if unmatched_keys:
        print(f"[transform] ⚠️  Unmatched result keys ({len(unmatched_keys)} rows skipped):")
        for k in list(unmatched_keys)[:5]:  # show first 5 only
            print(f"            → {k}")

    df_out = pd.DataFrame(rows)
    print(f"[transform] 🧗 Route results ready: {len(df_out)} rows "
          f"(from {len(df)} source rows × up to 5 routes each)")
    return df_out
