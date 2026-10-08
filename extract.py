"""
extract.py
----------
Extraction layer for the CompetitionClimbing ETL pipeline.
Connects to Google Sheets via gspread and returns raw DataFrames.
"""

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

SHEET_NAME = "BD-CompetitionClimbing"
CREDENTIALS_FILE = "CC_google_credentials.json"


def get_gspread_client() -> gspread.Client:
    """
    Authenticate with Google Sheets API using the service account
    credentials file (CC_google_credentials.json) and return an
    authorised gspread client.

    Returns:
        gspread.Client: An authenticated client ready to open spreadsheets.
    """
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)
    print("[extract] ✅ Google Sheets client authenticated.")
    return client


def _worksheet_to_df(client: gspread.Client, tab_name: str) -> pd.DataFrame:
    """
    Private helper: open the spreadsheet, fetch a named worksheet, and
    return all its data as a pandas DataFrame.

    Uses get_all_records() which:
      - Uses the first row as column headers.
      - Returns empty cells as empty strings (we normalise later in transform).

    Args:
        client (gspread.Client): Authenticated gspread client.
        tab_name (str): Exact name of the worksheet tab to fetch.

    Returns:
        pd.DataFrame: Raw data from the specified worksheet.

    Raises:
        gspread.exceptions.SpreadsheetNotFound: If SHEET_NAME does not exist
            or the service account has not been granted access.
        gspread.exceptions.WorksheetNotFound: If tab_name does not exist
            inside the spreadsheet.
    """
    spreadsheet = client.open(SHEET_NAME)
    worksheet = spreadsheet.worksheet(tab_name)
    records = worksheet.get_all_records(empty2zero=False, head=1)
    df = pd.DataFrame(records)
    print(f"[extract] 📋 Tab '{tab_name}' — {len(df)} rows × {len(df.columns)} cols")
    return df


def extract_atletas(client: gspread.Client) -> pd.DataFrame:
    """
    Open the 'atletas' worksheet from the Google Sheet and return its
    contents as a raw pandas DataFrame (no transformation applied here).

    Columns expected from the sheet:
        id_atleta, nombre_completo, nombre, apellido_1, apellido_2,
        nacionalidad, fecha_nacimiento, edad, ciudad, primera_competicion

    Args:
        client (gspread.Client): An authenticated gspread client.

    Returns:
        pd.DataFrame: Raw data from the 'atletas' tab.
    """
    df = _worksheet_to_df(client, tab_name="atletas")
    print(f"[extract] 🧍 Athletes columns: {df.columns.tolist()}")
    return df


def extract_cb_results(client: gspread.Client) -> pd.DataFrame:
    """
    Open the 'CB_22A25' worksheet from the Google Sheet and return its
    contents as a raw pandas DataFrame (no transformation applied here).

    Columns expected from the sheet:
        id_atleta, Nombre, Evento, Modalidad, Categoria_evento, Año, Sede,
        Categoria, Etapa,
        Ruta1_Top, Ruta1_IntentoTop, Ruta1_Zona, Ruta1_IntentoZona,
        Ruta2_Top, Ruta2_IntentoTop, Ruta2_Zona, Ruta2_IntentoZona,
        Ruta3_Top, Ruta3_IntentoTop, Ruta3_Zona, Ruta3_IntentoZona,
        Ruta4_Top, Ruta4_IntentoTop, Ruta4_Zona, Ruta4_IntentoZona,
        Ruta5_Top, Ruta5_IntentoTop, Ruta5_Zona, Ruta5_IntentoZona,
        Top, Intentos_Tops, Zona, Intentos_Zona, Posicion,
        P_R1, P_R2, P_R3, P_R4, P_R5, Puntaje_total

    Args:
        client (gspread.Client): An authenticated gspread client.

    Returns:
        pd.DataFrame: Raw data from the 'CB_22A25' tab.
    """
    df = _worksheet_to_df(client, tab_name="CB_22A25")
    print(f"[extract] 🏆 Results columns: {df.columns.tolist()}")
    return df
