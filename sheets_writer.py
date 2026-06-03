"""
sheets_writer.py
----------------
Handles both reading from and writing to Google Sheets for the IoT pipeline.

Usage
-----
    client = SheetsClient("credentials.json", "YOUR_SHEET_ID")

    # Write
    client.write("01_RawSensorData", raw_df)
    client.write_many({"Tab1": df1, "Tab2": df2})

    # Read
    raw_df  = client.read("01_RawSensorData")
    all_dfs = client.read_all(["01_RawSensorData", "02_ProcessedFeatures", ...])
"""

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe


SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_ROWS = 3000
DEFAULT_COLS = 40


class SheetsClient:
    """
    Unified read/write client for the IoT pipeline Google Sheets integration.

    Parameters
    ----------
    creds_file : str   Path to the service-account JSON key (credentials.json).
    sheet_id   : str   Spreadsheet ID from the Google Sheets URL.
    """

    def __init__(self, creds_file: str, sheet_id: str):
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, SCOPE)
        client = gspread.authorize(creds)
        self.spreadsheet = client.open_by_key(sheet_id)

    # ──────────────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────────────

    def write(self, tab_name: str, df: pd.DataFrame,
              rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> None:
        """Write df to tab_name, creating the tab if needed and clearing first."""
        ws = self._get_or_create(tab_name, rows, cols)
        ws.clear()
        set_with_dataframe(ws, df)
        print(f"  ✔  '{tab_name}' → {len(df):,} rows × {len(df.columns)} cols written")

    def write_many(self, mapping: dict) -> None:
        """Write multiple DataFrames: {tab_name: df, ...}"""
        for tab_name, df in mapping.items():
            self.write(tab_name, df)

    # ──────────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────────

    def read(self, tab_name: str) -> pd.DataFrame:
        """
        Read a worksheet tab into a DataFrame.
        Drops completely empty rows/columns that gspread sometimes pads.

        Raises
        ------
        gspread.exceptions.WorksheetNotFound  if the tab doesn't exist yet.
        ValueError                            if the tab exists but is empty.
        """
        ws = self.spreadsheet.worksheet(tab_name)
        df = get_as_dataframe(ws, evaluate_formulas=True, na_filter=True)

        # Drop padding rows/columns that gspread adds
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"Tab '{tab_name}' exists but contains no data. "
                "Run main_pipeline.py first to populate it."
            )

        print(f"  ✔  '{tab_name}' → {len(df):,} rows × {len(df.columns)} cols read")
        return df

    def read_all(self, tab_names: list[str]) -> dict:
        """
        Read multiple tabs at once.

        Returns
        -------
        dict  {tab_name: DataFrame}  — only tabs that were read successfully.
        Missing or empty tabs raise, so callers should handle errors per tab.
        """
        return {name: self.read(name) for name in tab_names}

    def tab_exists(self, tab_name: str) -> bool:
        """Return True if the worksheet tab is present in the spreadsheet."""
        try:
            self.spreadsheet.worksheet(tab_name)
            return True
        except gspread.exceptions.WorksheetNotFound:
            return False

    def available_tabs(self) -> list[str]:
        """Return a list of all existing worksheet tab titles."""
        return [ws.title for ws in self.spreadsheet.worksheets()]

    # ──────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────

    def _get_or_create(self, title: str, rows: int, cols: int) -> gspread.Worksheet:
        try:
            return self.spreadsheet.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            return self.spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)


# ── backward-compat alias so main_pipeline.py import still works ──────────────
SheetsWriter = SheetsClient
