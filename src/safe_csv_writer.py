"""
safe_csv_writer.py
==================
Safely write/overwrite CSV files using atomic operations to avoid corruption
and file locking issues in production environments (especially with Power BI).

Key principle: Write to a temporary file first, then atomically replace the
target file. This ensures Power BI never reads a partial/corrupt file.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def enforce_powerbi_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    *,
    keep_extra_columns: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame with a stable, Power BI-friendly schema.

    - Ensures all required columns exist (missing columns are created as empty strings)
    - Preserves the exact required column order first
    - Optionally appends remaining columns in deterministic alphabetical order
    """
    out = df.copy()

    for col in required_columns:
        if col not in out.columns:
            out[col] = ""

    ordered = list(required_columns)
    if keep_extra_columns:
        extras = sorted(c for c in out.columns if c not in required_columns)
        ordered.extend(extras)

    return out[ordered]


def add_rupee_formatted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add formatted currency columns in Indian rupee format for Power BI display.

    Looks for numeric columns like 'expected_revenue_loss', 'MonthlyCharges', 'TotalCharges'
    and creates formatted versions for display.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    pd.DataFrame
        Copy with added formatted columns
    """
    from src.indian_currency import format_indian_currency

    result_df = df.copy()

    currency_cols_to_format = [
        ("expected_revenue_loss", "expected_revenue_loss_rupees"),
        ("MonthlyCharges", "monthly_charges_rupees"),
        ("TotalCharges", "total_charges_rupees"),
    ]

    for source_col, target_col in currency_cols_to_format:
        if source_col in result_df.columns:
            try:
                result_df[target_col] = result_df[source_col].apply(
                    lambda x: format_indian_currency(float(x)) if pd.notna(x) else ""
                )
            except Exception as exc:
                logger.warning(
                    "Could not format %s to %s: %s", source_col, target_col, exc
                )

    return result_df


def safe_write_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    columns_order: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """
    Safely write a DataFrame to CSV with atomic file replacement.

    This function:
    1. Writes to a temporary file in the same directory as the target
    2. Atomically replaces the target file (prevents partial writes)
    3. Returns the final path on success
    4. Raises meaningful errors if something fails

    Parameters
    ----------
    df : pd.DataFrame
        The data to write
    output_path : str or Path
        Final destination for the CSV file
    columns_order : list[str], optional
        If provided, ensures columns appear in this exact order.
        Missing columns are skipped; extra columns in df trigger a warning.
    verbose : bool, default True
        Print progress messages

    Returns
    -------
    Path
        The final path where the file was written

    Raises
    ------
    ValueError
        If columns_order is provided and no valid columns exist in df
    OSError
        If the temporary file cannot be created or replaced atomically
    """
    output_path = Path(output_path)

    # Validate and reorder columns if requested
    working_df = df.copy()
    if columns_order is not None:
        # Find which columns from columns_order actually exist in the DataFrame
        valid_cols = [c for c in columns_order if c in working_df.columns]
        if not valid_cols:
            raise ValueError(
                f"No columns from columns_order exist in DataFrame.\n"
                f"Expected one of: {columns_order}\n"
                f"Got: {working_df.columns.tolist()}"
            )
        # Reorder to match specification
        missing_in_spec = [c for c in working_df.columns if c not in columns_order]
        if missing_in_spec:
            logger.warning(
                "Columns not in columns_order (will be skipped): %s", missing_in_spec
            )
        working_df = working_df[valid_cols]

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file in the same directory (atomic replacement on the same volume)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_path.parent,
            suffix=".csv",
            delete=False,
            newline="",
            encoding="utf-8-sig",  # UTF-8 with BOM for Excel compatibility
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            working_df.to_csv(tmp_file, index=False)

        if verbose:
            logger.info(
                "Temporary CSV written (%d rows × %d cols) → %s",
                len(working_df),
                len(working_df.columns),
                tmp_path.name,
            )
    except Exception as exc:
        logger.error("Failed to write temporary CSV: %s", exc)
        raise

    # Atomically replace the target file
    try:
        # On Windows, we need to remove the old file first if it exists
        if output_path.exists():
            output_path.unlink()
        # Rename (atomic on same volume)
        tmp_path.replace(output_path)

        if verbose:
            logger.info(
                "CSV file safely replaced → %s (%d rows)", output_path, len(working_df)
            )

        return output_path

    except Exception as exc:
        # Clean up the temp file if replacement fails
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        logger.error("Failed to replace CSV file: %s", exc)
        raise


def safe_write_csv_with_fallback(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    backup_path: str | Path | None = None,
    columns_order: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """
    Safely write CSV with automatic backup on success.

    Useful for keeping a backup of the previous version in case of issues.

    Parameters
    ----------
    df : pd.DataFrame
        The data to write
    output_path : str or Path
        Final destination for the CSV file
    backup_path : str or Path, optional
        Where to save the previous version (if it exists).
        If None, no backup is created.
    columns_order : list[str], optional
        Column order specification (see safe_write_csv)
    verbose : bool, default True
        Print progress messages

    Returns
    -------
    Path
        The final path where the file was written
    """
    output_path = Path(output_path)
    backup_path = Path(backup_path) if backup_path else None

    # Save backup if the current file exists and a backup path was provided
    if backup_path and output_path.exists():
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(output_path, backup_path)
            if verbose:
                logger.info("Backup created → %s", backup_path)
        except Exception as exc:
            logger.warning("Could not create backup: %s", exc)

    # Write the new file
    return safe_write_csv(df, output_path, columns_order=columns_order, verbose=verbose)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    import numpy as np

    test_df = pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(5)],
        "tenure": np.random.randint(1, 72, 5),
        "churn_probability": np.random.uniform(0, 1, 5),
    })

    test_path = Path("test_output.csv")
    safe_write_csv(test_df, test_path, verbose=True)
    print(f"\nTest file written to: {test_path.resolve()}")

    # Verify the read-back
    loaded = pd.read_csv(test_path)
    print(f"Verification: {len(loaded)} rows re-loaded successfully")

    # Clean up
    test_path.unlink()
    print("Test file cleaned up")
