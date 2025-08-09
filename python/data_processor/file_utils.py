import os
from typing import Optional

import numpy as np
import pandas as pd

try:
    import scipy.io

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SCIPY_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PYARROW_AVAILABLE = False


class FileFormatDetector:
    """Detect file format based on extension and content."""

    @staticmethod
    def detect_format(file_path: str) -> Optional[str]:
        """Detect file format from extension and content."""
        if not os.path.exists(file_path):
            return None

        # Check extension first
        ext = os.path.splitext(file_path)[1].lower()
        extension_map = {
            ".csv": "csv",
            ".tsv": "tsv",
            ".parquet": "parquet",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".pkl": "pickle",
            ".npy": "numpy",
            ".mat": "matlab",
            ".feather": "feather",
            ".arrow": "arrow",
            ".db": "sqlite",
        }

        if ext in extension_map:
            return extension_map[ext]

        # Try to detect from content
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)

            if header.startswith(b"PAR1"):
                return "parquet"
            if header.startswith(b"ARROW1"):
                return "arrow"
            if header.startswith(b"PK\x03\x04"):
                return "excel"
            if header.startswith(b"\x89PNG"):
                return "png"
            if header.startswith(b"SQLite format 3"):
                return "sqlite"
        except Exception:
            pass

        return None


class DataReader:
    """Handles reading data from various file formats."""

    @staticmethod
    def read_file(file_path: str, format_type: str, **kwargs) -> pd.DataFrame:
        """Read file based on format type."""
        try:
            if format_type == "csv":
                return pd.read_csv(file_path, **kwargs)
            if format_type == "tsv":
                return pd.read_csv(file_path, sep="\t", **kwargs)
            if format_type == "parquet":
                if PYARROW_AVAILABLE:
                    return pq.read_table(file_path).to_pandas()
                return pd.read_parquet(file_path, **kwargs)
            if format_type == "excel":
                return pd.read_excel(file_path, engine="openpyxl", **kwargs)
            if format_type == "json":
                return pd.read_json(file_path, **kwargs)
            if format_type == "hdf5":
                return pd.read_hdf(file_path, **kwargs)
            if format_type == "pickle":
                return pd.read_pickle(file_path)
            if format_type == "numpy":
                return pd.DataFrame(np.load(file_path))
            if format_type == "matlab" and SCIPY_AVAILABLE:
                return pd.DataFrame(scipy.io.loadmat(file_path))
            if format_type == "feather":
                if PYARROW_AVAILABLE:
                    return feather.read_feather(file_path)
                return pd.read_feather(file_path)
            if format_type == "arrow" and PYARROW_AVAILABLE:
                return pa.ipc.open_file(file_path).read_pandas()
            if format_type == "sqlite":
                return pd.read_sql_query("SELECT * FROM data", f"sqlite:///{file_path}")
            raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:  # pragma: no cover - data errors
            raise Exception(f"Error reading {file_path}: {e}")


class DataWriter:
    """Handles writing data to various file formats."""

    @staticmethod
    def write_file(
        df: pd.DataFrame, file_path: str, format_type: str, **kwargs
    ) -> None:
        """Write file based on format type."""
        try:
            if format_type == "csv":
                df.to_csv(file_path, index=False, **kwargs)
                return
            if format_type == "tsv":
                df.to_csv(file_path, sep="	", index=False, **kwargs)
                return
            if format_type == "parquet":
                if PYARROW_AVAILABLE:
                    table = pa.Table.from_pandas(df)
                    pq.write_table(
                        table,
                        file_path,
                        compression=kwargs.get("compression", "snappy"),
                    )
                else:
                    df.to_parquet(file_path, **kwargs)
                return
            if format_type == "excel":
                df.to_excel(file_path, index=False, engine="openpyxl", **kwargs)
                return
            if format_type == "json":
                df.to_json(file_path, orient="records", **kwargs)
                return
            if format_type == "hdf5":
                df.to_hdf(file_path, key="data", **kwargs)
                return
            if format_type == "pickle":
                df.to_pickle(file_path)
                return
            if format_type == "numpy":
                np.save(file_path, df.values)
                return
            if format_type == "matlab" and SCIPY_AVAILABLE:
                scipy.io.savemat(
                    file_path, {"data": df.values, "columns": df.columns.tolist()}
                )
                return
            if format_type == "feather":
                if PYARROW_AVAILABLE:
                    feather.write_feather(df, file_path)
                else:
                    df.to_feather(file_path)
                return
            if format_type == "arrow" and PYARROW_AVAILABLE:
                table = pa.Table.from_pandas(df)
                with pa.ipc.new_file(file_path, table.schema) as writer:
                    writer.write_table(table)
                return
            if format_type == "sqlite":
                import sqlite3

                conn = sqlite3.connect(file_path)
                df.to_sql("data", conn, if_exists="replace", index=False)
                conn.close()
                return
            raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:  # pragma: no cover - data errors
            raise Exception(f"Error writing {file_path}: {e}")
