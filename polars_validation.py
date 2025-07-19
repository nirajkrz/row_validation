import polars as pl
import pandas as pd
import logging
import time


class DataValidatorPolars:
    def __init__(self, mapping_file: str):
        self.mapping_file = mapping_file
        self.logger = self._init_logger()
        self.column_mapping = self._load_column_mapping()

    def _init_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging.getLogger("DataValidatorPolars")

    def _load_column_mapping(self):
        df = pd.read_excel(self.mapping_file) if self.mapping_file.endswith(".xlsx") else pd.read_csv(self.mapping_file)
        mapping = {}
        excluded = set()

        for _, row in df.iterrows():
            feed_col = str(row["feed_column"]).strip()
            output_col = str(row["output_column"]).strip() if not pd.isna(row["output_column"]) else feed_col
            is_excluded = str(row.get("exclude", "")).strip().lower() == "yes"

            if is_excluded:
                excluded.add(feed_col)
            else:
                mapping[feed_col] = output_col if output_col else feed_col

        return {"mapping": mapping, "excluded": excluded}

    def _load_file(self, path: str) -> pl.DataFrame:
        if path.endswith(".dat"):
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip("|\n") for line in f if line.strip()]
                if "date" in lines[0].lower():
                    lines = lines[1:]
                header = lines[0].split("||")
                rows = [line.split("||") for line in lines[1:]]
            return pl.DataFrame(rows, schema=header)
        elif path.endswith(".csv"):
            return pl.read_csv(path, ignore_errors=True)
        else:
            raise ValueError("Unsupported file format")

    def _normalize_columns(self, df: pl.DataFrame, is_output=False) -> pl.DataFrame:
        mapping = self.column_mapping["mapping"]
        excluded = self.column_mapping["excluded"]

        if not is_output:
            df = df.drop([col for col in df.columns if col in excluded])
            rename_map = {col: mapping[col] for col in df.columns if col in mapping and mapping[col] != col}
            df = df.rename(rename_map)

        df = df.rename({col: col.strip().lower() for col in df.columns})
        return df

    def _generate_key(self, df: pl.DataFrame, key_columns):
        key_cols = [col.lower() for col in key_columns if col.lower() in df.columns]
        if not key_cols:
            df = df.with_row_index("key")
        else:
            df = df.with_columns(
                pl.concat_str([pl.col(c).cast(str).fill_null('') for c in key_cols], separator="|").alias("key")
            )
        return df

    def _compare(self, feed_df: pl.DataFrame, output_df: pl.DataFrame, key_columns):
        feed_df = self._normalize_columns(feed_df, is_output=False)
        output_df = self._normalize_columns(output_df, is_output=True)

        feed_df = self._generate_key(feed_df, key_columns)
        output_df = self._generate_key(output_df, key_columns)

        joined = feed_df.join(output_df, on="key", how="full", suffix="_out")

        common_cols = list(set(c.replace("_out", "") for c in joined.columns if c.endswith("_out")) &
                           set(c for c in joined.columns if not c.endswith("_out") and c != "key"))

        mismatches = []
        for col in common_cols:
            col_feed = col
            col_out = f"{col}_out"
            mismatched = joined.filter(
                (pl.col(col_feed).cast(str).fill_null('') != pl.col(col_out).cast(str).fill_null(''))
                & pl.col(col_feed).is_not_null()
                & pl.col(col_out).is_not_null()
            ).select(["key", col_feed, col_out])
            if mismatched.height > 0:
                for row in mismatched.iter_rows(named=True):
                    mismatches.append({
                        "key": row["key"],
                        "column": col,
                        "feed_value": row[col_feed],
                        "output_value": row[col_out],
                        "mismatch_type": "data_mismatch"
                    })

        keys_feed = set(feed_df["key"].to_list())
        keys_output = set(output_df["key"].to_list())

        missing = keys_feed - keys_output
        extra = keys_output - keys_feed

        for k in missing:
            mismatches.append({"key": k, "column": "ALL", "feed_value": "Present", "output_value": "Missing", "mismatch_type": "missing_in_output"})
        for k in extra:
            mismatches.append({"key": k, "column": "ALL", "feed_value": "Missing", "output_value": "Present", "mismatch_type": "extra_in_output"})

        summary = {
            "total_feed_rows": feed_df.shape[0],
            "total_output_rows": output_df.shape[0],
            "matched_rows": len(keys_feed & keys_output) - len([m for m in mismatches if m["mismatch_type"] == "data_mismatch"]),
            "missing_in_output": len(missing),
            "extra_in_output": len(extra),
            "data_mismatches": len([m for m in mismatches if m["mismatch_type"] == "data_mismatch"])
        }

        return mismatches, summary

    def _write_excel_report(self, mismatches, summary, report_path: str):
        with pd.ExcelWriter(report_path, engine="xlsxwriter") as writer:
            pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
            if mismatches:
                df = pd.DataFrame(mismatches)
                for mismatch_type in df["mismatch_type"].unique():
                    df[df["mismatch_type"] == mismatch_type].to_excel(
                        writer, sheet_name=mismatch_type.replace("_", " ").title()[:31], index=False
                    )

    def validate(self, feed_file: str, output_file: str, key_columns, report_path="validation_report_polars.xlsx"):
        start = time.time()
        try:
            self.logger.info("🔍 Loading feed and output files...")
            feed_df = self._load_file(feed_file)
            output_df = self._load_file(output_file)

            self.logger.info("📊 Comparing data...")
            mismatches, summary = self._compare(feed_df, output_df, key_columns)

            self._write_excel_report(mismatches, summary, report_path)

            elapsed = time.time() - start
            self.logger.info(f"✅ Validation complete in {elapsed:.2f}s")
            for k, v in summary.items():
                self.logger.info(f"{k}: {v}")
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {str(e)}")
            raise


if __name__ == "__main__":
    validator = DataValidatorPolars(mapping_file='column_mapping.xlsx')
    validator.validate(
        feed_file='input_feed.dat',
        output_file='api_output.csv',
        key_columns=['partyIdentifier', 'sourcePartyIdentifier', 'phoneNumber'],
        report_path='validation_report_polars.xlsx'
    )
