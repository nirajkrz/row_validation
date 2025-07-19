#!/usr/bin/env python3
"""
Complete test runner for the data validation script.
This will generate test files and run the validation process.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random
import string
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional

# Import the DataValidator class (assuming it's in the same directory)
# If you saved the validator as a separate file, adjust the import accordingly
class DataValidator:
    def __init__(self):
        """Initialize the Data Validator with logging."""
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Define column mappings (feed -> output)
        self.column_mappings = {
            'preferredPhoneIndicator': 'prefrrdIndicator',
            'sourcePartyTechnicalidentifier': None,  # Dropped
            'phoneTechnicalIdentifier': None,        # Dropped
        }
        
        # Columns to exclude from comparison (typically dropped technical identifiers)
        self.excluded_columns = [
            'sourcePartyTechnicalidentifier',
            'phoneTechnicalIdentifier'
        ]

    def load_data_efficiently(self, file_path: str, delimiter: str = '|') -> pd.DataFrame:
        """Load large data files efficiently, handling different formats."""
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.dat':
                # Handle DAT files with specific format
                self.logger.info("Processing DAT file with || delimiter and date header")
                
                # Read all lines and skip the first redundant date line
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Skip first line (contains date)
                if lines and ('date' in lines[0].lower() or len(lines[0].strip().split('||')) < 3):
                    self.logger.info(f"Skipping header line: {lines[0].strip()}")
                    data_lines = lines[1:]
                else:
                    data_lines = lines
                
                # Parse lines with || delimiter
                rows = []
                headers = None
                
                for i, line in enumerate(data_lines):
                    if line.strip():  # Skip empty lines
                        parts = [part.strip() for part in line.strip().split('||')]
                        
                        # First data row becomes headers
                        if headers is None:
                            headers = parts
                            self.logger.info(f"Found {len(headers)} columns in DAT file")
                        else:
                            # Ensure all rows have same number of columns as headers
                            while len(parts) < len(headers):
                                parts.append('')
                            if len(parts) > len(headers):
                                parts = parts[:len(headers)]
                            rows.append(parts)
                
                # Create DataFrame
                df = pd.DataFrame(rows, columns=headers)
                
            elif file_extension == '.csv':
                # Handle CSV files (output from API)
                self.logger.info("Processing CSV file")
                df = pd.read_csv(
                    file_path,
                    dtype=str,  # Keep everything as string for comparison
                    na_values=['', 'NULL', 'null', 'None'],
                    keep_default_na=False,
                    low_memory=False,
                    encoding='utf-8'
                )
                
            else:
                # Fallback for other formats
                self.logger.warning(f"Unknown file extension {file_extension}, trying generic CSV parsing")
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    dtype=str,
                    na_values=['', 'NULL', 'null', 'None'],
                    keep_default_na=False,
                    low_memory=False,
                    encoding='utf-8'
                )
            
            # Clean up column names (remove extra spaces, normalize)
            df.columns = df.columns.str.strip()
            
            # Replace empty strings and 'null' variations with actual NaN for consistency
            df = df.replace(['', 'null', 'NULL', 'None', 'NONE'], pd.NA)
            
            self.logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def normalize_column_names(self, df: pd.DataFrame, is_output: bool = False) -> pd.DataFrame:
        """Normalize column names and apply mappings."""
        df_copy = df.copy()
        
        if not is_output:
            # For feed file, apply column mappings and remove excluded columns
            new_columns = {}
            columns_to_keep = []
            
            for col in df_copy.columns:
                if col in self.excluded_columns:
                    continue  # Skip excluded columns
                elif col in self.column_mappings and self.column_mappings[col] is not None:
                    new_columns[col] = self.column_mappings[col]
                    columns_to_keep.append(col)
                else:
                    columns_to_keep.append(col)
            
            df_copy = df_copy[columns_to_keep]
            df_copy = df_copy.rename(columns=new_columns)
        
        # Normalize column names (lowercase, strip spaces)
        df_copy.columns = df_copy.columns.str.lower().str.strip()
        
        return df_copy

    def create_composite_key(self, df: pd.DataFrame, key_columns: List[str]) -> pd.Series:
        """Create a composite key for matching rows."""
        available_keys = [col for col in key_columns if col in df.columns]
        
        if not available_keys:
            self.logger.warning("No key columns found. Using row index.")
            return df.index.astype(str)
        
        # Handle NaN values and create composite key
        composite_key = df[available_keys].fillna('').astype(str).agg('|'.join, axis=1)
        return composite_key

    def compare_dataframes(self, feed_df: pd.DataFrame, output_df: pd.DataFrame, 
                          key_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Compare two dataframes and identify mismatches."""
        self.logger.info("Starting data comparison...")
        
        # Normalize data
        feed_norm = self.normalize_column_names(feed_df, is_output=False)
        output_norm = self.normalize_column_names(output_df, is_output=True)
        
        # Create composite keys
        feed_key = self.create_composite_key(feed_norm, key_columns)
        output_key = self.create_composite_key(output_norm, key_columns)
        
        feed_norm['_key'] = feed_key
        output_norm['_key'] = output_key
        
        # Find common columns for comparison
        common_cols = list(set(feed_norm.columns) & set(output_norm.columns))
        common_cols.remove('_key')  # Remove the key column from comparison
        
        self.logger.info(f"Comparing {len(common_cols)} common columns")
        
        # Merge dataframes on key
        merged = pd.merge(
            feed_norm.set_index('_key'),
            output_norm.set_index('_key'),
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('_feed', '_output'),
            indicator=True
        )
        
        # Identify different types of mismatches
        mismatches = []
        summary = {
            'total_feed_rows': len(feed_df),
            'total_output_rows': len(output_df),
            'matched_rows': 0,
            'missing_in_output': 0,
            'extra_in_output': 0,
            'data_mismatches': 0,
            'column_comparison': {}
        }
        
        # Check for missing/extra rows
        missing_in_output = merged[merged['_merge'] == 'left_only']
        extra_in_output = merged[merged['_merge'] == 'right_only']
        both_present = merged[merged['_merge'] == 'both']
        
        summary['missing_in_output'] = len(missing_in_output)
        summary['extra_in_output'] = len(extra_in_output)
        
        # For rows present in both, check data mismatches
        for col in common_cols:
            feed_col = f"{col}_feed"
            output_col = f"{col}_output"
            
            if feed_col in both_present.columns and output_col in both_present.columns:
                # Compare values (handle NaN appropriately)
                col_mismatches = both_present[
                    (both_present[feed_col].fillna('') != both_present[output_col].fillna(''))
                ]
                
                if len(col_mismatches) > 0:
                    summary['column_comparison'][col] = len(col_mismatches)
                    
                    # Add to mismatches list
                    for idx, row in col_mismatches.iterrows():
                        mismatches.append({
                            'key': idx,
                            'column': col,
                            'feed_value': row[feed_col],
                            'output_value': row[output_col],
                            'mismatch_type': 'data_mismatch'
                        })
        
        # Add missing/extra row mismatches
        for idx, row in missing_in_output.iterrows():
            mismatches.append({
                'key': idx,
                'column': 'ALL',
                'feed_value': 'Present',
                'output_value': 'Missing',
                'mismatch_type': 'missing_in_output'
            })
        
        for idx, row in extra_in_output.iterrows():
            mismatches.append({
                'key': idx,
                'column': 'ALL',
                'feed_value': 'Missing',
                'output_value': 'Present',
                'mismatch_type': 'extra_in_output'
            })
        
        summary['matched_rows'] = len(both_present) - len([m for m in mismatches if m['mismatch_type'] == 'data_mismatch'])
        summary['data_mismatches'] = len([m for m in mismatches if m['mismatch_type'] == 'data_mismatch'])
        
        mismatch_df = pd.DataFrame(mismatches)
        
        return mismatch_df, summary

    def generate_excel_report(self, mismatch_df: pd.DataFrame, summary: Dict, 
                             output_path: str = 'data_validation_report.xlsx'):
        """Generate detailed Excel report."""
        self.logger.info(f"Generating Excel report: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Column comparison details
            if summary['column_comparison']:
                col_comp_df = pd.DataFrame([
                    {'Column': col, 'Mismatch_Count': count} 
                    for col, count in summary['column_comparison'].items()
                ])
                col_comp_df.to_excel(writer, sheet_name='Column_Comparison', index=False)
            
            # Detailed mismatches
            if not mismatch_df.empty:
                # Split by mismatch type for easier analysis
                for mismatch_type in mismatch_df['mismatch_type'].unique():
                    type_df = mismatch_df[mismatch_df['mismatch_type'] == mismatch_type]
                    sheet_name = mismatch_type.replace('_', ' ').title()[:31]  # Excel sheet name limit
                    type_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Excel report generated successfully: {output_path}")

    def validate_data(self, feed_file: str, output_file: str, 
                     key_columns: List[str], output_report: str = 'validation_report.xlsx'):
        """Main validation function."""
        start_time = time.time()
        
        try:
            # Load data
            self.logger.info("=== Starting Data Validation Process ===")
            feed_df = self.load_data_efficiently(feed_file)
            output_df = self.load_data_efficiently(output_file)
            
            # Compare data
            mismatch_df, summary = self.compare_dataframes(feed_df, output_df, key_columns)
            
            # Generate report
            self.generate_excel_report(mismatch_df, summary, output_report)
            
            # Print summary
            elapsed_time = time.time() - start_time
            self.logger.info("=== Validation Summary ===")
            self.logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Feed file rows: {summary['total_feed_rows']:,}")
            self.logger.info(f"Output file rows: {summary['total_output_rows']:,}")
            self.logger.info(f"Matched rows: {summary['matched_rows']:,}")
            self.logger.info(f"Missing in output: {summary['missing_in_output']:,}")
            self.logger.info(f"Extra in output: {summary['extra_in_output']:,}")
            self.logger.info(f"Data mismatches: {summary['data_mismatches']:,}")
            
            return mismatch_df, summary
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise


def generate_test_files():
    """Generate realistic test data files."""
    
    np.random.seed(42)
    random.seed(42)
    
    num_rows = 1000  # Adjust for larger test datasets
    
    print(f"🔄 Generating {num_rows} rows of test data...")
    
    # Helper functions
    def generate_party_id():
        return f"PTY_{random.randint(100000, 999999)}"
    
    def generate_phone():
        return f"{random.randint(1000000000, 9999999999)}"
    
    def generate_country_code():
        return random.choice(['+1', '+44', '+91', '+86', '+33', '+49', '+81'])
    
    def generate_extension():
        return str(random.randint(1000, 9999)) if random.random() > 0.7 else ''
    
    def generate_flag():
        return random.choice(['Y', 'N'])
    
    def generate_purpose():
        return random.choice(['PRIMARY', 'SECONDARY', 'BUSINESS', 'HOME', 'MOBILE'])
    
    def generate_country():
        return random.choice(['USA', 'UK', 'INDIA', 'CHINA', 'FRANCE', 'GERMANY', 'JAPAN'])
    
    # Generate base data
    data = []
    for i in range(num_rows):
        party_id = generate_party_id()
        source_party_id = f"SRC_{party_id}"
        phone_num = generate_phone()
        
        row = {
            'partyIdentifier': party_id,
            'sourcePartyIdentifier': source_party_id,
            'sourcePartyTechnicalidentifier': f"TECH_{random.randint(10000, 99999)}",
            'phoneTechnicalIdentifier': f"PH_TECH_{random.randint(1000, 9999)}",
            'extension': generate_extension(),
            'verifiedPhoneflag': generate_flag(),
            'phonePurpose': generate_purpose(),
            'otpFlag': generate_flag(),
            'preferredPhoneIndicator': generate_flag(),
            'smsEnabledFlag': generate_flag(),
            'phoneCountryCode': generate_country_code(),
            'phoneCountry': generate_country(),
            'phoneIddCountryCode': str(random.randint(1, 999)),
            'whatsappEnabledFlag': generate_flag(),
            'phoneNumber': phone_num,
            'isdCode': str(random.randint(1, 999))
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create DAT feed file
    print("📄 Creating DAT feed file...")
    with open('input_feed.dat', 'w', encoding='utf-8') as f:
        f.write(f"Data extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        headers = '||'.join(df.columns)
        f.write(f"||{headers}||\n")
        
        for _, row in df.iterrows():
            row_data = '||'.join([str(val) if pd.notna(val) else '' for val in row.values])
            f.write(f"||{row_data}||\n")
    
    # Create CSV output file with transformations
    print("📊 Creating CSV output file with transformations...")
    output_df = df.copy()
    
    # Apply transformations
    output_df = output_df.drop(['sourcePartyTechnicalidentifier', 'phoneTechnicalIdentifier'], axis=1)
    output_df = output_df.rename(columns={'preferredPhoneIndicator': 'prefrrdIndicator'})
    
    # Introduce test scenarios
    missing_indices = random.sample(range(len(output_df)), 5)
    output_df = output_df.drop(missing_indices)
    
    # Add extra rows
    for i in range(3):
        party_id = generate_party_id()
        extra_row = {
            'partyIdentifier': party_id,
            'sourcePartyIdentifier': f"SRC_{party_id}",
            'extension': generate_extension(),
            'verifiedPhoneflag': generate_flag(),
            'phonePurpose': generate_purpose(),
            'otpFlag': generate_flag(),
            'prefrrdIndicator': generate_flag(),
            'smsEnabledFlag': generate_flag(),
            'phoneCountryCode': generate_country_code(),
            'phoneCountry': generate_country(),
            'phoneIddCountryCode': str(random.randint(1, 999)),
            'whatsappEnabledFlag': generate_flag(),
            'phoneNumber': generate_phone(),
            'isdCode': str(random.randint(1, 999))
        }
        output_df = pd.concat([output_df, pd.DataFrame([extra_row])], ignore_index=True)
    
    # Introduce data mismatches
    change_indices = random.sample(range(len(output_df)), min(8, len(output_df)))
    for idx in change_indices:
        output_df.loc[idx, 'phonePurpose'] = 'UPDATED_' + str(output_df.loc[idx, 'phonePurpose'])
    
    flag_change_indices = random.sample(range(len(output_df)), min(6, len(output_df)))
    for idx in flag_change_indices:
        current_flag = output_df.loc[idx, 'smsEnabledFlag']
        output_df.loc[idx, 'smsEnabledFlag'] = 'N' if current_flag == 'Y' else 'Y'
    
    # Shuffle output
    output_df = output_df.sample(frac=1).reset_index(drop=True)
    output_df.to_csv('palantir_output.csv', index=False)
    
    print("✅ Test files generated successfully!")
    print(f"   📁 input_feed.dat: {len(df)} rows")
    print(f"   📁 p_output.csv: {len(output_df)} rows")
    
    return len(missing_indices), 3, len(change_indices) + len(flag_change_indices)


def main():
    """Main test runner function."""
    print("🚀 Starting Complete Data Validation Test")
    print("=" * 50)
    
    # Step 1: Generate test files
    expected_missing, expected_extra, expected_mismatches = generate_test_files()
    
    # Step 2: Run validation
    print("\n🔍 Running validation process...")
    validator = DataValidator()
    
    key_columns = ['partyidentifier', 'sourcepartyidentifier', 'phonenumber']
    
    try:
        mismatches, summary = validator.validate_data(
            feed_file='input_feed.dat',
            output_file='palantir_output.csv',
            key_columns=key_columns,
            output_report='test_validation_report.xlsx'
        )
        
        # Step 3: Analyze results
        print("\n📋 Test Results Analysis")
        print("=" * 30)
        print(f"Expected missing rows: {expected_missing} | Found: {summary['missing_in_output']}")
        print(f"Expected extra rows: {expected_extra} | Found: {summary['extra_in_output']}")
        print(f"Expected data mismatches: ~{expected_mismatches} | Found: {summary['data_mismatches']}")
        
        if not mismatches.empty:
            print(f"\n📊 Mismatch breakdown:")
            print(mismatches['mismatch_type'].value_counts().to_string())
        
        print(f"\n📄 Detailed report saved as: test_validation_report.xlsx")
        print("🎉 Test completed successfully!")
        
        # Step 4: Show sample mismatches if any
        if not mismatches.empty:
            print(f"\n🔍 Sample mismatches (first 5):")
            print("-" * 40)
            sample_mismatches = mismatches.head()
            for _, row in sample_mismatches.iterrows():
                print(f"Key: {row['key']}")
                print(f"  Column: {row['column']}")
                print(f"  Feed: {row['feed_value']}")
                print(f"  Output: {row['output_value']}")
                print(f"  Type: {row['mismatch_type']}")
                print()
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        raise
    
    print("\n" + "=" * 50)
    print("✨ All tests completed! Check the Excel report for detailed analysis.")


if __name__ == "__main__":
    main()
