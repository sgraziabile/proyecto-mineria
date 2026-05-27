"""
Preprocessing utilities for aggregating exercise-level data into routine-level data.
"""
import pandas as pd
import numpy as np

def aggregate_routine_data(df, group_by_col='routine_id'):
    """
    Aggregates exercise-per-row data into routine-per-row format.
    
    Parameters:
    - df: pandas DataFrame containing exercise-level records.
    - group_by_col: The column name representing the unique routine identifier.
    
    Returns:
    - aggregated_df: pandas DataFrame with routines per row.
    """
    # Example aggregation dictionary - adapt to your specific column names
    # This addresses the main preprocessing challenge mentioned.
    agg_funcs = {
        'weight': ['mean', 'max', 'sum'],   # Systemic load proxy
        'reps': ['mean', 'sum'],            # Volume proxy
        'sets': ['sum'],                    # Total sets in routine
        'intensity': ['mean', 'max'],       # Intensity proxy
        # Add target variable aggregation (e.g., if 'experience_level' is same for all exercises in a routine)
        'experience_level': 'first'
    }
    
    # Note: Only aggregate columns that exist in the dataframe.
    # Filter agg_funcs to match columns present in df
    valid_agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    if not valid_agg_funcs:
        print("Warning: None of the default aggregation columns were found in the dataset.")
        return pd.DataFrame()
        
    aggregated_df = df.groupby(group_by_col).agg(valid_agg_funcs)
    
    # Flatten multi-index columns resulting from aggregation
    aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
    
    # Rename target column back to its original name for clarity
    if 'experience_level_first' in aggregated_df.columns:
        aggregated_df.rename(columns={'experience_level_first': 'experience_level'}, inplace=True)
        
    return aggregated_df.reset_index()
