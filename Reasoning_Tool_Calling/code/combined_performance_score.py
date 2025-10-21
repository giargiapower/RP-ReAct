import os
import pandas as pd

def calculate_cps_for_csv_files():
    """
    Reads each CSV file from 'table_total_improvement' directory,
    calculates the combined performance score for each row (agent type),
    and saves the result to a new CSV file in the 'combined_performance_score' directory.
    """
    input_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/table_total_improvement'
    output_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/combined_performance_score'

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            
            try:
                # Read the CSV, setting the first column as the index
                df = pd.read_csv(input_path, index_col='Agent Type')

                # Calculate max and average for each row (agent type)
                max_values = df.max(axis=1, numeric_only=True)
                avg_values = df.mean(axis=1, numeric_only=True)

                # Calculate saturation line
                saturation_line = 1 - (max_values - avg_values)

                # Calculate Combined Performance Score (CBS)
                cps_series = saturation_line * max_values

                # Create a new DataFrame from the results
                result_df = pd.DataFrame({
                    'Agent Type': cps_series.index,
                    'CPS': cps_series.values
                })

                # Check for rows in the original df containing None or 'No Matching File'
                # Create a boolean mask for invalid entries
                invalid_mask = df.apply(pd.to_numeric, errors='coerce').isnull() | (df == 'No Matching File')
                
                # Identify agent types (rows) that have at least one invalid entry
                rows_to_nullify = invalid_mask.any(axis=1)
                agent_types_to_nullify = rows_to_nullify[rows_to_nullify].index

                # Set CBS to None for those agent types in the result_df
                result_df.loc[result_df['Agent Type'].isin(agent_types_to_nullify), 'CPS'] = None


                # Construct the output file path
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_cps.csv"
                output_path = os.path.join(output_dir, output_filename)

                # Print the table with its name before saving
                print(f"\n--- Table for {output_filename} ---")
                print(result_df)
                print("--------------------------------------\n")

                # Save the new DataFrame to a CSV file
                result_df.to_csv(output_path, index=False)
                print(f"Processed '{filename}' and saved results to '{output_path}'")

            except Exception as e:
                print(f"Could not process file {filename}. Error: {e}")

if __name__ == '__main__':
    calculate_cps_for_csv_files()
