import os
import pandas as pd

def calculate_std_dev_for_csv_files():
    """
    Reads each CSV file from 'table_total_improvement' directory,
    calculates the standard deviation for each row (agent type),
    and saves the result to a new CSV file in the 'standard_deviation' directory.
    """
    input_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/table_total_improvement'
    output_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/standard_deviation'

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

                # Calculate the standard deviation for each row (axis=1)
                std_dev_series = df.std(axis=1, numeric_only=True)
                mean_series = df.mean(axis=1, numeric_only=True)

                # Create a new DataFrame from the results
                result_df = pd.DataFrame({
                    'Agent Type': std_dev_series.index,
                    'Standard Deviation': std_dev_series.values,
                    'Mean': mean_series.values
                })

                # Construct the output file path
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_std.csv"
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
    calculate_std_dev_for_csv_files()
