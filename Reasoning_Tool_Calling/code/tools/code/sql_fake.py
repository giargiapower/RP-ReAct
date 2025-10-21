import pandas as pd
from pandasql import sqldf
import re

# --- Cache for loaded data ---
_db_cache = {}
# --- Cache for pysqldf functions ---
_pysqldf_cache = {}

# --- Data Loading ---

def load_data(db_name):
    """
    Loads a specific pre-processed data source from a Parquet file.
    Uses a cache to avoid reloading data.
    """
    # Check cache first
    if db_name in _db_cache:
        print(f"Loading {db_name} data from cache...")
        return _db_cache[db_name]

    print(f"Loading {db_name} data from Parquet file...")
    
    base_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus"
    
    # Map db_name to its file path and the expected DataFrame variable name
    db_map = {
        'flights': {'path': f"{base_path}/flights/flights_data.parquet", 'df_name': 'flights_data'},
        'coffee': {'path': f"{base_path}/coffee/coffee_data.parquet", 'df_name': 'coffee_data'},
        'airbnb': {'path': f"{base_path}/airbnb/airbnb_data.parquet", 'df_name': 'airbnb_data'},
        'yelp': {'path': f"{base_path}/yelp/yelp_data.parquet", 'df_name': 'yelp_data'},
    }
    
    if db_name not in db_map:
        print(f"Error: Unknown database '{db_name}'")
        return None, None

    config = db_map[db_name]
    try:
        data = pd.read_parquet(config['path'])
        df_name = config['df_name']
        
        # Store in cache for future use
        _db_cache[db_name] = (data, df_name)
        
        print(f"{db_name} data loaded and cached.")
        return data, df_name
    except FileNotFoundError:
        print(f"Error: Data file not found at {config['path']}")
        return None, None

def get_table_name_from_query(sql_cmd):
    """Extracts the table name from an SQL query."""
    # Use regex to find the table name after FROM, ignoring potential schema prefixes
    match = re.search(r'\bFROM\s+(?:[\w]+\.)?(\w+)', sql_cmd, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def execute(sql_cmd):
    """
    Executes an SQL query against the appropriate in-memory database.
    It determines the database from the query, loads the data if needed,
    and then runs the query.
    """
    table_name = get_table_name_from_query(sql_cmd)
    if not table_name:
        return "Error: Could not find a valid table name in the SQL query."

    # Determine which database to load
    db_name = None
    if 'flights' in table_name:
        db_name = 'flights'
    elif 'coffee' in table_name:
        db_name = 'coffee'
    elif 'airbnb' in table_name:
        db_name = 'airbnb'
    elif 'yelp' in table_name:
        db_name = 'yelp'
    
    if not db_name:
        return f"Error: The table '{table_name}' does not correspond to a known database."

    # Check for a cached pysqldf function
    if db_name in _pysqldf_cache:
        print(f"Using cached pysqldf for {db_name}...")
        pysqldf, df_name = _pysqldf_cache[db_name]
    else:
        print(f"Creating and caching pysqldf for {db_name}...")
        # Load the data (from cache if available) and get the DataFrame variable name
        data, df_name = load_data(db_name)
        if data is None:
            return f"Error: Failed to load data for database '{db_name}'."

        # Create a local scope for sqldf with the loaded DataFrame
        local_scope = {df_name: data}
        pysqldf = lambda q: sqldf(q, local_scope)
        
        # Cache the function and df_name for future use
        _pysqldf_cache[db_name] = (pysqldf, df_name)

    # Replace the public table name with the internal DataFrame variable name
    # This makes the query executable by pandasql
    modified_sql_cmd = re.sub(r'\b' + re.escape(table_name) + r'\b', df_name, sql_cmd, count=1, flags=re.IGNORECASE)

    try:
        # Execute query on the loaded pandas DataFrame
        result_df = pysqldf(modified_sql_cmd)
    except Exception as e:
        return f"Error executing query: {e}"

    if result_df is None or result_df.empty:
        return ""

    # Format the output
    rows_string = []
    for _, row in result_df.iterrows():
        current_row = [f"{col}: {val}" for col, val in row.items()]
        rows_string.append(', '.join(current_row))
    
    return '\n'.join(rows_string)

if __name__ == "__main__":
    # Example usage:
    #print("--- Testing Flights Query (First Time) ---")
    #flights_query = "SELECT COUNT(*) AS total_flights, SUM(CASE WHEN DepDelay > 0 OR ArrDelay > 0 THEN 1 ELSE 0 END) AS delayed_flights FROM flights WHERE Origin = 'JAC' AND FlightDate = '2022-02-24';"
    #flights_query = "SELECT * FROM flights WHERE origin = 'JFK' AND destination = 'LAX';"
    #flights_result = execute(flights_query)
    #print(flights_result)

    #print("\n--- Testing Flights Query (Second Time) ---")
    #flights_result_2 = execute(flights_query_2)
    ##flights_query_2 = "SELECT COUNT(*) FROM flights WHERE Origin = 'JFK';"
    #print(flights_result_2)

    print("\n--- Testing Coffee Query ---")
    coffee_query = "SELECT host name FROM airbnb WHERE NAME = 'Amazing One Bedroom Apartment in Prime Brooklyn in Bushwick' LIMIT 1;"
    coffee_result = execute(coffee_query)
    print(coffee_result)