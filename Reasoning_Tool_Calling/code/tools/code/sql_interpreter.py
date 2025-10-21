import mysql
import mysql.connector as msql


def check_sql(sql_cmd):
    # Split the SQL command into parts
    sql_parts = sql_cmd.split()
    
    # Find the FROM keyword position
    try:
        from_index = [i for i, part in enumerate(sql_parts) if part.upper() == 'FROM'][0]
        
        # Check the table name after FROM
        if from_index + 1 < len(sql_parts) and sql_parts[from_index + 1].lower() == 'flights':
            # Replace 'flights' with 'flights.flight_data'
            sql_parts[from_index + 1] = 'flights.flight_data'
        elif from_index + 1 < len(sql_parts) and sql_parts[from_index + 1].lower() == 'coffee':
            # Replace 'coffee' with 'coffee.coffee_data'
            sql_parts[from_index + 1] = 'coffee.coffee_data'
        elif from_index + 1 < len(sql_parts) and sql_parts[from_index + 1].lower() == 'airbnb':
            # Replace 'airbnb' with 'airbnb.airbnb_data'
            sql_parts[from_index + 1] = 'airbnb.airbnb_data'
        elif from_index + 1 < len(sql_parts) and sql_parts[from_index + 1].lower() == 'yelp':
            # Replace 'yelp' with 'yelp.yelp_data'
            sql_parts[from_index + 1] = 'yelp.yelp_data'       
        # Reconstruct the SQL command
        return ' '.join(sql_parts)
    except (IndexError, ValueError):
        # If 'FROM' is not found or any other error occurs, return the original command
        return sql_cmd



def execute(sql_cmd):
    sql_cmd = check_sql(sql_cmd)
    conn = msql.connect(host="localhost", user='root',  
                        password="")#give ur username, password
    cursor = conn.cursor()
    cursor.execute(sql_cmd)

    column_names = [i[0] for i in cursor.description]
    rows = cursor.fetchall()
    # return rows
    rows_string = []
    for row in rows:
        current_row = [column_names[i]+": "+str(row[i]) for i in range(len(row))]
        current_row = ', '.join(current_row)
        rows_string.append(current_row)
    rows_string = '\n'.join(rows_string)
    return rows_string

if __name__ == "__main__":
    sql_cmd = "SELECT Flight_Number_Marketing_Airline FROM flights.flights_data WHERE Airline = 'Southwest Airlines Co.' AND Origin = 'MIA' AND Dest = 'BWI' AND FlightDate = '2022-02-11';"
    rows = execute(sql_cmd)
    print(rows)
