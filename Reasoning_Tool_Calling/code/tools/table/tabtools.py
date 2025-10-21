import pandas as pd
import jsonlines
import json
import re

class table_toolkits():
    # init
    def __init__(self, path):
        self.data = None
        self.path = path
        self.backup_data = None

    def get_column_names(self, target_db):
        if self.data is None:
            return "No database loaded. Please load a database first."
        return ', '.join(self.data.columns.tolist())

    def reset_data(self):
        if self.backup_data is not None:
            mid_point = len(self.backup_data) // 2
            first_half = self.backup_data.iloc[:mid_point].copy()
            second_half = self.backup_data.iloc[mid_point:].copy()
            self.data = pd.concat([first_half, second_half], ignore_index=True)
            del first_half, second_half
        else:
            self.data = None
        return "The data has been reset. Please load a database again."

    def db_loader(self, target_db):
        if target_db == 'flights':
            file_path = "{}/data/external_corpus/flights/Combined_Flights_2022.csv".format(self.path)
            self.backup_data = pd.read_csv(file_path)
        elif target_db == 'coffee':
            file_path = "{}/data/external_corpus/coffee/coffee_price.csv".format(self.path)
            self.backup_data = pd.read_csv(file_path)
        elif target_db =='airbnb':
            file_path = "{}/data/external_corpus/airbnb/Airbnb_Open_Data.csv".format(self.path)
            self.backup_data = pd.read_csv(file_path)
        elif target_db == 'yelp':
            data_file = open("{}/data/external_corpus/yelp/yelp_academic_dataset_business.json".format(self.path))
            data = []
            for line in data_file:
                data.append(json.loads(line))
            self.backup_data = pd.DataFrame(data)
            data_file.close()
        self.backup_data = self.backup_data.astype(str)
        column_names = ', '.join(self.backup_data.columns.tolist())
        self.data = self.backup_data.copy()
        return "We have successfully loaded the {} database, including the following columns: {}.".format(target_db, column_names)

    # def get_column_names(self, target_db):
    #     return ', '.join(self.data.columns.tolist())

    
    

    def data_filter(self, argument):
       # print("Filtering data with argument:", argument)
        # commands = re.sub(r' ', '', argument)
        #if self.data is None:
        #    print("Data is None, loading backup data.")
            #self.data = self.backup_data.copy()

        #print("passed first if")
        # Remove single quotes from argument before splitting
        argument = argument.replace("'", "")
        argument = argument.replace('"', '')
        commands = argument.split(', ')
        
        for i in range(len(commands)):
            try:
                commands[i] = commands[i].strip()
                if '>=' in commands[i]:
                    command = commands[i].split('>=')
                    column_name = command[0]
                    value = command[1]
                    self.data = self.data[self.data[column_name] >= value]
                elif '<=' in commands[i]:
                    command = commands[i].split('<=')
                    column_name = command[0]
                    value = command[1]
                    self.data = self.data[self.data[column_name] <= value]
                elif '>' in commands[i]:
                    command = commands[i].split('>')
                    column_name = command[0]
                    value = command[1]
                    self.data = self.data[self.data[column_name] > value]
                elif '<' in commands[i]:
                    command = commands[i].split('<')
                    column_name = command[0]
                    value = command[1]
                    self.data = self.data[self.data[column_name] < value]
                elif '=' in commands[i]:
                    #print("here inside equal")
                    command = commands[i].split('=')
                    column_name = command[0]
                   # print("\n", column_name, "\n")
                    value = command[1]
                    self.data = self.data[self.data[column_name] == value]
                    #print("after equal")
                if len(self.data) == 0:
                    #self.data = None 
                    return "No data found. inside the db there is 0 data with the filtering query {} and mabye this can be helpfull for your research or the filtering query {} is incorrect.".format(commands[i])
            except:
                return "we have failed when conducting the {} command. Please make changes.".format(commands[i])
        current_length = len(self.data)
        if len(self.data) > 0:
           # print(self.data)
            return "We have successfully filtered the data ({} rows).".format(current_length)
        else:
            # convert to strings, with comma as column separator and '\n' as row separator
            return_answer = []
            for i in range(len(self.data)):
                outputs = []
                for attr in list(self.data.columns):
                    outputs.append(str(attr)+": "+str(self.data.iloc[i][attr]))
                outputs = ', '.join(outputs)
                return_answer.append(outputs)
            return_answer = '\n'.join(return_answer)
            return return_answer, self.data

    def get_value(self, argument):
        column = argument
        if len(self.data) == 1:
            return str(self.data.iloc[0][column])
        else:
            return ', '.join(self.data[column].tolist())

if __name__ == "__main__":
    print("start")
    db = table_toolkits("/home/ubuntu/vdb1/Agent_design_architectures/ToolQA")
    print(db.db_loader('flights'))
    #print(db.get_column_names('flights'))
    #print(db.data_filter('IATA_Code_Marketing_Airline=AA, Flight_Number_Marketing_Airline=5647, Origin=BUF, Dest=PHL, FlightDate=2022-04-20'))
    #print(db.get_value('DepTime')) Origin=BOS, Dest=BZN
    print(db.data_filter('Origin=PIT, FlightDate=2022-05-17'))
    print(db.get_value('DepTime'))