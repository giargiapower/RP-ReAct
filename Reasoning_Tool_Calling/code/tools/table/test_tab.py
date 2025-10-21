import tabtools as tabtools

db = tabtools.table_toolkits("/leonardo_scratch/fast/IscrC_EAGLE-DE/gianni/Agent_design_architectures/ToolQA")
print(db.db_loader('flights'))
    #print(db.get_column_names('flights'))
    #print(db.data_filter('IATA_Code_Marketing_Airline=AA, Flight_Number_Marketing_Airline=5647, Origin=BUF, Dest=PHL, FlightDate=2022-04-20'))
    #print(db.get_value('DepTime')) Origin=BOS, Dest=BZN  Flight_Number_Marketing_Airline=4631, FlightDate=2022-06-27
data = db.data_filter(' Flight_Number_Marketing_Airline=4869, FlightDate=2022-05-06, Origin=HHH, Dest=CLT ')
print(data)