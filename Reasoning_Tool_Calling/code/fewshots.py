TOOLQA_EASY8 = """Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
Thought 1: This is a question related to flights. We need to load the flights database.
Action 1: LoadDB[flights]
Observation 1: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings
Thought 2: We need to filter the information related to the query.
Action 2: FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]
Observation 2: We have successfully filtered the data (1 row).
Thought 3: We then need to know the departure delayed time.
Action 3: GetValue[DepDelay]
Observation 3: -7.0
Thought 4: We then need to know the arrival delayed time.
Action 4: GetValue[ArrDelay]
Observation 4: -17.0
Thought 5: To compute the extra minutes, we need to subtract the departure delayed time from the arrival delayed time.
Action 5: Calculate[(-17)-(-7)]
Observation 5: -10
Thought 6: After calculation, we know that the answer is -10.
Action 6: Finish[-10]

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Thought 1: Maybe I can write a python program to solve this problem.
Action 1: PythonInterpreter[# solution in Python:
def solution():
    # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result]
Observation 1: 33
Thought 2: After execution of Python Code, we know that the answer is 33.
Action 2: Finish[33]

Question: What is the corresponding Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task?
Thought 1: The question is asking some performance information about Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task, we need to retrieve some useful information from the scirex database.
Action 1: RetrieveScirex[Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task]
Observation 1: section : Abstract Semantic image segmentation is an essential component of modern autonomous driving systems , as an accurate understanding of the surrounding scene is crucial to navigation and action planning . Current state - of - the - art approaches in semantic image segmentation rely on pretrained networks that were initially developed for classifying images as a whole . While these networks exhibit outstanding recognition performance ( i.e. , what is visible ? ) , they lack localization accuracy ( i.e. , where precisely is something located ? ) . Therefore , additional processing steps have to be performed in order to obtain pixel - accurate segmentation masks at the full image resolution . To alleviate this problem we propose a novel ResNet - like architecture that exhibits strong localization and recognition performance . We combine multi - scale context with pixel - level accuracy by using two processing streams within our network : One stream carries information at the full image resolution , enabling precise adherence to segment boundaries . The other stream undergoes a sequence of pooling operations to obtain robust features for recognition . The two streams are coupled at the full image resolution using residuals . Without additional processing steps and without pretraining , our approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset .
Thought 2: As the approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset, the answer should be 71.8%.
Action 2: Finish[71.8%]

Question: When was the paper Learning the Principle of Least Action with Reinforcement Learning. published?
Thought 1: The question is asking some basic information of a dblp paper. We need to load the DBLP graph.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: The question is asking the published date of a paper, we need to check the node from the PaperNet in DBLP graph.
Action 2: NodeCheck[PaperNet, Learning the Principle of Least Action with Reinforcement Learning.]
Observation 2: {'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference', 'page_start': '', 'page_end': ''}
Thought 3: The published date of the paper is 2021.
Action 3: Finish[2021]

Question: How many collaborators does Chao Zhang have in the DBLP graph?
Thought 1: The question is asking the number of collaborators of a person, we need load the DBLP graph.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: The question is asking the collaborators of a person, we need to check the neighboring nodes from the AuthorNet in DBLP graph.
Action 2: NeighbourCheck[AuthorNet, Chao Zhang]
Observation 2: ['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']
Thought 3: The number of collaborators of Chao Zhang is 6.
Action 3: Finish[6]

Question: How many papers does Chao Zhang and Weihong Lin have in common in the DBLP graph?
Thought 1: The question is asking the number of common papers of two persons, we need load the DBLP graph.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: The question is asking the common papers of two persons, we need to check the edges between them from the PaperNet in DBLP graph.
Action 2: EdgeCheck[PaperNet, Chao Zhang, Weihong Lin]
Observation 2: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}
Thought 3: The number of common papers of Chao Zhang and Weihong Lin is 1.

Question: Where did Stephen's Opera performance take place?
Thought 1: The question is asking the location of Stephen's Opera performance from agenda.
Action 1: RetrieveAgenda[Stephen's Opera performance]
Observation 1: On January 29, 2022, there will be an opera performance at the Lyric Opera House, featuring Stephen. The show will start at 7:00 PM and end at 9:00 PM. It promises to be a wonderful evening of beautiful music and powerful performances in a stunning venue. Come and experience the magic of opera at its finest!
Thought 2: The event happened in Lyric Opera.
Action 2: Finish[Lyric Opera]

Question: What was the trading volume of coffee on 2000-01-14?
Thought 1: Maybe I can write a SQL query to solve this problem.
Action 1: SQLInterpreter(SELECT Volume FROM coffee.coffee_data WHERE Date = '2000-01-14';
Observation 1: Volume: 10115
Thought 2: The volume of coffee on 2000-01-14 is 10115.
Action 2: Finish[10115]
"""

TOOLQA_HARD3 = """Question: What is the average airtime of the flights from PHX to LGA host by Spirit Air Lines?
Thought 1: This is a question related to flights. We need to load the flights database.
Action 1: LoadDB[flights]
Observation 1: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings
Thought 2: We need to filter the information related to the query.
Action 2: FilterDB[Origin=PHX, Dest=ATL, Airline=Spirit Air Lines]
Observation 2: We have successfully filtered the data (16 rows).
Thought 2: We then obtain all the AirTime values of filtered records.
Action 2: GetValue[AirTime]
Observation 2: [247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, nan, 229.0, nan, 259.0, 253.0]
Thought 3: We then calculate the average of the values. NaN values are ignored.
Action 3: Calculate(mean(247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0))
Observation 3: 249.214
Thought 4: The average airtime of the flights from PHX to LGA host by Spirit Air Lines is 249.2.
Action 4: Finish[249.2]

Question: Can you recommend a Guamanian business with the highest star rating within a 5-mile radius of 6830 Rising Sun Ave?
Thought 1: This is a question related to business. We need to load the yelp database.
Action 1: LoadDB[yelp]
Observation 1: We have successfully loaded the yelp database, including the following columns: business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours
Thought 2: Maybe we can write SQL code to obtain the latitude and longitude of the address.
Action 2: SQLInterpreter[SELECT latitude, longitude FROM yelp.yelp_data WHERE address='6830 Rising Sun Ave']  
Observation 2: latitude: 40.05555, longitude: -75.090723 
Thought 3: Maybe we can write another python code to calculate the coordinates range of 5-mile radius.
Action 3: import geopy\nimport geopy.distance\nlatitude = 40.05555\nlongitude = -75.090723\n_, lo_max, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=90)\n_, lo_min, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=270)\nla_max, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=0)\nla_min, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=180)\nans = (la_max, la_min, lo_max, lo_min)
Observation 3: (40.10058038657396, 40.010519262352226, -75.03212327463275, -75.14932272536724)
Thought 4: We then filter the information related to the query in the yelp database.
Action 4: FilterDB[latitude>=40.010519262352226, longitude>=-75.14932272536724, latitude<=40.10058038657396, longitude<=-75.03212327463275, categories=Guamanian]
Observation 4: We have successfully filtered the data (2 rows).
Thought 5: We then obtain all the stars values of filtered records.
Action 5: GetValue[stars]
Observation 5: [4.0, 4.5]
Thought 6: The highest star rating is 4.5. Select the business with the highest star rating.
Action 6: FilterDB[stars=4.5]
Observation 6: We have successfully filtered the data (1 row).
Thought 7: Get the corresponding name.
Action 7: GetValue[name]
Observation 7: Damaso Painting
Thought 8: The business with the highest star rating is Damaso Painting.
Action 8: Finish[Damaso Painting]

Question: What venue does Chao Zhang with his collarborator publish paper on?
Thought 1: This is a question related to DBLP. We need to load the DBLP graph.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: We need to check the neighbouring nodes of Chao Zhang in AuthorNet.
Action 2: NeighbourCheck[AuthorNet, Chao Zhang]
Observation 2: Weihong Lin
Thought 3: We need to check the between Chao Zhang and Weihong Lin to check their collaboration.
Action 3: EdgeCheck[AuthorNet, Chao Zhang, Weihong Lin]
Observation 3: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}
Thought 4: We need to check the paper detail in the node feature from PaperNet they published.
Action 4: NodeCheck[PaperNet, HRFormer: High-Resolution Vision Transformer for Dense Predict.]
Observation 4: {'title': 'HRFormer: High-Resolution Vision Transformer for Dense Predict.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference', 'page_start': '', 'page_end': ''}
Thought 5: The paper is published on AAAI Spring Symposium - MLPS.
Action 5: Finish[AAAI Spring Symposium - MLPS]
"""

TOOLQA_EASY_SELF_REFINE = """Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
Action 1: LoadDB[flights]
Observation 1: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings
Action 2: FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]
Observation 2: We have successfully filtered the data (1 row).
Action 3: GetValue[DepDelay]
Observation 3: -7.0
Action 4: GetValue[ArrDelay]
Observation 4: -17.0.
Action 5: Calculate[(-17)-(-7)]
Observation 5: -10
Action 6: Finish[-10]

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Action 1: PythonInterpreter[# solution in Python:
def solution():
    # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result]
Observation 1: 33
Action 2: Finish[33]

Question: What is the corresponding Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task?
Action 1: RetrieveScirex[Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task]
Observation 1: section : Abstract Semantic image segmentation is an essential component of modern autonomous driving systems , as an accurate understanding of the surrounding scene is crucial to navigation and action planning . Current state - of - the - art approaches in semantic image segmentation rely on pretrained networks that were initially developed for classifying images as a whole . While these networks exhibit outstanding recognition performance ( i.e. , what is visible ? ) , they lack localization accuracy ( i.e. , where precisely is something located ? ) . Therefore , additional processing steps have to be performed in order to obtain pixel - accurate segmentation masks at the full image resolution . To alleviate this problem we propose a novel ResNet - like architecture that exhibits strong localization and recognition performance . We combine multi - scale context with pixel - level accuracy by using two processing streams within our network : One stream carries information at the full image resolution , enabling precise adherence to segment boundaries . The other stream undergoes a sequence of pooling operations to obtain robust features for recognition . The two streams are coupled at the full image resolution using residuals . Without additional processing steps and without pretraining , our approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset .
Action 2: Finish[71.8%]

Question: When was the paper Learning the Principle of Least Action with Reinforcement Learning. published?
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Action 2: NodeCheck[PaperNet, Learning the Principle of Least Action with Reinforcement Learning.]
Observation 2: {'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference', 'page_start': '', 'page_end': ''}
Action 3: Finish[2021]

Question: How many collaborators does Chao Zhang have in the DBLP graph?
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Action 2: NeighbourCheck[AuthorNet, Chao Zhang]
Observation 2: ['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']
Action 3: Finish[6]

Question: How many papers does Chao Zhang and Weihong Lin have in common in the DBLP graph?
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Action 2: EdgeCheck[PaperNet, Chao Zhang, Weihong Lin]
Observation 2: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}
Action 3: Finish[1]

Question: Where did Stephen's Opera performance take place?
Action 1: RetrieveAgenda[Stephen's Opera performance]
Observation 1: On January 29, 2022, there will be an opera performance at the Lyric Opera House, featuring Stephen. The show will start at 7:00 PM and end at 9:00 PM. It promises to be a wonderful evening of beautiful music and powerful performances in a stunning venue. Come and experience the magic of opera at its finest!
Action 2: Finish[Lyric Opera]

Question: What was the trading volume of coffee on 2000-01-14?
Action 1: SQLInterpreter(SELECT Volume FROM coffee.coffee_data WHERE Date = '2000-01-14';
Observation 1: Volume: 10115
Action 2: Finish[10115]
"""

TOOLQA_HARD3_SELF_REFINE = """Question: What is the average airtime of the flights from PHX to LGA host by Spirit Air Lines?
Action 1: LoadDB[flights]
Observation 1: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings
Action 2: FilterDB[Origin=PHX, Dest=ATL, Airline=Spirit Air Lines]
Observation 2: We have successfully filtered the data (16 rows).
Action 3: GetValue[AirTime]
Observation 3: [247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, nan, 229.0, nan, 259.0, 253.0]
Action 4: Calculate(mean(247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0))
Observation 4: 249.214
Action 5: Finish[249.2]

Question: Can you recommend a Guamanian business with the highest star rating within a 5-mile radius of 6830 Rising Sun Ave?
Action 1: LoadDB[yelp]
Observation 1: We have successfully loaded the yelp database, including the following columns: business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours
Action 2: SQLInterpreter[SELECT latitude, longitude FROM yelp.yelp_data WHERE address='6830 Rising Sun Ave']  
Observation 2: latitude: 40.05555, longitude: -75.090723 
Action 3: PythonInterpreter[import geopy\nimport geopy.distance\nlatitude = 40.05555\nlongitude = -75.090723\n_, lo_max, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=90)\n_, lo_min, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=270)\nla_max, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=0)\nla_min, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=180)\nans = (la_max, la_min, lo_max, lo_min)]
Observation 3: (40.10058038657396, 40.010519262352226, -75.03212327463275, -75.14932272536724)
Action 4: FilterDB[latitude>=40.010519262352226, longitude>=-75.14932272536724, latitude<=40.10058038657396, longitude<=-75.03212327463275, categories=Guamanian]
Observation 4: We have successfully filtered the data (2 rows).
Action 5: GetValue[stars]
Observation 5: [4.0, 4.5]
Action 6: FilterDB[stars=4.5]
Observation 6: We have successfully filtered the data (1 row).
Action 7: GetValue[name]
Observation 7: Damaso Painting
Action 8: Finish[Damaso Painting]

Question: What venue does Chao Zhang with his collarborator publish paper on?
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Action 2: NeighbourCheck[AuthorNet, Chao Zhang]
Observation 2: Weihong Lin
Action 3: EdgeCheck[AuthorNet, Chao Zhang, Weihong Lin]
Observation 3: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}
Action 4: NodeCheck[PaperNet, HRFormer: High-Resolution Vision Transformer for Dense Predict.]
Observation 4: {'title': 'HRFormer: High-Resolution Vision Transformer for Dense Predict.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference', 'page_start': '', 'page_end': ''}
Action 5: Finish[AAAI Spring Symposium - MLPS]
"""



COTQA_SIMPLE6 = """Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
Let's think step by step. I need to first load the flights database with LoadDB[flights]. And then, I need to filter the information I want according to the query with FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]. Then, I need to get the value of the departure time from the database with GetValue[DepDelay]. We then need to know the arrival delayed time with GetValue[ArrDelay]. To compute the extra minutes, we need to subtract the departure delayed time from the arrival delayed time with Calculate[(-17)-(-7)].  After calculation, we know that the answer is -10.
Action 1: LoadDB[flights]-->FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]-->GetValue[DepDelay]-->GetValue[ArrDelay]-->Calculate[(-17)-(-7)]-->Finish[-10]
"""
COT = """
"""

COT_REFLECT = """
"""

REFLECTIONS = """
"""


TOOLQA_EASY8_REASONING = r"""Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
<think>
I load the flights database.
<LoadDB>flights</LoadDB>
<LoadDB_response>database schema loaded</LoadDB_response>

I filter the flight we need.
<FilterDB>Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO</FilterDB>
<FilterDB_response>1 matching row</FilterDB_response>

I obtain the departure delay.
<GetValue>DepDelay</GetValue>
<GetValue_response>-7.0</GetValue_response>

I obtain the arrival delay.
<GetValue>ArrDelay</GetValue>
<GetValue_response>-17.0</GetValue_response>

I compute extra minutes (arrival -- departure).
<Calculate>(-17)-(-7)</Calculate>
<Calculate_response>-10</Calculate_response>

The flight took --10 extra minutes (arrived 10 minutes early).
</think>
Answer: -10


Question: Michael had 58 golf balls. On Tuesday he lost 23, on Wednesday he lost 2 more. How many golf balls did he have at the end of Wednesday?
<think>
Compute 58 - 23 - 2 in Python.
<PythonInterpreter>
golf_balls = 58 - 23 - 2
golf_balls
</PythonInterpreter>
<PythonInterpreter_response>33</PythonInterpreter_response>

Michael has 33 golf balls left.
</think>
Answer: 33


Question: What is the corresponding Mean IoU score of the FRRN method on the Cityscapes dataset for Semantic Segmentation?
<think>
Retrieve the score from the scientific-papers database.
<RetrieveScirex>Mean_IoU score of the FRRN method on Cityscapes dataset</RetrieveScirex>
<RetrieveScirex_response>text mentions “intersection-over-union score of 71.8 %”.</RetrieveScirex_response>

The Mean IoU score is 71.8 %.
</think>
Answer: 71.8 %


Question: When was the paper “Learning the Principle of Least Action with Reinforcement Learning.” published?
<think>
Load the DBLP graph.
<LoadGraph>dblp</LoadGraph>
<LoadGraph_response>graph loaded</LoadGraph_response>

Look up the paper node.
<NodeCheck>PaperNet, Learning the Principle of Least Action with Reinforcement Learning.</NodeCheck>
<NodeCheck_response>year field is 2021</NodeCheck_response>

The paper was published in 2021.
</think>
Answer: 2021


Question: How many collaborators does Chao Zhang have in the DBLP graph?
<think>
Load the DBLP graph.
<LoadGraph>dblp</LoadGraph>
<LoadGraph_response>graph loaded</LoadGraph_response>

Check neighbouring authors.
<NeighbourCheck>AuthorNet, Chao Zhang</NeighbourCheck>
<NeighbourCheck_response>6 names returned</NeighbourCheck_response>

Chao Zhang has 6 collaborators.
</think>
Answer: 6


Question: How many papers do Chao Zhang and Weihong Lin have in common in the DBLP graph?
<think>
Load the DBLP graph.
<LoadGraph>dblp</LoadGraph>
<LoadGraph_response>graph loaded</LoadGraph_response>

Check shared papers.
<EdgeCheck>PaperNet, Chao Zhang, Weihong Lin</EdgeCheck>
<EdgeCheck_response>weight = 1</EdgeCheck_response>

They share 1 paper.
</think>
Answer: 1


Question: Where did Stephen's opera performance take place?
<think>
Retrieve the agenda entry.
<RetrieveAgenda>Stephen's Opera performance</RetrieveAgenda>
<RetrieveAgenda_response>description lists venue “Lyric Opera House”.</RetrieveAgenda_response>

The performance took place at the Lyric Opera House.
</think>
Answer: Lyric Opera House


Question: What was the trading volume of coffee on 2000-01-14?
<think>
Run SQL query.
<SQLInterpreter>SELECT Volume FROM coffee.coffee_data WHERE Date = '2000-01-14';</SQLInterpreter>
<SQLInterpreter_response>Volume = 10115</SQLInterpreter_response>

The trading volume was 10 115.
</think>
Answer: 10115
"""



TOOLQA_EASY8_PLANNER = r"""Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?

<think>
Need the departure and arrival delays, then subtract.
</think>
<search_query>Load flights DB</search_query>
<search_query_response>database schema loaded</search_query_response>
<think>
then i need to filter the flights_db for flight number DL1575 on 2022-01-12,
</think>
<search_query>
Filter flights_db for flight number DL1575 on 2022-01-12,
origin = ATL, destination = MCO 
</search_query>
<search_query_response>1 row selected</search_query_response>
<think>
let's get the departure delay from the flight_row
</think>
<search_query>Get DepDelay from flight_row</search_query>
<search_query_response>-7.0</search_query_response>
<think>
then i need to get the arrival delay from the flight_row
</think>
<search_query>Get ArrDelay from flight_row</search_query>
<search_query_response>-17.0</search_query_response>
<think>
Finally i can calculate the extra minutes as 7.0  - 17.0
</think>
<search_query>Calculate 7.0  - 17.0</search_query>
<search_query_response>-10</search_query_response>
<think>
i think that i have the answer now
</think>
<Finish>-10</Finish>"""



TOOLQA_EASY8_REASONING_EXECUTOR = r""" 

QUESTION 1 : Load flights DB in variable flights_db

<think>Let me load the flights database so I can query flight details.</think>
<LoadDB>flights</LoadDB>
<LoadDB_response>database schema loaded</LoadDB_response>
<think>Great, the flights table is available.</think>
<Finish>database schema loaded</Finish>


QUESTION 2 : Filter flights_db for flight number DL1575 on 2022-01-12, origin ATL, destination MCO in variable flight_row

<think>I must isolate the single row that matches the flight/date/route criteria. Also i have to extract the number from the flight number that is 1575</think>
<FilterDB>Flight_Number_Marketing_Airline=1575, FlightDate='2022-01-12', Origin='ATL', Dest='MCO'</FilterDB>
<FilterDB_response>1 row selected</FilterDB_response>
<think>Perfect—exactly one record returned.</think>
<Finish>1 row selected</Finish>



QUESTION 3 : Get DepDelay from flight_row in variable dep_delay

<think>Retrieve the departure delay so I know how late (or early) the plane left.</think>
<GetValue>DepDelay</GetValue>
<GetValue_response>-7.0</GetValue_response>
<think>Captured the departure delay.</think>
<Finish>-7.0</Finish>



QUESTION 4 : Get ArrDelay from flight_row in variable arr_delay

<think>Now grab the arrival delay for the same flight.</think>
<GetValue>ArrDelay</GetValue>
<GetValue_response>-17.0</GetValue_response>
<think>Arrival delay recorded.</think>
<Finish>-17.0</Finish>



QUESTION 5 : Calculate arr_delay − dep_delay in variable extra_minutes

<think>Compute the extra minutes the flight actually took beyond scheduled time.</think>
<Calculate>-17 - (-7)</Calculate>
<Calculate_response>-10</Calculate_response>
<think>Finished the calculation.</think>
<Finish>-10</Finish>
"""



TOOLQA_EASY8_PLANNER_FINAL = r"""Question: How many extra minutes did the AA1575 flight take from ATL to MCO on 2022-01-12?
<think>Need the departure and arrival delays, then subtract.</think>
<|begin_search_query|>Load flights DB<|end_search_query|>
<|begin_search_result|>database schema loaded<|end_search_result|>
<think>then i need to filter the flights DB for flight number 1575 on 2022-01-12</think>
<|begin_search_query|>Filter flights DB for flight number 1575 on 2022-01-12, origin = ATL, destination = MCO<|end_search_query|>
<|begin_search_result|>1 row selected<|end_search_result|>
<think>let's get the departure delay from the flights DB</think>
<|begin_search_query|>Get DepDelay from flights DB<|end_search_query|>
<|begin_search_result|>-7.0<|end_search_result|>
<think>then i need to get the arrival delay from the flights DB</think>
<|begin_search_query|>Get ArrDelay from flights DB<|end_search_query|>
<|begin_search_result|>-17.0<|end_search_result|>
<think>Finally i can calculate the extra minutes as 7.0 - 17.0</think>
<|begin_search_query|>Calculate 7.0  - 17.0<|end_search_query|>
<|begin_search_result|>-10<|end_search_result|>
<think>i think that i have the answer now</think>
<Finish>-10</Finish>

Question: How many extra minutes did the DL1573 flight take from ATL to MCO on 2021-01-12?
<think>Need the departure and arrival delays, then subtract.</think>
<|begin_search_query|>Load flights DB<|end_search_query|>
<|begin_search_result|>database schema loaded<|end_search_result|>
<think>then i need to filter the flights DB for flight number 1573 on 2021-01-12,</think>
<|begin_search_query|>Filter flights DB for flight number 1573 on 2021-01-12, origin = ATL, destination = MCO<|end_search_query|>
<|begin_search_result|>1 row selected<|end_search_result|>
<think>let's get the departure delay from the flights DB</think>
<|begin_search_query|>Get DepDelay from flights DB<|end_search_query|>
<|begin_search_result|>-4.0.<|end_search_result|>
<think>then i need to get the arrival delay from the flights DB</think>
<|begin_search_query|>Get ArrDelay from flights DB<|end_search_query|>
<|begin_search_result|>-12.0.<|end_search_result|>
<think>Finally i can calculate the extra minutes as value0 - value1</think>
<|begin_search_query|>Calculate (-4.0)-(-12) <|end_search_query|>
<|begin_search_result|>8<|end_search_result|>
<think>i think that i have the answer now </think>
<Finish>8</Finish>

Question: What is the corresponding Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task?
<think>I need to retrieve the score from the scientific-papers database</think>
<|begin_search_query|>Retrieve from Scirex the Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task<|end_search_query|>
<|begin_search_result|>71.8%<|end_search_result|>
<think>i think that i have the answer now</think>
<Finish>71.8%</Finish>

Question: When was the paper Learning the Principle of Least Action with Reinforcement Learning. published?
<think>Load DBLP graph to retrieve metadata.</think>
<|begin_search_query|>Load DBLP graph<|end_search_query|>
<|begin_search_result|>graph succesfully loaded<|end_search_result|>
<think>Ok the graph is loaded. now i have to search the paper</think>
<|begin_search_query|>search on the graph the paper "Learning the Principle of Least Action with Reinforcement Learning" in PaperNet<|end_search_query|>
<|begin_search_result|>{'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, …}]<|end_search_result|>
<think>i think that i have the answer now</think>
<Finish>2021</Finish>

Question: How many collaborators does Chao Zhang have in the DBLP graph?
<think>Load DBLP graph to retrieve metadata.</think>
<|begin_search_query|>Load DBLP graph<|end_search_query|>
<|begin_search_result|>graph succesfully loaded<|end_search_result|>
<think> Ok the graph is loaded. now i have to search the neighbors of Chao Zhang</think>
<|begin_search_query|>Find the author neighbors of Chao Zhang<|end_search_query|>
<|begin_search_result|>'YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang<|end_search_query|>
<think>Chao Zhang has 6 collaborators</think>
<Finish>6</Finish>

Question: How many papers does Chao Zhang and Weihong Lin have in common in the DBLP graph?
<think>Load DBLP graph to retrieve metadata.</think>
<|begin_search_query|>Load DBLP graph<|end_search_query|>
<|begin_search_result|>graph succesfully loaded<|end_search_result|>
<>think> Ok the graph is loaded. now i have to chack the edge between Chao Zhang and Weihong Lin in PaperNet</think>
<|begin_search_query|>Check the edge between Chao Zhang and Weihong Lin in PaperNet<|end_search_query|>
<|begin_search_result|>{'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}<|end_search_result|>
<think>They share 1 paper</think>
<Finish>1</Finish>

Question: Where did Stephen's Opera performance take place?
<think>I need to retrieve the agenda entry</think>
<|begin_search_query|>Retrieve from the agenda where did Stephen’s Opera performance take place<|end_search_query|>
<|begin_search_result|>On January 29, 2022 … at the Lyric Opera House, featuring Stephen.<|end_search_result|>
<think>The location is Lyric Opera House</think>
<Finish>Lyric Opera</Finish>
"""



TOOLQA_EASY8_REASONING_EXECUTOR_FINAL = r""" 

EXAMPLE 1 : Load flights DB in variable flights_db

<think>Let me load the flights database so I can query flight details.</think>
<ToolCall>{{
  "action": {{
    "name": "LoadDB",
    "reason": "I need to load the flights database",
    "input": {{
      "variables": {{}},
      "strings": {{
        "db_name": "flights"
      }}
    }}
  }}
}}</ToolCall>
<ToolCall_response>We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings</ToolCall_response>
<think>Great, the flights table is available.</think>
<ToolCall>{{
  "action": {{
    "name": "Finish",
    "reason": "return the result of loading",
    "input": {{
      "variables": {{}},
      "strings": {{
        "result": "We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings"
      }}
    }}
  }}
}}</ToolCall>


EXAMPLE 2 : Filter flights_db for flight number DL1575 on 2022-01-12, origin ATL, destination MCO in variable flight_row

<think>I must isolate the single row that matches the flight/date/route criteria. Also I have to extract the number 1575 from the flight number.</think>
<ToolCall>{{
  "action": {{
    "name": "FilterDB",
    "reason": "Filter the flights table to match the given criteria",
    "input": {{
      "variables": {{}},
      "strings": {{
        "query": "Flight_Number_Marketing_Airline=1575, FlightDate='2022-01-12', Origin='ATL', Dest='MCO'"
      }}
    }}
  }}
}}</ToolCall>
<ToolCall_response>1 row selected</ToolCall_response>
<think>Perfect—exactly one record returned.</think>
<ToolCall>{{
  "action": {{
    "name": "Finish",
    "reason": "return the result of filtering",
    "input": {{
      "variables": null,
      "strings": {{
        "result": "1 row selected"
      }}
    }}
  }}
}}</ToolCall>


EXAMPLE 3 : Get DepDelay from flight_row

<think>Retrieve the departure delay so I know how late (or early) the plane left.</think>
<ToolCall>{{
  "action": {{
    "name": "GetValue",
    "reason": "Extract the DepDelay value from the selected row",
    "input": {{
      "variables": {{}},
      "strings": {{
        "query": "DepDelay"
      }}
    }}
  }}
}}</ToolCall>
<ToolCall_response>-7.0. This result is stored in variable 'value0' with type <string>.</ToolCall_response>
<think>Captured the departure delay.</think>
<ToolCall>{{
  "action": {{
    "name": "Finish",
    "reason": "return the departure delay",
    "input": {{
      "variables": {{}},
      "strings": {{
        "result": "-7.0. This result is stored in variable 'value0'."
      }}
    }}
  }}
}}</ToolCall>


EXAMPLE 4 : Get ArrDelay from flight_row

<think>Now grab the arrival delay for the same flight.</think>
<ToolCall>{{
  "action": {{
    "name": "GetValue",
    "reason": "Extract the ArrDelay value from the selected row",
    "input": {{
      "variables": {{}},
      "strings": {{
        "query": "ArrDelay"
      }}
    }}
  }}
}}</ToolCall>
<ToolCall_response>-17.0. This result is stored in variable 'value1' with type <string>.</ToolCall_response>
<think>Arrival delay recorded.</think>
<ToolCall>{{
  "action": {{
    "name": "Finish",
    "reason": "return the arrival delay",
    "input": {{
      "variables": null,
      "strings": {{
        "result": "-17.0. This result is stored in variable 'value1'."
      }}
    }}
  }}
}}</ToolCall>


EXAMPLE 5 : Calculate -17 - (-7)
<think>Compute the extra minutes the flight actually took beyond scheduled time.</think>
<ToolCall>{{
  "action": {{
    "name": "Calculate",
    "reason": "Use a calculator tool for arithmetic",
    "input": {{
      "variables": {{}},
      "strings": {{
        "query": "-17 - (-7)"
      }}
    }}
  }}
}}</ToolCall>
<ToolCall_response>-10</ToolCall_response>
<think>Finished the calculation.</think>
<ToolCall>{{
  "action": {{
    "name": "Finish",
    "reason": "return the calculated result",
    "input": {{
      "variables": {{}},
      "strings": {{
        "result": "-10"
      }}
    }}
  }}
}}</ToolCall>


EXAMPLE 6 : Calculate value1 - value0
<think>Compute the extra minutes the flight actually took beyond scheduled time.</think>
<ToolCall>{{
  "action": {{
    "name": "PythonInterpreter",
    "reason": "Use a calculator tool for arithmetic",
    "input": {{
      "variables": {{
        "variables" : ["value0", "value1"]
        }},
      "strings": {{
        "code": "result = float(value1) - float(value0)\n"
      }}
    }}
  }}
}}</ToolCall>
<ToolCall_response>-10. This result is stored in variable 'result0'.</ToolCall_response>
<think>Finished the calculation.</think>
<ToolCall>{{
  "action": {{
    "name": "Finish",
    "reason": "return the calculated result",
    "input": {{
      "variables": {{}},
      "strings": {{
        "result": "-10. This result is stored in variable 'result0'."
      }}
    }}
  }}
}}</ToolCall>


"""

BASE_JSON = r"""
{
  "action": {
    "name": "ToolName",
    "reason": "Why you are choosing this tool",
    "input": {
      "variables": {...},        
      "strings": {...}        
    }
  }
}
"""

GRAPH = r"""
{}
"""


TOOLQA_HARD3_PLANNER= r"""Question 1: What is the average airtime of the flights from PHX to LGA host by Spirit Air Lines?
<think>
This is a question about flights. I need to load the flights database first.
</think>
<|begin_search_query|>Load flights DB<|end_search_query|>
<|begin_search_result|>database schema loaded<|end_search_result|>

<think>
Next, I should filter the database for flights with Origin=PHX, Dest=LGA, and Airline=Spirit Air Lines.
</think>
<|begin_search_query|>Filter flights DB for Origin=PHX, Dest=LGA, Airline=Spirit Air Lines<|end_search_query|>
<|begin_search_result|>16 rows selected<|end_search_result|>

<think>
Now I need to get the AirTime values from these rows.
</think>
<|begin_search_query|>Get AirTime values from flight DB<|end_search_query|>
<|begin_search_result|>[247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, nan, 229.0, nan, 259.0, 253.0]<|end_search_result|>

<think>
Finally, I can calculate the mean ignoring NaN values.
</think>
<|begin_search_query|>Calculate mean of [247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0]<|end_search_query|>
<|begin_search_result|>249.214<|end_search_result|>

<Finish>249.2</Finish>

Question 2: Can you recommend a Guamanian business with the highest star rating within a 5-mile radius of 6830 Rising Sun Ave?
<think>
This is a business-related question. I need to load the Yelp database first.
</think>
<|begin_search_query|>Load Yelp DB<|end_search_query|>
<|begin_search_result|>database schema loaded<|end_search_result|>

<think>
Next, I should get the latitude and longitude for 6830 Rising Sun Ave.
</think>
<|begin_search_query|>Get latitude and longitude for address '6830 Rising Sun Ave'<|end_search_query|>
<|begin_search_result|>latitude: 40.05555, longitude: -75.090723<|end_search_result|>

<think>
Now I calculate the coordinate range within a 5-mile radius.
</think>
<|begin_search_query|>Calculate bounding box 5-mile radius from (40.05555, -75.090723)<|end_search_query|>
<|begin_search_result|>(40.10058038657396, 40.010519262352226, -75.03212327463275, -75.14932272536724)<|end_search_result|>

<think>
Next, filter Yelp businesses in that bounding box with category=Guamanian.
</think>
<|begin_search_query|>Filter Yelp DB with latitude and longitude bounds, categories=Guamanian<|end_search_query|>
<|begin_search_result|>2 rows selected<|end_search_result|>

<think>
Now get the star ratings of the filtered rows.
</think>
<|begin_search_query|>Get stars from Yelp DB<|end_search_query|>
<|begin_search_result|>[4.0, 4.5]<|end_search_result|>

<think>
The maximum is 4.5, so select the corresponding business.
</think>
<|begin_search_query|>Filter flights DB with stars=4.5<|end_search_query|>
<|begin_search_result|>Damaso Painting<|end_search_result|>

<Finish>Damaso Painting</Finish>

Question 3: What venue does Chao Zhang with his collaborator publish paper on?
<think>
This is about DBLP. I need to load the DBLP graph.
</think>
<|begin_search_query|>Load DBLP graph<|end_search_query|>
<|begin_search_result|>DBLP graph loaded<|end_search_result|>

<think>
Now check collaborators (neighbors) of Chao Zhang in AuthorNet.
</think>
<|begin_search_query|>Get neighbors of Chao Zhang in AuthorNet<|end_search_query|>
<|begin_search_result|>Weihong Lin<|end_search_result|>

<think>
Check the edge between Chao Zhang and Weihong Lin to see their collaboration.
</think>
<|begin_search_query|>Get collaboration edge Chao Zhang - Weihong Lin<|end_search_query|>
<|begin_search_result|>{'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}<|end_search_result|>

<think>
Now I check the paper details in PaperNet.
</think>
<|begin_search_query|>Get node info for paper 'HRFormer: High-Resolution Vision Transformer for Dense Predict.'<|end_search_query|>
<|begin_search_result|>{'title': 'HRFormer: High-Resolution Vision Transformer for Dense Predict.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'doc_type': 'Conference'}<|end_search_result|>

<think>
So the venue is AAAI Spring Symposium - MLPS.
</think>
<Finish>AAAI Spring Symposium - MLPS</Finish>


"""

TOOLQA_EASY8_PROXY = """Question: Load flights DB
Thought 1: This requires the flights database.
Action 1: LoadDB[flights]
Observation 1: database schema loaded including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings
Thought 2: question solved let's return the result.
Action 2: Finish[database schema loaded, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings]


Question: Filter flights DB for flight number 1575 on 2022-01-12, origin = ATL, destination = MCO
Thought 1: Filter flight db to the target flight/date/route.
Action 1: FilterDB[Flight_Number_Operating_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]
Observation 1: 1 row selected
Thought 2: The filter result is ready, so I can return it.
Action 2: Finish[1 row]

Question: Get DepDelay from the flights DB
Thought 1: Retrieve the departure delay from the flights DB.
Action 1: GetValue[DepDelay]
Observation 1: -7.0
Thought 2: the departure delay is -7.0.
Action 2: Finish[-7.0]

Question: Get ArrDelay from flights DB
Thought 1: Retrieve the arrival delay from the flights DB.
Action 1: GetValue[ArrDelay]
Observation 1: -17.0
Thought 2: the arrival delay is -17.0.
Action 2: Finish[-17.0]

Question: Calculate 7.0 - 17.0
Thought 1: calculate the operation.
Action 1: Calculate[7.0 - 17.0]
Observation 1: -10
Thought 2: The calculation is complete, and the result is -10.
Action 2: Finish[-10]

Question: Calculate the number of strings divided by spaces in the variable value0
Thought 1: I need to write a python program to count the number of words in the string stored in variable value0.
Action 1: PythonInterpreter(value0)[print(len(value0.split(' ')))]
Observation 1: 2000
Thought 2: The calculation is complete, and the result is 2000.
Action 2: Finish[2000]

Question: Retrieve from Scirex the Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task
Thought 1: Query Scirex for the performance results.
Action 1: RetrieveScirex[Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task]
Observation 1: section : Abstract … our approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset.
Thought 2: The IoU score is 71.8%.
Action 2: Finish[71.8%]

Question: Load the DBLP graph
Thought 1: Load DBLP graph to retrieve metadata.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: Ok the graph is loaded.
Action 2: Finish[graph succesfully loaded]

Question: search on the graph the paper "Learning the Principle of Least Action with Reinforcement Learning" in PaperNet
Thought 1: Query the paper node.
Action 1: NodeCheck[PaperNet, Learning the Principle of Least Action with Reinforcement Learning.]
Observation 1: {'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, …}
Thought 2: Paper founded let's return it.
Action 2: Finish[{'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, …}]

Question: Check in the DBLP graph the neighbors of Chao Zhang
Thought 1: Check neighbors of Chao Zhang.
Action 1: NeighbourCheck[AuthorNet, Chao Zhang]
Observation 1: ['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']
Thought 2: Now i return the collaborators.
Action 2: Finish['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']

Question: Check the edge between Chao Zhang and Weihong Lin in PaperNet.
Thought 1: Check edge between the two authors.
Action 1: EdgeCheck[PaperNet, Chao Zhang, Weihong Lin]
Observation 1: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}
Thought 2: Weight indicates the shared paper.
Action 2: Finish['HRFormer: High-Resolution Vision Transformer for Dense Predict.']

Question: Retrieve from the agenda where did Stephen’s Opera performance take place
Thought 1: Retrieve event details from agenda.
Action 1: RetrieveAgenda[Stephen’s Opera performance]
Observation 1: On January 29, 2022 … at the Lyric Opera House, featuring Stephen.
Thought 2: The location is Lyric Opera House.
Action 2: Finish[Lyric Opera]
"""

TOOLQA_HARD_PROXY = """Question: Load flights DB
Thought 1: This requires the flights database.
Action 1: LoadDB[flights]
Observation 1: database schema loaded including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings
Thought 2: question solved let's return the result.
Action 2: Finish[database schema loaded, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings]


Question: Filter flights DB for Origin=PHX, Dest=LGA, Airline=Spirit Air Lines
Thought 1: I should filter by origin, destination, and airline.
Action 1: FilterDB[Origin=PHX, Dest=LGA, Airline=Spirit Air Lines]
Observation 1: 16 rows selected
Thought 2: The filter result is ready, so I can return it.
Action 2: Finish[16 rows selected]

Question: Get AirTime values from flights DB
Thought 1: I need the AirTime values.
Action 1: GetValue[AirTime]
Observation 1: [247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, nan, 229.0, nan, 259.0, 253.0]
Thought 2: I now return the list of AirTime values.
Action 2: Finish[247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0]

Question: Calculate mean of [247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0]
Thought 1: To answer the question, I need to calculate the mean of these values.
Action 1: Calculate[mean([247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0])]
Observation 1: 249.214
Thought 2: I now return the calculated mean.
Action 2: Finish[249.214]

Question: Load Yelp DB
Thought 1: I need to load the Yelp database.
Action 1: LoadDB[yelp]
Observation 1: yelp database schema loaded including the following columns: business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours
Thought 2: The database is ready, so I can return this result.
Action 2: Finish[database schema loaded with the following columns: business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours]

Question: Get latitude and longitude for address ‘6830 Rising Sun Ave’
Thought 1: I need the latitude and longitude of this address.
Action 1: SQLInterpreter[SELECT latitude, longitude FROM yelp.yelp_data WHERE address=‘6830 Rising Sun Ave’;]
Observation 1: latitude: 40.05555, longitude: -75.090723
Thought 2: I can return the coordinates.
Action 2: Finish[latitude: 40.05555, longitude: -75.090723]

Question: Calculate bounding box 5-mile radius from (40.05555, -75.090723)
Thought 1: I should calculate the bounding box around these coordinates.
Action 1: PythonInterpreter()[geopy distance calculations for 5 miles]
Observation 1: (40.10058038657396, 40.010519262352226, -75.03212327463275, -75.14932272536724)
Thought 2: I now return the bounding box.
Action 2: Finish[(40.10058038657396, 40.010519262352226, -75.03212327463275, -75.14932272536724)]

Question: Filter Yelp DB with latitude and longitude bounds, categories=Guamanian
Thought 1: I have to filter Yelp businesses by location and category.
Action 1: FilterDB[latitude>=40.010519262352226, longitude>=-75.14932272536724, latitude<=40.10058038657396, longitude<=-75.03212327463275, categories=Guamanian]
Observation 1: 2 rows selected
Thought 2: I return the filtered businesses.
Action 2: Finish[2 rows selected]

Question: Get stars from filtered yelp DB
Thought 1: I need to extract the star ratings.
Action 1: GetValue[stars]
Observation 1: [4.0, 4.5]
Thought 2: I return the list of ratings.
Action 2: Finish[4.0, 4.5]

Question: Filter Yelp DB with stars=4.5
Thought 1: I filter for the highest star rating.
Action 1: FilterDB[stars=4.5]
Observation 1: 1 row selected
Thought 2: I return the filtered row.
Action 2: Finish[1 row selected]

Question: Get name from yelp DB
Thought 1: Let's get the value name. 
Action 1: GetValue[name]
Observation 1: Damaso Painting
Thought 2: I return the business name.
Action 2: Finish[Damaso Painting]

Question: Calculate the number of strings divided by spaces in the variable value0
Thought 1: I need to write a python program to count the number of words in the string stored in variable value0.
Action 1: PythonInterpreter(value0)[print(len(value0.split(' ')))]
Observation 1: 2000
Thought 2: The calculation is complete, and the result is 2000.
Action 2: Finish[2000]

Question: Retrieve from Scirex the Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task
Thought 1: Query Scirex for the performance results.
Action 1: RetrieveScirex[Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task]
Observation 1: section : Abstract … our approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset.
Thought 2: The IoU score is 71.8%.
Action 2: Finish[71.8%]

Question: Load the DBLP graph
Thought 1: Load DBLP graph to retrieve metadata.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: Ok the graph is loaded.
Action 2: Finish[graph succesfully loaded]

Question: search on the graph the paper "Learning the Principle of Least Action with Reinforcement Learning" in PaperNet
Thought 1: Query the paper node.
Action 1: NodeCheck[PaperNet, Learning the Principle of Least Action with Reinforcement Learning.]
Observation 1: {'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, …}
Thought 2: Paper founded let's return it.
Action 2: Finish[{'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, …}]

Question: Check in the DBLP graph the neighbors of Chao Zhang
Thought 1: Check neighbors of Chao Zhang.
Action 1: NeighbourCheck[AuthorNet, Chao Zhang]
Observation 1: ['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']
Thought 2: Now i return the collaborators.
Action 2: Finish['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']

Question: Check the edge between Chao Zhang and Weihong Lin in PaperNet.
Thought 1: Check edge between the two authors.
Action 1: EdgeCheck[PaperNet, Chao Zhang, Weihong Lin]
Observation 1: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}
Thought 2: Weight indicates the shared paper.
Action 2: Finish['HRFormer: High-Resolution Vision Transformer for Dense Predict.']

Question: Retrieve from the agenda where did Stephen’s Opera performance take place
Thought 1: Retrieve event details from agenda.
Action 1: RetrieveAgenda[Stephen’s Opera performance]
Observation 1: On January 29, 2022 … at the Lyric Opera House, featuring Stephen.
Thought 2: The location is Lyric Opera House.
Action 2: Finish[Lyric Opera]"""