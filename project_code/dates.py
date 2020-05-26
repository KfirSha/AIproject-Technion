import csv
import pandas as pd
import numpy as np
import os

class Dates:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def make_table_with_coordinates_and_dates(self):
        dayPerMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        with open(self.coordinates, 'r', encoding="utf8") as csv_file1:
            reader = csv.reader(csv_file1, delimiter="\n")
            list2008 = []
            list2009 = []
            list2010 = []
            list2011 = []
            list2012 = []
            list2013 = []
            list2014 = []
            list2015 = []
            list2016 = []
            list2017 = []

            first_row = True
            for row in reader:
                if first_row:
                    first_row = False
                    continue
                for month in range(1, 13):
                    for day in range(1, dayPerMonth[month - 1] + 1):
                        list2008.append((row[0] + "," + "2008" + "," + str(month) + "," + str(day)).split(','))
                        list2009.append((row[0] + "," + "2009" + "," + str(month) + "," + str(day)).split(','))
                        list2010.append((row[0] + "," + "2010" + "," + str(month) + "," + str(day)).split(','))
                        list2011.append((row[0] + "," + "2011" + "," + str(month) + "," + str(day)).split(','))
                        list2012.append((row[0] + "," + "2012" + "," + str(month) + "," + str(day)).split(','))
                        list2013.append((row[0] + "," + "2013" + "," + str(month) + "," + str(day)).split(','))
                        list2014.append((row[0] + "," + "2014" + "," + str(month) + "," + str(day)).split(','))
                        list2015.append((row[0] + "," + "2015" + "," + str(month) + "," + str(day)).split(','))
                        list2016.append((row[0] + "," + "2016" + "," + str(month) + "," + str(day)).split(','))
                        list2017.append((row[0] + "," + "2017" + "," + str(month) + "," + str(day)).split(','))

        List_of_list = [list2008, list2009, list2010, list2011, list2012, list2013, list2014, list2015, list2016,
                        list2017]
        with open('dates_and_coordinates_tmp.csv', 'w', encoding="utf8") as temp_file:
            firstLine = ['x', 'y', 'year', 'month', 'day']
            writer = csv.writer(temp_file, delimiter=',')
            writer.writerow(firstLine)
            for i in range(2008, 2018):
                writer = csv.writer(temp_file, delimiter=',')
                for line in List_of_list[i - 2008]:
                    writer.writerow(line)

        with open('dates_and_coordinates_tmp.csv') as in_file:
            with open('dates_and_coordinates.csv', 'w', newline='') as out_file:
                writer = csv.writer(out_file)
                for row in csv.reader(in_file):
                    if any(row):
                        writer.writerow(row)

        fields = ('day','month','year')
        dates = pd.read_csv("dates_and_coordinates.csv")
        print("dates_And_coirdinates size")
        print(len(dates.index))
        holidays = pd.read_csv('holiday.csv')
        merge_file = dates.merge(holidays, on=fields,how="left")
        merge_file['sug_yom'].replace(np.nan, 0, inplace=True)
        print("size after merge with holidays")
        merge_file = merge_file.drop_duplicates()
        print("rows in date and coordiantes")
        print(len(merge_file.index))
        merge_file.to_csv('dates_and_coordinates.csv', index=False)
        print("finish dates_and_coordinates")
        os.remove('dates_and_coordinates_tmp.csv')






