# from pypdf import PdfReader
import pandas as pd
import os
# import requests
# from bs4 import BeautifulSoup

# def findNextEventStart(allLines, prevStart):
#     i = prevStart
#     while i < len(allLines):


#         i+= 1

# reader = PdfReader("C:/Users/ucg8nb\Downloads\Results.pdf")
# allLines = []
# for page in reader.pages:
#     for l in page.extract_text().splitlines():
#         allLines.append(l)

# for i in range(0,len(allLines)):


#     i+= 1

vcuSignups = pd.read_csv("C:/Users/ucg8nb/Downloads/VCU Signups - Signups.csv")
vcuSignups['Gender'] = vcuSignups['Gender'].str.upper()
firstYearGirls = vcuSignups[vcuSignups['Gender'] == 'F']
firstYearGirls = firstYearGirls[firstYearGirls['Year'].str.contains('1')]
firstYearGirls.to_csv("C:/Users/ucg8nb/Downloads/fyg.csv")