from pypdf import PdfReader
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup

def findNextEventStart(allLines, prevStart):
    i = prevStart
    while i < len(allLines):


        i+= 1

reader = PdfReader("C:/Users/ucg8nb\Downloads\Results.pdf")
allLines = []
for page in reader.pages:
    for l in page.extract_text().splitlines():
        allLines.append(l)

for i in range(0,len(allLines)):


    i+= 1

