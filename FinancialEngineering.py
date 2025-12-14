import numpy as np
from scipy.stats import norm
import pandas as pd
import re
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import statsmodels.api as sm
from functools import reduce

def printInLatexTable(listOfLists, colNames):
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|" + "c|" * len(colNames)  + "}")
    print("\\hline")
    print(" & ".join(colNames) + "\\\\")
    print("\\hline")
    for i in range(len(listOfLists[0])):
        row_items = []
        for l in listOfLists:
            toAdd = l[i]
            try:
                toAdd = float(toAdd)
                toAdd = np.round(toAdd, decimals=4)
            except ValueError:
                pass
            row_items.append(str(toAdd))
        outputString = " & ".join(row_items) + " \\\\"
        print(outputString)
    print("\\hline")
    print("\\end{tabular}")
    print('\\end{table}')

def d1(S, K, r, T, sigma):
    return (np.log(S/K) + (r + np.power(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma)- sigma * np.sqrt(T)

def blackscholes(S, K, r, T, sigma):
    d1 = (np.log(S/K) + (r + np.power(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    # Nd1 = np.array([norm.cdf(x) for x in list(d1)])
    # Nd2 = np.array([norm.cdf(x) for x in list(d2)])
    output = S * Nd1 - K * np.exp(-1 * r * T) * Nd2
    return output

llmsheet = "C:/Users/ucg8nb/Downloads/LLM Stock Prices.xlsx"
llmCols = ['Meta', "NVIDIA", 'Palantir']
bigDf = pd.read_excel("C:/Users/ucg8nb/Downloads/LLM Stock Information.xlsx")

mask = (bigDf['Date'] >= '2021-01-01')
bigDf = bigDf.loc[mask]

for c in llmCols:
    plt.scatter(bigDf[c + ' Earnings'], bigDf[c + ' Price'], label = c)
plt.plot([0,0], [0,800], linestyle = '--')
plt.ylim(0,800)
plt.title('Stock Price vs. Earnings per Share')
plt.xlabel("Earnings per Share")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig('C:/Users/ucg8nb/Downloads/Price v earning new.png')
plt.clf()

metaStock = pd.read_excel(llmsheet, sheet_name = "Meta")
NVIDIAStock = pd.read_excel(llmsheet, sheet_name= 'NVIDIA')
palantirStock = pd.read_excel(llmsheet, sheet_name= "Palantir")

metaStock['Date'] = pd.to_datetime(metaStock['Date'])
NVIDIAStock['Date'] = pd.to_datetime(NVIDIAStock['Date'])
palantirStock['Date'] = pd.to_datetime(palantirStock['Date'])

dfs = [metaStock, NVIDIAStock, palantirStock]
stockDf = reduce(lambda left, right: pd.merge(left, right, how = 'outer', on = 'Date'), dfs)

stockDf[[c + ' Price' for c in llmCols]].plot(ax = plt.gca())
plt.title("Price of AI companies over time")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.savefig('C:/Users/ucg8nb/Downloads/AI Stock Price.png')

# metaEarnings = pd.read_excel(llmsheet, sheet_name='Meta Earnings')
# NVIDIAEarnings = pd.read_excel(llmsheet, sheet_name= 'NVIDIA Earnings')
# palantirEarnings = pd.read_excel(llmsheet, sheet_name= 'Palantir Earnings')

# metaStock['Date'] = pd.to_datetime(metaStock['Date'])
# metaStock = metaStock.set_index('Date')['Meta Price']
# metaStock = metaStock.groupby(pd.Grouper(freq = 'QE')).mean()

# NVIDIAStock['Date'] = pd.to_datetime(NVIDIAStock['Date'])
# NVIDIAStock = NVIDIAStock.set_index('Date')['NVIDIA Price']
# NVIDIAStock = NVIDIAStock.groupby(pd.Grouper(freq = 'QE')).mean()

# palantirStock['Date'] = pd.to_datetime(palantirStock['Date'])
# palantirStock = palantirStock.set_index('Date')['Palantir Price']
# palantirStock = palantirStock.groupby(pd.Grouper(freq = 'QE')).mean()

# renameCols = {'Fiscal Quarter End': 'Date'}
# metaEarnings = metaEarnings.rename(columns = renameCols)
# NVIDIAEarnings = NVIDIAEarnings.rename(columns=renameCols)
# palantirEarnings = palantirEarnings.rename(columns=renameCols)

# dfs = [metaStock, NVIDIAStock, palantirStock, metaEarnings, NVIDIAEarnings, palantirEarnings]

# bigDf = reduce(lambda left, right: pd.merge(left, right, on = "Date", how = 'outer'), dfs)
# bigDf.to_excel('C:/Users/ucg8nb/Downloads/LLM Stock Information.xlsx')

# stockDf = pd.read_excel("C:/Users/ucg8nb/Downloads/Stock Prices.xlsx")
# earningsDf = pd.read_excel("C:/Users/ucg8nb/Downloads/Stock Prices.xlsx", sheet_name = 'Earnings')
# yahooEngageDf = pd.read_excel("C:/Users/ucg8nb/Downloads/Stock Prices.xlsx", sheet_name = 'Website Views')
# amazonEngageDf = pd.read_excel("C:/Users/ucg8nb/Downloads/Stock Prices.xlsx", sheet_name = 'Customer Accounts')

# stockMask = (stockDf['month'] >= '1998-01-01') & (stockDf['month'] <= '2003-01-01')
# stockDf = stockDf.loc[stockMask]
# earningsMask = (earningsDf['Year'] >= 1998) & (earningsDf['Year'] <= 2003)
# earningsDf = earningsDf.loc[earningsMask]
# stockCols = ['Cisco', 'Amazon', 'Yahoo', 'Pets']

# stockDf['month'] = pd.to_datetime(stockDf['month'])
# stockDf['Year'] = stockDf['month'].dt.year

# earningsDf['Year'] = pd.to_datetime(earningsDf['Year'])

# stockRenames = {}
# earningRenames = {}
# for c in stockCols:
#     stockRenames[c] = c + ' Price'
#     earningRenames[c] = c + ' Earnings'
# stockDf = stockDf.rename(columns = stockRenames)
# earningsDf = earningsDf.rename(columns = earningRenames)

# stockDf = stockDf[['Year'] + [c + " Price" for c in stockCols]]
# earningsDf = earningsDf[['Year'] + [c + " Earnings" for c in stockCols]]

# stockDf = stockDf.groupby('Year', as_index = False)[[c + " Price" for c in stockCols]].mean()
# stockDf['Year'] = pd.to_datetime(stockDf['Year'])

# bigDf = pd.merge(stockDf, earningsDf, how = 'outer', on = 'Year')

# for c in stockCols:
#     tempDf = bigDf[[c + ' Earnings', c + ' Price']].dropna()
#     if len(tempDf) > 1:
#         x = tempDf[c + ' Earnings']
#         y = tempDf[c + ' Price']
#         x = sm.add_constant(x)
#         result = sm.OLS(y, x).fit()
#         print(f"Results for {c}:")
#         print(result.summary())

# for c in stockCols:
#     plt.scatter(bigDf[c + " Earnings"], bigDf[c + ' Price'], label = c)

# plt.plot([0,0], [0,60], linestyle = '--')
# plt.ylim(0, 60)
# plt.legend()
# plt.title("Stock Price vs. Earnings for each year for companies")
# plt.xlabel("Earnings in $")
# plt.ylabel("Stock Price in $")
# plt.savefig('C:/Users/ucg8nb/Downloads/Price v earning old.png')

# fig, ax = plt.subplots(1,2)

# yahooEngageDf.set_index('Date')['Yahoo'].plot(ax = ax[0])
# amazonEngageDf.set_index('Year')['Amazon'].plot(ax = ax[1])
# fig.suptitle("User Engagement for Amazon and Yahoo")
# quarters = yahooEngageDf['Date'].tolist()

# ax[0].set_xlabel('Quarter')
# ax[0].set_ylabel('Website Views')
# ax[0].set_title('Yahoo')

# fmt = ScalarFormatter(useMathText=True)
# fmt.set_powerlimits((6, 6))        # fix the order of magnitude to 10^6
# ax[0].yaxis.set_major_formatter(fmt)
# ax[0].ticklabel_format(style='sci', axis='y', scilimits=(6, 6))  # same effect via helper
# ax[1].yaxis.set_major_formatter(fmt)
# ax[1].ticklabel_format(style = 'sci', axis = 'y', scilimits = (6,6))

# step =2
# tick_pos = list(range(0, len(yahooEngageDf), step))
# ax[0].set_xticks(tick_pos)
# ax[0].set_xticklabels([quarters[i] for i in tick_pos], rotation = 30)

# ax[1].set_xlabel("Year")
# ax[1].set_ylabel("Customer Accounts")
# ax[1].set_title("Amazon")

# fig.tight_layout()
# fig.savefig("C:/Users/ucg8nb/Downloads/Tester Engagement.png")

# stockDf['month'] = pd.to_datetime(stockDf['month'])

# mask = (stockDf['month'] >= '1998-01-01') & (stockDf['month'] <= '2003-01-01')
# bubbleDf = stockDf.loc[mask].sort_values('month')

# stockCols = ['Cisco', 'Amazon', 'Yahoo', 'Pets']
# normalizedStocks = bubbleDf.copy()
# max_per_stock = normalizedStocks[stockCols].max()
# normalizedStocks[stockCols] = normalizedStocks[stockCols] / max_per_stock

# normalizedStocks.set_index('month')[stockCols].plot(ax = plt.gca())


# plt.title('Stock Prices by Company from 1998 to 2003')
# plt.xlabel("Date")
# plt.ylabel("Normalized Stock Price")
# plt.savefig('C:/Users/ucg8nb/Downloads/Stock Price Chart.png')


# yahooDf = pd.read_csv("C:/Users/ucg8nb/Downloads/Yahoo stock price.csv")
# petsDf = pd.read_csv("C:/Users/ucg8nb/Downloads/Pets dot com stock price.csv")
# ciscoDf = pd.read_excel("C:/Users/ucg8nb/Downloads/Stock Prices Cisco and Amazon.xlsx", sheet_name = 'Cisco')
# amazonDf = pd.read_excel("C:/Users/ucg8nb/Downloads/Stock Prices Cisco and Amazon.xlsx", sheet_name = 'Amazon')

# yahooDf['date'] = pd.to_datetime(yahooDf['date'])
# petsDf['date'] = pd.to_datetime(petsDf['date'])
# ciscoDf['date'] = pd.to_datetime(ciscoDf['Date'])
# amazonDf['date'] = pd.to_datetime(amazonDf['Date'])

# ciscoDf = ciscoDf.rename(columns = lambda c: c.strip())
# amazonDf = amazonDf.rename(columns = lambda c: c.strip())

# ciscoDf = ciscoDf.rename(columns = {'Adj Close': 'Cisco'})
# amazonDf = amazonDf.rename(columns = {'Adj Close': 'Amazon'})
# yahooDf = yahooDf.rename(columns = {'value': 'Yahoo'})
# petsDf = petsDf.rename(columns = {'value': 'Pets'})


# ciscoDf = (
#     ciscoDf.assign(month = ciscoDf['date'].dt.to_period('M')).groupby('month', as_index = False)['Cisco'].mean()
# )
# amazonDf = (
#     amazonDf.assign(month = amazonDf['date'].dt.to_period('M')).groupby('month', as_index = False)['Amazon'].mean()
# )
# yahooDf = (
#     yahooDf.assign(month = yahooDf['date'].dt.to_period('M')).groupby('month', as_index = False)['Yahoo'].mean()
# )
# petsDf = (
#     petsDf.assign(month = petsDf['date'].dt.to_period('M')).groupby('month', as_index = False)['Pets'].mean()
# )

# bigDf = pd.merge(ciscoDf, amazonDf,how ='outer', on = 'month')
# bigDf = pd.merge(bigDf, yahooDf, how = 'outer', on = 'month')
# bigDf = pd.merge(bigDf, petsDf, how = 'outer', on = 'month')

# bigDf.to_csv("C:/Users/ucg8nb/Downloads/Stock Prices.csv")


# petsurl = 'https://companiesmarketcap.com/pets-dot-com-ipet-holdings/stock-price-history/'
# amazonurl = 'https://companiesmarketcap.com/amazon/stock-price-history/'
# yahoourl = 'https://companiesmarketcap.com/yahoo/stock-price-history/'
# response = requests.get(yahoourl)
# response.raise_for_status()


# soup = BeautifulSoup(response.text, 'html.parser')

# script_tags = soup.find_all('script')

# data_list = None

# for tag in script_tags:
#     script = tag.string or tag.get_text()
    
#     if script and "data = " in script:
#         m = re.search(r"data\s*=\s*(\[\s*\{.*?\}\s*\])", script, flags = re.DOTALL)
#         if m:
#             json_text = m.group(1)

#             data_list = json.loads(json_text)
#             break

# if data_list is None:
#     raise ValueError("Could not locate 'data = [...]' in any script tag")

# records = [
#     {
#         "date": datetime.fromtimestamp(item['d'], tz = timezone.utc),
#         'value': item['v']
#     }
#     for item in data_list
# ]

# print(len(records))

# pd.DataFrame(records).to_csv("C:/Users/ucg8nb/Downloads/Yahoo stock price.csv")