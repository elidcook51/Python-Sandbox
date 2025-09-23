import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# callCenterPage1 = pd.read_excel("C:/Users/ucg8nb/Downloads/SYS6001__18_Call_Center_data.xls", sheet_name = 'RPCs')
# callCenterPage2 = pd.read_excel("C:/Users/ucg8nb/Downloads/SYS6001__18_Call_Center_data.xls", sheet_name = 'Time')

# callCenterPage2['Work time in hours'] = callCenterPage2['Work Time In Seconds'] / 3600
# relevantColumns = ['Dialer Id', 'Broken Promise Flag', 'Close Promise Flag', 'Open Promise Flag', 'Kept Promise Flag', 'Sum of Promise Amount', 'Sum of Payment Amount']
# payByID = callCenterPage1[relevantColumns].groupby(['Dialer Id']).sum()

# hoursAndPay = callCenterPage2.merge(payByID, how = 'left', on = 'Dialer Id')
# hoursAndPay.to_csv("C:/Users/ucg8nb/Downloads/hoursAndPay.csv")

hoursAndPay = pd.read_csv("C:/Users/ucg8nb/Downloads/hoursAndPay.csv")
# hoursAndPay['Percent Recieved'] = hoursAndPay['Sum of Payment Amount'] / hoursAndPay['Sum of Promise Amount']
# hoursAndPay['Cost of Work'] = hoursAndPay['Work time in hours'] * 40
# hoursAndPay['Net Profit'] = hoursAndPay['Sum of Payment Amount'] - hoursAndPay['Cost of Work']
hoursAndPay['Profit Per Hour'] = hoursAndPay['Net Profit'] / hoursAndPay['Work time in hours']
hoursAndPay.to_csv("C:/Users/ucg8nb/Downloads/hoursAndPay.csv")