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

hours_and_pay = pd.read_csv("C:/Users/ucg8nb/Downloads/hoursAndPay.csv")
# hoursAndPay['Percent Recieved'] = hoursAndPay['Sum of Payment Amount'] / hoursAndPay['Sum of Promise Amount']
# hoursAndPay['Cost of Work'] = hoursAndPay['Work time in hours'] * 40
# hoursAndPay['Net Profit'] = hoursAndPay['Sum of Payment Amount'] - hoursAndPay['Cost of Work']
# hoursAndPay['Profit Per Hour'] = hoursAndPay['Net Profit'] / hoursAndPay['Work time in hours']
# hoursAndPay.to_csv("C:/Users/ucg8nb/Downloads/hoursAndPay.csv")

profit_per_hour = hours_and_pay['Profit Per Hour'].tolist()

negative_profit = hours_and_pay[hours_and_pay['Profit Per Hour'] < 0]
positive_profit = hours_and_pay[hours_and_pay['Profit Per Hour'] > 0]

total_profit = np.sum(hours_and_pay['Net Profit'])
total_negative = np.sum(negative_profit['Net Profit'])
total_positive = np.sum(positive_profit['Net Profit'])
total_hours = np.sum(hours_and_pay['Work time in hours'])
negative_hours = np.sum(negative_profit['Work time in hours'])
positive_hours = np.sum(positive_profit['Work time in hours'])
print(total_profit, total_negative, total_positive)
print(total_profit / total_hours, total_negative / total_hours, total_positive / total_hours)
print(total_negative / negative_hours, total_positive / positive_hours)
print(total_hours)
