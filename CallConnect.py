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
# hours_and_pay = hours_and_pay[hours_and_pay['Work time in hours'] > 50]
# hoursAndPay['Percent Recieved'] = hoursAndPay['Sum of Payment Amount'] / hoursAndPay['Sum of Promise Amount']
# hoursAndPay['Cost of Work'] = hoursAndPay['Work time in hours'] * 40
# hoursAndPay['Net Profit'] = hoursAndPay['Sum of Payment Amount'] - hoursAndPay['Cost of Work']
# hoursAndPay['Profit Per Hour'] = hoursAndPay['Net Profit'] / hoursAndPay['Work time in hours']
# hoursAndPay.to_csv("C:/Users/ucg8nb/Downloads/hoursAndPay.csv")

profit_per_hour = hours_and_pay['Profit Per Hour'].tolist()

negative_profit = hours_and_pay[hours_and_pay['Profit Per Hour'] < 0]
positive_profit = hours_and_pay[hours_and_pay['Profit Per Hour'] > 0]

total_profit = np.sum(hours_and_pay['Net Profit'])
goal_profit = total_profit * 1.4
total_negative = np.sum(negative_profit['Net Profit'])
total_positive = np.sum(positive_profit['Net Profit'])
total_hours = np.sum(hours_and_pay['Work time in hours'])
negative_hours = np.sum(negative_profit['Work time in hours'])
positive_hours = np.sum(positive_profit['Work time in hours'])

profit_per_hour_required = goal_profit / total_hours
profitable_dataframe = hours_and_pay[hours_and_pay['Profit Per Hour'] > profit_per_hour_required]


print(total_negative)
print(np.sum(profitable_dataframe['Net Profit']))

# print(total_profit, total_negative, total_positive)
# print(total_profit / total_hours, total_negative / total_hours, total_positive / total_hours)
# print(total_negative / negative_hours, total_positive / positive_hours)
# print(total_hours)

# rpc_neg = np.mean(negative_profit['Right Party Contact Count'] / negative_profit['Call Count'])
# rpc_pos = np.mean(positive_profit['Right Party Contact Count'] / positive_profit['Call Count'])

# ptp_neg = np.mean(negative_profit['Promise To Pay Count'] / negative_profit['Right Party Contact Count'])
# ptp_pos = np.mean(positive_profit['Promise To Pay Count'] / positive_profit['Right Party Contact Count'])

# kp_neg = np.mean(negative_profit['Kept Promise Flag'] / negative_profit['Promise To Pay Count'])
# kp_pos = np.mean(positive_profit['Kept Promise Flag'] / positive_profit['Promise To Pay Count'])

# print(rpc_neg, rpc_pos)
# print(ptp_neg, ptp_pos)
# print(kp_neg, kp_pos)
