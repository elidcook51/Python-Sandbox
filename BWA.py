
startOil = 100000

def calcFlow(oil, cost):
    noPumpFlow = [0, oil]
    pumpFlow = [0.2 * oil * cost - 130000, 0.8 * oil]
    extraPumpFlow = [0.36 * oil * cost - 300000, 0.64 * oil]
    return


def calcNodeValue(oil, year):
    if year > 1:

        pass
    noPumpValue = calcNodeValue(oil, year + 1)
    pumpValue = calcNodeValue(oil * 0.8, year + 1)
    extraPumpValue = calcNodeValue(oil * 0.64, year + 1)
    if year == 0:
        cost = 20
    else:
        for cost in [20, 30]:
            noPumpFlow = 0
            pumpFlow = 0.2 * oil * cost - 130000
            extraPumpFlow = 0.36 * oil * cost - 300000
