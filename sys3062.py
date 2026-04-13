import pandas as pd
import numpy as np
import random
import Stats_Functions as stat
import warnings
warnings.filterwarnings("ignore")


def createExponentialDistribution(ratePerHour):
    ratePerSecond = ratePerHour / 3600
    dist = distribution('EX', 0, 1 / ratePerSecond, 0, 0 ,0)
    return dist

def findOpenServer(servers):
    openServers = []
    for s in servers:
        if s.getOnOff:
            openServers.append(s)
    minServer = openServers[0]
    minQueue = minServer.getQueueLength()
    for s in openServers:
        if s.getQueueLength() < minQueue:
            minServer = s
            minQueue = s.getQueueLength()
    return minServer

def findClosedServer(servers):
    for i in range(len(servers)):
        if servers[i].getOnOff:
            return servers[i]
    return None

def getHour(universalTime):
    return int(universalTime / (60 * 60))

def checkQueueFull(servers):
    for s in servers:
        if not (s.getOnOff and s.getQueueLength() >= 2):
            return False
    return True

def getAverageQueueLen(servers):
    activeServers = 0
    queueLen = 0
    for s in servers:
        if s.getOnOff:
            activeServers += 1
            queueLen += s.getQueueLength()
    return queueLen, queueLen / activeServers

class Queue:
    def __init__(self):
        self.list = []

    def addCustomer(self, timeLeft):
        self.list.append(timeLeft)

    def removeCustomer(self, newTime):
        toReturn = self.list.pop(0)
        for i in range(len(self.list)):
            self.list[i] += newTime
        return toReturn

class Server:
    def __init__(self, distribution, on, sink, queue):
        self.on = on
        self.distribution = distribution
        self.queueLen = 0
        self.serving = False
        self.timeLeft = 0
        self.sink = sink
        self.waitTime = 0
        self.queue = queue

    def getQueueLength(self):
        return self.queueLen

    def getInSystem(self):
        if self.serving:
            return self.queueLen + 1
        else:
            return self.queueLen

    def getCustomer(self):
        if self.serving == False:
            self.serving = True
            self.timeLeft = self.distribution.getRealization()
            self.waitTime = self.timeLeft
        else:
            self.queueLen += 1
            self.queue.addCustomer(self.timeLeft)

    def update(self, timeChange):
        if self.on:
            if self.serving == True:
                self.timeLeft -= timeChange
                if self.timeLeft <= 0:
                    self.sink.getCustomer(self.waitTime)
                    if self.queueLen > 0:
                        self.queueLen -= 1
                        self.timeLeft = self.distribution.getRealization()
                        self.waitTime = self.timeLeft + self.queue.removeCustomer(self.timeLeft)
                    else:
                        self.serving = False
        if self.on == False and self.serving == True:
            self.timeLeft -= timeChange
            if self.timeLeft <= 0:
                self.sink.getCustomer(self.waitTime)
                if self.queueLen > 0:
                    self.queueLen -= 1
                    self.timeLeft = self.distribution.getRealization()
                    self.waitTime = self.timeLeft
                    self.waitTime = self.timeLeft + self.queue.removeCustomer(self.timeLeft)
                else:
                    self.serving = False
    def changeOnOff(self):
        if self.on:
            self.on = False
        else:
            self.on = True

    def getOnOff(self):
        return self.on


class Source:
    def __init__(self, distribution, outputNode):
        self.distribution = distribution
        self.timeLeft = self.distribution.getRealization()
        self.outputNode = outputNode

    def update(self, timePassed):
        self.timeLeft -= timePassed
        if self.timeLeft < 0:
            self.outputNode.recieveCustomer()
            self.timeLeft = self.distribution.getRealization()

    def updateDistribution(self, distribution):
        self.distribution = distribution

class TransferNode:
    def __init__(self, outputList):
        self.outputList = outputList

    def recieveCustomer(self):
        minOutput = self.outputList[0]
        minQueue = self.outputList[0].getQueueLength()
        for output in self.outputList:
            if output.getQueueLength() < minQueue and output.getOnOff:
                minOutput = output
                minQueue = output.getQueueLength()
        minOutput.getCustomer()

class Sink:
    def __init__(self):
        self.totCustomer = 0
        self.totTime = 0
        self.totWaitTime = 0

    def getCustomer(self, waitTime):
        self.totCustomer += 1
        self.totWaitTime += waitTime

    def update(self, timeChange):
        self.totTime += timeChange

    def getTotalCustomers(self):
        return self.totCustomer

    def getAverageCustomer(self):
        return self.totCustomer / self.totTime

    def getTotalWaitTime(self):
        return self.totWaitTime

class distribution:
    def __init__(self, name, alpha, beta, eta, etaL, etaU):
        self.type = name
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.etaL = etaL
        self.etaU = etaU

    def getRealization(self):
        seed = random.random()
        if self.type == 'LG':
            return stat.inverseLG(seed, self.alpha, self.beta)
        if self.type == 'LP':
            return stat.inverseLP(seed, self.alpha, self.beta)
        if self.type == "GB":
            return stat.inverseGB(seed, self.alpha, self.beta)
        if self.type == "RG":
            return stat.inverseRG(seed, self.alpha, self.beta)
        if self.type == "EX":
            return stat.inverseEX(seed, self.beta, self.eta)
        if self.type == "WB":
            return stat.inverseWB(seed, self.alpha, self.beta, self.eta)
        if self.type == "IW":
            return stat.inverseIW(seed, self.alpha, self.beta, self.eta)
        if self.type == "LW":
            return stat.inverseLW(seed, self.alpha, self.beta, self.eta)
        if self.type == "LL":
            return stat.inverseLL(seed, self.alpha, self.beta, self.eta)
        if self.type == "P1":
            return stat.inverseP1(seed, self.beta, self.etaL, self.etaU)
        if self.type == "P2":
            return stat.inverseP2(seed, self.beta, self.etaL, self.etaU)
        if self.type == "LRLG":
            return stat.inverseLR(seed, 'LG', self.alpha, self.beta, self.etaL, self.etaU)
        if self.type == "LRLP":
            return stat.inverseLR(seed, 'LP', self.alpha, self.beta, self.etaL, self.etaU)
        if self.type == 'LRGB':
            return stat.inverseLR(seed, 'GB', self.alpha, self.beta, self.etaL, self.etaU)
        if self.type == 'LRRG':
            return stat.inverseLR(seed, 'RG', self.alpha, self.beta, self.etaL, self.etaU)

def runSimulation():
    universalTime = 0
    endTime = 7 * 60 * 60
    arrivalRates = [40, 60, 60, 30, 20, 20, 20]
    arrivalDist = []
    for rate in arrivalRates:
        arrivalDist.append(createExponentialDistribution(rate))
    serviceDist = createExponentialDistribution(20)
    sink = Sink()
    servers = []
    totServersOn = 10
    for i in range(10):
        newQueue = Queue()
        servers.append(Server(serviceDist, True, sink, newQueue))
    for i in range(4, 10):
        servers[i].changeOnOff()
        totServersOn -= 1
    trans = TransferNode(servers)
    source = Source(arrivalDist[0], trans)
    outputDf = pd.DataFrame()
    laneOpenings = []
    laneClosings = []
    while universalTime < endTime-1:
        source.update(1)
        for s in servers:
            s.update(1)
        sink.update(1)
        universalTime += 1
        if universalTime % (60 * 60) == 0:
            source.updateDistribution(arrivalDist[int(universalTime / 3600)])
            #print(f"Hour {int(universalTime / 3600)} completed")
        queueLen, avgQueueLen = getAverageQueueLen(servers)
        if universalTime % 900 == 0:
            if avgQueueLen < 1 and totServersOn > 4:
                server = findOpenServer(servers)
                server.changeOnOff()
                totServersOn -= 1
                laneClosings.append(universalTime)
            if checkQueueFull(servers):
                server = findClosedServer(servers)
                server.changeOnOff()
                if totServersOn < 10:
                    totServersOn += 1
                    laneOpenings.append(universalTime)
        totInSystem = 0
        for s in servers:
            totInSystem += s.getInSystem()
        utilized = 1
        for s in servers:
            if s.getInSystem() < 1 and s.getOnOff():
                utilized = 0
        newRow = {
            'Time': universalTime,
            'Customers Served': sink.getTotalCustomers(),
            'Lanes Open': totServersOn,
            'Total Customers in Queue': queueLen,
            'Total Customers in System': totInSystem,
            'Total Wait Time': sink.getTotalWaitTime(),
            'Utilized': utilized
        }
        outputDf = outputDf._append(newRow, ignore_index = True)
    return outputDf, laneClosings, laneOpenings

def getCurState(df, time):
    tempDf = df[df['Time'] == time]
    custServed = tempDf['Customers Served'].values[0]
    laneOpen = tempDf['Lanes Open'].values[0]
    totCustInQueue = tempDf['Total Customers in Queue'].values[0]
    totCustInSystem = tempDf['Total Customers in System'].values[0]
    totWaitTime = tempDf['Total Wait Time'].values[0]
    return custServed, laneOpen, totCustInQueue, totCustInSystem, totWaitTime

def standardize(df, fullDf, time):
    custServed, laneOpen, totCustInQueue, totCustInSystem, totWaitTime = getCurState(fullDf, time)
    df['Customers Served'] = df['Customers Served'] - custServed
    df['Total Wait Time'] = df['Total Wait Time'] - totWaitTime
    return df

'''outputDf = pd.DataFrame()
for i in range(100):
    tempDf, laneClosings, laneOpenings = runSimulation()
    print(f"Run {i+1} completed")
    tempDf['Extra Lanes'] = tempDf['Lanes Open'] - 4
    before8 = tempDf[tempDf['Time'] < 14400]
    after8 = tempDf[tempDf['Time'] >= 14400]
    after8 = standardize(after8, tempDf, 14400)
    custServedBef8, laneOpenBef8, totCustInQueueBef8, totCustInSystemBef8, totWaitTimeBef8 = getCurState(before8, 14400-1)
    custServedAft8, laneOpenAf8, totCustInQueueAft8, totCustInSystemAf8, totWaitTimeAft8 = getCurState(after8, np.max(after8['Time']))
    laneClosingsBef8 = 0
    laneClosingsAft8 = 0
    laneOpeningsBef8 = 0
    laneOpeningsAft8 = 0
    for c in laneClosings:
        if c < 14400:
            laneClosingsBef8 += 1
        else:
            laneClosingsAft8 += 1
    for o in laneOpenings:
        if o < 14400:
            laneOpeningsBef8 += 1
        else:
            laneOpeningsAft8 += 1
    newRow = {
        'Average Lanes 4 - 8': np.mean(before8['Lanes Open']),
        'Average Lanes 8 - 11':np.mean(after8['Lanes Open']),
        'Waiting Customers 4 - 8': np.mean(before8['Total Customers in Queue']),
        'Waiting Customers 8 - 11': np.mean(after8['Total Customers in Queue']),
        'Average Wait Time 4 - 8': totWaitTimeBef8 / custServedBef8,
        'Average Wait Time 8 - 11': totWaitTimeAft8 / custServedAft8,
        'Lane Openings 4 - 8': laneOpeningsBef8,
        'Lane Openings 8 - 11': laneOpeningsAft8,
        'Lane Closings 4 - 8': laneClosingsBef8,
        'Lane Closings 8 - 11': laneClosingsAft8,
        'Average Extra Lanes 4 - 8': np.sum(before8['Extra Lanes']) / len(before8),
        'Average Extra Lanes 8 - 11': np.sum(before8['Extra Lanes']) / len(after8),
        'Utilization': np.mean(tempDf['Utilized']),
        'Average delay': np.max(tempDf['Total Wait Time']) / len(tempDf),
        'Average line length': np.mean(tempDf['Total Customers in Queue'])
    }
    outputDf = outputDf._append(newRow, ignore_index = True)
outputDf.to_csv(stat.getDownloadsTab() + '/1000 Runs good util results.csv')'''

# numCounts = [10, 50, 100, 1000]
# for counts in numCounts:
#     dist = createExponentialDistribution(20)
#     estimatesList = []
#     for i in range(counts):
#         estimatesList.append(dist.getRealization())
#     sortedEstList = sorted(estimatesList)
#     ratePerSecond = 20 / 3600
#     sortedEstCumDist = stat.exponentialDF(np.array(sortedEstList), 1/ ratePerSecond, 0).tolist()
#     print(stat.calcKStest(sortedEstCumDist))