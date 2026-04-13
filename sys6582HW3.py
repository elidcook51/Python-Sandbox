import numpy as np
import copy

class DirectionalNetwork():

    #Creates a network stored a list of arcs and nodes (maybe not the best way...)

    def __init__(self, nodesList, arcList):
        self.nodes = nodesList

        #Arcs of the form (out, in, weight)
        self.arcs = [arc for arc in arcList if len(arc) == 3]
        self.arcs += [arc + (1,) for arc in arcList if len(arc) == 2]

    def __str__(self):
        outputString = ""
        for node in self.nodes:
            outTo = self.getOutCons(node)
            outputString += f"{node} -> "
            for out in outTo:
                outputString += f"{out} ({self.getArcWeight(node, out)}), "
            outputString += '\n'
        return outputString

    def add_node(self, node):
        self.nodes.append(node)
    
    def add_node_list(self, nodeList):
        for node in nodeList:
            self.add_node(node)

    def add_arc(self, arc):
        if len(arc) == 2:
            arc = arc + (1,)
        self.arcs.append(arc)

    def add_arc_list(self, arcList):
        for arc in arcList:
            self.add_arc(arc)

    def getArcWeight(self, outnode, innode):
        if innode not in self.nodes or outnode not in self.nodes:
            return -1
        for arc in self.arcs:
            if arc[0] == outnode and arc[1] == innode:
                return arc[2]
    
    def getInCons(self, node):
        if node not in self.nodes:
            return []
        outputList = []
        for arc in self.arcs:
            if arc[1] == node:
                outputList.append(arc[0])
        return outputList

    def getOutCons(self, node):
        if node not in self.nodes:
            return []
        outputList = []
        for arc in self.arcs:
            if arc[0] == node:
                outputList.append(arc[1])
        return outputList
    
    def updateNodes(self, nextNode, distanceSet, predSet):
        nextNodeDistance = distanceSet[self.nodes.index(nextNode)]
        for i in self.getOutCons(nextNode):
            weight = self.getArcWeight(nextNode, i)
            newDistance = weight + nextNodeDistance
            curDistance = distanceSet[self.nodes.index(i)]
            if newDistance < curDistance:
                distanceSet[self.nodes.index(i)] = newDistance
                predSet[self.nodes.index(i)] = nextNode
        return distanceSet, predSet

    def dijkstraAlg(self, startNode):
        if startNode not in self.nodes:
            return None
    
        permSet = set()
        tempSet = set(copy.deepcopy(self.nodes))
        distanceSet = [np.inf] * len(tempSet)
        predSet = [-1] * len(tempSet)

        distanceSet[self.nodes.index(startNode)] = 0
        predSet[self.nodes.index(startNode)] == 0

        permSet.add(startNode)
        tempSet.remove(startNode)

        distanceSet, predSet = self.updateNodes(startNode, distanceSet, predSet)

        while len(permSet) < len(self.nodes):
            curMinDistance = np.inf
            curNextNode = None
            for node in tempSet:
                if distanceSet[self.nodes.index(node)] <= curMinDistance:
                    curMinDistance = distanceSet[self.nodes.index(node)]
                    curNextNode = node
                
            distanceSet, predSet = self.updateNodes(curNextNode, distanceSet, predSet)

            permSet.add(curNextNode)
            tempSet.remove(curNextNode)
        
        return distanceSet, predSet
    
    def findShortestPath(self, startNode, endNode):
        distanceSet, predSet = self.dijkstraAlg(startNode)
        backwardsNodes = []
        curNode = endNode
        while curNode != startNode:
            backwardsNodes.append(curNode)
            curNode = predSet[self.nodes.index(curNode)]
        backwardsNodes.append(startNode)
        backwardsNodes.reverse()
        return backwardsNodes, distanceSet[self.nodes.index(endNode)]
    
newNetwork = DirectionalNetwork([], [])
newNetwork.add_node_list([1,2,3,4,5,6])
newNetwork.add_arc_list([(1,2,2), (1,3,8), (2,4,3), (2,3,5), (3,2,6), (3,5,0), (4,5,7), (4,3,1), (4,6,6), (5,4,4), (5,6,2)])
# print(newNetwork.findShortestPath(1,6))

class UndirectedNetwork():

    def __init__(self, nodeList, arcList):
        self.nodes = nodeList
        
        #Arcs of the form (node1, node2, weight)
        self.arcs = [arc for arc in arcList if len(arc) == 3]
        self.arcs += [arc + (1,) for arc in arcList if len(arc) == 2]
    
    def __str__(self):
        outputString = ""
        for node in self.nodes:
            outputString += f"{node} connected to "
            for connection in self.get_connections(node):
                outputString += f"{connection} ({self.get_arc_weight(node, connection)}), "
            outputString = outputString[:-2]
            outputString += '\n'
        return outputString

    def add_node(self, node):
        self.nodes.append(node)
    
    def _check_arc(self, arc):
        node1, node2, _ = arc
        for otherArc in self.arcs:
            if otherArc[0] == node1 and otherArc[1] == node2:
                return True
            elif otherArc[1] == node1 and otherArc[0] == node2:
                return True
        return False
    
    def add_arc(self, arc):
        if len(arc) == 2:
            arc += (1,)
        if not self._check_arc(arc):
            self.arcs.append(arc)
    
    def add_node_list(self, nodeList):
        for node in nodeList:
            self.add_node(node)
    
    def add_arc_list(self, arcList):
        for arc in arcList:
            self.add_arc(arc)

    def get_arc_from_nodes(self, node1, node2):
        for arc in self.arcs:
            if arc[0] == node1 and arc[1] == node2:
                return arc
            elif arc[1] == node1 and arc[0] == node2:
                return arc

    def remove_arc(self, arc):
        arc = self.get_arc_from_nodes(arc[0], arc[1])

    def remove_node(self, node):
        self.nodes.remove(node)
        for arc in self.arcs:
            if arc[0] == node or arc[1] == node:
                self.remove_arc(arc)

    def get_connections(self, node):
        outputList = []
        for arc in self.arcs:
            if arc[0] == node:
                outputList.append(arc[1])
            elif arc[1] == node:
                outputList.append(arc[0])
        return outputList
    
    def get_arc_weight(self, node1, node2):
        for arc in self.arcs:
            if arc[0] == node1 and arc[1] == node2:
                return arc[2]
            elif arc[1] == node1 and arc[0] == node2:
                return arc[2]

    def check_cycle(self, currentNode, parentNodes, visitedNodes):
        visitedNodes[self.nodes.index(currentNode)] = True
        connections = self.get_connections(currentNode)
        for node in connections:
            if not visitedNodes[self.nodes.index(node)]:
                parentNodes[self.nodes.index(node)] = currentNode
                if self.check_cycle(node, parentNodes, visitedNodes):
                    return True
            
            elif parentNodes[self.nodes.index(currentNode)] != node:
                return True
        
        return False

    def DFS_for_cycle(self):
        visitedNodes = [False] * len(self.nodes)
        parentNodes = [-1] * len(self.nodes)

        for node in self.nodes:
            if not visitedNodes[self.nodes.index(node)]:
                if self.check_cycle(node, parentNodes, visitedNodes):
                    return True
        return False
    
    def Kruskal(self, min = True):
        if min:
            orderedLinks = sorted(copy.deepcopy(self.arcs), key = lambda tup: tup[2])
        else:
            orderedLinks = sorted(copy.deepcopy(self.arcs), key = lambda tup: tup[2], reverse= True)
        outputList = []
        tempNetwork = UndirectedNetwork(self.nodes, [])
        for arc in orderedLinks:
            tempNetwork.add_arc(arc)
            if tempNetwork.DFS_for_cycle():
                tempNetwork.remove_arc(arc)
            else:
                outputList.append(arc)
        return outputList
    
    def find_min_cut(self, S, Sprime, min = True):
        possibleArcs = []
        for node in S:
            connections = self.get_connections(node)
            for conn in connections:
                if conn in Sprime:
                    possibleArcs.append(self.get_arc_from_nodes(node, conn))
        if min:
            return sorted(possibleArcs, key = lambda tup: tup[2])[0]
        else:
            return sorted(possibleArcs, key = lambda tup: tup[2], reverse = True)[0]

    def Prim(self, min = True):
        S = [self.nodes[0]]
        Sprime = self.nodes[1:]
        outputList = []
        while len(Sprime) > 0:
            minCutArc = self.find_min_cut(S, Sprime, min)
            outputList.append(minCutArc)
            if minCutArc[0] in S:
                S.append(minCutArc[1])
                Sprime.remove(minCutArc[1])
            else:
                S.append(minCutArc[0])
                Sprime.remove(minCutArc[0])
        return outputList

problem3Network = UndirectedNetwork([], [])
problem3Network.add_node_list([1,2,3,4,5,6,7,8])
problem3Network.add_arc_list([(1,2,9), (1,3,10), (1,4,12), (2,5,15), (2,6,4), (3,5,5), (3,6,12), (3,7,14), (4,6,8), (4,7,16), (5,8,17), (6,8,11), (7,8,7)])
# problem3Network.add_arc_list([(1,2), (2,3), (3,4), (4,1)])
# print(problem3Network.DFS_for_cycle())
print(problem3Network.Kruskal(min = False))