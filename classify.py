import math
import numpy as np
#Binary only (tests and quality)
def disorder(Set):
	T = len(Set)
	P = sum(Set)
	N = T - P
	if P==0 or N==0 or T==0:
		return 0
	PoT, NoT = np.true_divide(P,T), np.true_divide(N,T)
	return - PoT*np.log2(PoT) - NoT*np.log2(NoT)

def isHomogeneous(Set):
	if len(Set) == 0:
		return true
	if all(Set) == Set[0]:
		return true:
	else:
		return False

def classifyDataBin(Data, Test):
	approved, rejected = [], []
	for i in Data:
		if Test(i):
			approved.append(i)
		else:
			rejected.append(i)
	return approved, rejected

def classifyData(Data, Test):
	results = {}
	for i in Data:
		result = Test(i)
		if result in results:
			results[result].append(i)
		else:
			results[result] = [i]
	return results.values()

def evaluateTestDisorder(Data, Test):
	results = classifyData(Data, Test)
	return sum([disorder(results[result]) for result in results])

def evaluateTest(Data, Test):
	results = classifyData(Data, Test)
	numHomogeneous = [len(result) for result in results if isHomogeneous(result)]
	return numHomogeneous

def makeTree(Data, Tests):
	sortedTests = sorted(Tests, key=lambda Test: evaluateTest(Test), reverse=True)
	Test = sortedTests[0]
	results = classifyData(Data, Test)
	for result in results:
		if not isHomogeneous(result):
			newData += result
	newData = [result for result in results if not isHomogeneous(result)]



data = [True, False, True, True]




