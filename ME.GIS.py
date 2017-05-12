  #! c:\python27\python
#Usage:
#Training: NB.py 1 TrainingDataFile ModelFile
#Testing: NB.py 0 TestDataFile ModelFile OutFile

import sys
import os
import math

TrainingDataFile = "newdata.train"
ModelFile = "newdata.model"
TestDataFile = "newdata.test"
TestOutFile = "newdata.out"
DocList = []
WordDic = {}
FeaClassTable = {}
FeaWeights = {}
ClassList = []
C = 100
MaxIteration = 1000
LogLLDiff = 0.1
CommonFeaID = 10000001

def Dedup(items):
	tempDic = {}
	for item in items:
		if item not in tempDic:
			tempDic[item] = True
	return tempDic.keys()

def LoadData():
	global CommonFeaID
	i =0
	infile = file(TrainingDataFile, 'r')
	sline = infile.readline().strip()
	maxwid = 0
	while len(sline) > 0:
		pos = sline.find("#")
		if pos > 0:
			sline = sline[:pos].strip()
		words = sline.split(' ')
		if len(words) < 1:
			print "Format error!"
			break
		classid = int(words[0])
		if classid not in ClassList:
			ClassList.append(classid)
		words = words[1:]
		#remove duplicate words, binary distribution
		words = Dedup(words)
		newDoc = {}
		for word in words:
			if len(word) < 1:
				continue
			wid = int(word)
			if wid > maxwid:
				maxwid = wid
			if wid not in WordDic:
				WordDic[wid] = 1
			if wid not in newDoc:
				newDoc[wid] = 1
		i += 1
		DocList.append((newDoc,classid))
		sline = infile.readline().strip()
	infile.close()
	print len(DocList), "instances loaded!"
	print len(ClassList), "classes!", len(WordDic), "words!"
	CommonFeaID = maxwid + 1
	print "Max wid:", maxwid
	WordDic[CommonFeaID] = 1

def ComputeFeaEmpDistribution():
	global C
	global FeaClassTable
	FeaClassTable = {}
	for wid in WordDic.keys():
		temppair = ({},{})
		FeaClassTable[wid] = temppair
	maxCount = 0  # count the max length doc among all doc
	for doc in DocList:
		if len(doc[0]) > maxCount:
			maxCount = len(doc[0])

	C = maxCount + 1
	for doc in DocList:
		doc[0][CommonFeaID] = C - len(doc[0])
		for wid in doc[0].keys():
			if doc[1] not in FeaClassTable[wid][0]:# doc[1] means classId
				FeaClassTable[wid][0][doc[1]] = doc[0][wid]
			else:
				FeaClassTable[wid][0][doc[1]] += doc[0][wid]
	return

def GIS():
	global C
	global FeaWeights
	for wid in WordDic.keys():
		FeaWeights[wid] = {}
		for classid in ClassList:
			FeaWeights[wid][classid] = 0.0 # with commonFeaId
	n = 0
	prelogllh = -1000000.0
	logllh = -10000.0
	while logllh - prelogllh >= LogLLDiff and n < MaxIteration:
		n += 1
		prelogllh = logllh
		logllh = 0.0
		print "Iteration",n
		##　each iteration for updating weight, sum over all samples docs ,need to re initialize model E(p)
		for wid in WordDic.keys():
			for classid in ClassList:
				FeaClassTable[wid][1][classid] = 0.0
		#compute expected values of features subject to the model p(y|x)        E(p)
		for doc in DocList:
			classProbs = [0.0]*len(ClassList)   #p(c1|x)  p(c2|x)...
			sum = 0.0
			for i in range(len(ClassList)):
				classid = ClassList[i]
				pyx = 0.0
				for wid in doc[0].keys():
					pyx += FeaWeights[wid][classid]  # weight x fi_count?  multiply doc[0][wid]??
				pyx = math.exp(pyx)
				classProbs[i] = pyx
				sum += pyx
			for i in range(len(ClassList)):
				classProbs[i] = classProbs[i] / sum
			for i in range(len(ClassList)):
				classid = ClassList[i]
				if classid == doc[1]:
					logllh += math.log(classProbs[i])    #MLE  likelyhood function
				for wid in doc[0].keys():
					FeaClassTable[wid][1][classid] += classProbs[i]*doc[0][wid]    # E(p) model
		#update feature weights              lambda
		for wid in WordDic.keys():
			for classid in ClassList:
				empValue = 0.0
				if classid in FeaClassTable[wid][0]:
					empValue = FeaClassTable[wid][0][classid]
				modelValue = 0.0
				if classid in FeaClassTable[wid][1]:
					modelValue = FeaClassTable[wid][1][classid]
				if empValue == 0.0 or modelValue == 0.0:
					continue
				FeaWeights[wid][classid] += math.log(FeaClassTable[wid][0][classid]/FeaClassTable[wid][1][classid]) / C
		print "Loglikelihood:",logllh
	return

def SaveModel():
	outfile = file(ModelFile, 'w')
	for wid in FeaWeights.keys():
		outfile.write(str(wid))
		outfile.write(' ')
		for classid in FeaWeights[wid]:
			outfile.write(str(classid))
			outfile.write(' ')
			outfile.write(str(FeaWeights[wid][classid]))
			outfile.write(' ' )
		outfile.write('\n')
	outfile.close()

def LoadModel():
	global ClassList
	global FeaWeights
	FeaWeights = {}
	infile = file(ModelFile, 'r')
	sline = infile.readline()
	while len(sline) > 0:
		sline = sline.strip()
		items = sline.split(' ')
		wid = int(items[0])
		FeaWeights[wid] = {}
		i = 1
		while i < len(items):
			classid = int(items[i])
			i += 1
			FeaWeights[wid][classid] = float(items[i])
			i += 1
		sline = infile.readline().strip()
	infile.close()
	ClassList = []
	for classidlist in FeaWeights.values():
		for classid in classidlist.keys():
			ClassList.append(classid)
		break
	print len(FeaWeights), "words!",len(ClassList),"classes!"

def Predict(doc):
	classid = ClassList[0]
	maxClass = classid
	sum = 0.0
	for wid in doc.keys():
		if wid in FeaWeights:
			sum += FeaWeights[wid][classid]
	max = sum
	i = 1
	while i < len(ClassList):
		sum = 0.0
		for wid in doc.keys():
			if wid in FeaWeights:
				sum += FeaWeights[wid][ClassList[i]]
		if sum > max:
			max = sum
			maxClass = ClassList[i]
		i += 1
	return maxClass

def Test():
	TrueLabelList = []
	PredLabelList = []
	i =0
	infile = file(TestDataFile, 'r')
	outfile = file(TestOutFile, 'w')
	sline = infile.readline().strip()
	scoreDic = {}
	iline = 0
	while len(sline) > 0:
		iline += 1
		if iline % 10 == 0:
			print iline," lines finished!\r",
		pos = sline.find("#")
		if pos > 0:
			sline = sline[:pos].strip()
		words = sline.split(' ')
		if len(words) < 1:
			print "Format error!"
			break
		classid = int(words[0])
		TrueLabelList.append(classid)
		words = words[1:]
		#remove duplicate words, binary distribution
		words = Dedup(words)
		maxScore = 0.0
		newdoc = {}
		for word in words:
			if len(word) < 1:
				continue
			wid = int(word)
			if wid not in newdoc:
				newdoc[wid] = 1
		maxClass = Predict(newdoc)
		PredLabelList.append(maxClass)
		sline = infile.readline().strip()
	infile.close()
	outfile.close()
	print len(PredLabelList),len(TrueLabelList)
	return TrueLabelList,PredLabelList

def Evaluate(TrueList, PredList):
	accuracy = 0
	i = 0
	while i < len(TrueList):
		if TrueList[i] == PredList[i]:
			accuracy += 1
		i += 1
	accuracy = (float)(accuracy)/(float)(len(TrueList))
	print "Accuracy:",accuracy

def CalPreRec(TrueList,PredList,classid):
	correctNum = 0
	allNum = 0
	predNum = 0
	i = 0
	while i < len(TrueList):
		if TrueList[i] == classid:
			allNum += 1
			if PredList[i] == TrueList[i]:
				correctNum += 1
		if PredList[i] == classid:
			predNum += 1
		i += 1
	return (float)(correctNum)/(float)(predNum),(float)(correctNum)/(float)(allNum)

#main framework

if len(sys.argv) < 4:
	print "Usage incorrect!"
elif sys.argv[1] == '1':
	print "start training:"
	TrainingDataFile = sys.argv[2]
	ModelFile = sys.argv[3]
	LoadData()
	ComputeFeaEmpDistribution()
	GIS()
	SaveModel()
elif sys.argv[1] == '0':
	print "start testing:"
	TestDataFile = sys.argv[2]
	ModelFile = sys.argv[3]
	TestOutFile = sys.argv[4]
	LoadModel()
	TList,PList = Test()
	i = 0
	outfile = file(TestOutFile, 'w')
	while i < len(TList):
		outfile.write(str(TList[i]))
		outfile.write(' ')
		outfile.write(str(PList[i]))
		outfile.write('\n')
		i += 1 
	outfile.close()
	Evaluate(TList,PList)
	for classid in ClassList:
		pre,rec = CalPreRec(TList, PList,classid)
		print "Precision and recall for Class",classid,":",pre,rec
else:
	print "Usage incorrect!"
