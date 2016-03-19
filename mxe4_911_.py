#!/usr/bin/env python
# encoding=utf-8


import random
import os
import sys
import math



inpath = "d：//python_code//maxEntropy//me-0"
 
     
 
global feaDic; 
global feaParaDic; 
global feaEmp; 
global docList; 
global classList;classList=["business","auto","sport","it","yule"]
 
 
global maxIter;maxIter=20
######################

def loadData():
    global feaDic;feaDic={}
    global feaParaDic;feaParaDic={}
    global feaEmp;feaEmp={}
    global feaMod;feaMod={}
    global docList;docList=[]

    ###################build docList wordDic 

    for filename in os.listdir(inpath):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},c,0]
        content=open(inpath+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        #############
        for word in words:
            if len(word.strip())<=2:continue
            if word not in feaDic:
                feaDic[word]={};
                eachDoc[0][word]=1;
            elif word in feaDic:
                if word not in eachDoc[0]:eachDoc[0][word]=1
               # else:eachDoc[0][word]+=1   #DO NOT COUNT REPEATED WORD except for 'f0' commonFea
        docList.append(eachDoc)
    #########maxlen of each doc
    global maxlen;maxlen=-1
    for doc in docList:
        if maxlen<=len(doc[0]):
            maxlen=len(doc[0])
    maxlen+=1
    ##########make each doc the same length of words
    for doc in docList:
        doc[0]['fc']=maxlen-doc[0].__len__()
	
    ############ wid fc ->feaDic feaEmp feaMod feaParaDic
    feaDic['fc']={}
    for wid in feaDic.keys():
        if wid not in feaEmp:feaEmp[wid]={}
        if wid not in feaMod:feaMod[wid]={}
        if wid not in feaParaDic:feaParaDic[wid]={}
        for c in classList:
            if c not in feaEmp[wid]:feaEmp[wid][c]=0
            if c not in feaMod[wid]:feaMod[wid][c]=0
            if c not in feaParaDic[wid]:feaParaDic[wid][c]=0
    ############calc feaEmp
    for wid in feaDic:
        for c in classList:
            ######for each f(w,c)
            for doc in docList:
                if doc[1]==c and wid in doc[0]:
                    feaEmp[wid][c]+=doc[0][wid]
   
                
                    
     
 

  

    

def train():
	global feaDic; 
	global feaParaDic; 
	global feaEmp,feaMod
	global docList,maxlen

	ll=0.0
        ###feaMod back 0  VERY IMPORTANT
	for wid in feaDic.keys():
            for c in classList:
                feaMod[wid][c]=0
	########
	for doc in docList:
		pyx={}
		#####calc p(c|x)
		for c in classList:
			pyx[c]=0
			for wid in doc[0]:
				pyx[c]+=feaParaDic[wid][c]#*doc[0][wid]??
			pyx[c]=math.exp(pyx[c])
		####
		fenmu=sum(pyx.values())
		####normalize probability
		for c in classList:
			pyx[c]/=fenmu
		######loglikely
		for c in classList:
			if c==doc[1]:
				ll+=math.log(pyx[c])
		########E(f)mod
		"""
		for wid in feaDic:### in doc[0]  or in feaDic.keys()????
			for c in classList:
				###for each f(w,c)
				if c==doc[1] and wid in doc[0]:fi=1.
				else:fi=0.
				
				#wid may not in doc[0]
				if wid in doc[0]:
				feaMod[wid][c]+=pyx[c]*doc[0][wid]*fi
		"""
		for c in classList:
			for wid in doc[0]:### in doc[0]  or in feaDic.keys()????
			 
				###for each f(w,c)
				#if c==doc[1] and wid in doc[0]:fi=1.
				#else:fi=0.
				feaMod[wid][c]+=pyx[c]*doc[0][wid]#*fi
		
	#########
	for wid in feaDic:
		for c in classList:
			if feaEmp[wid][c]!=0 and feaMod[wid][c]!=0:
				feaParaDic[wid][c]+=math.log(feaEmp[wid][c]/feaMod[wid][c])  / maxlen


	####
	print 'll',ll
				
			
			

		
		
			
                 
            
 
        
    
    



#########################main
loadData()

for i in range(10):
    
    train()

        
    
    
    







    
    
