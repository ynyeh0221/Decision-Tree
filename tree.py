import csv
import sys
import numpy as np
import copy
import collections
from collections import Counter
import random
import math

attr=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide","density", "pH", "sulphates", "alcohol"]
training=[]

# Implement your decision tree below
class DecisionTree():
	def get_entr(self, llist):
		n=llist.size
		count={}
		for curlabel in llist:
			if curlabel not in count.keys():
				count[curlabel]=0
			count[curlabel]=count[curlabel]+1
		ss=0
		for i in range (0, len(llist)):
			ss=ss+llist[i]
		entropy=0
		if (ss!=0):
		    for key in count:
			    pxi=float(llist[i]/ss)
			    if(pxi!=0):
			        entropy-=pxi*math.log(pxi,2)
		return entropy		

	def get_id(self, j, llist):		
		for i in range(0, len(llist)):
			if j==llist[i]:
				return i
		return -1

	def ig_calculation(self, data, attr_list, t_attr):
		b_id=-1
		min_entr=1000
		labels=[]
		for i in range(0, len(data)):
			if data[i][t_attr] not in labels:
				labels.append(data[i][t_attr])
		for i in range(0, len(attr_list)):
			values=[]
			checked=[]
			for j in range(0, len(data)):
				if data[j][i] not in checked:
					checked.append(data[j][i])
					values.append(data[j][i])
			if ((len(labels)*1.0/len(values))<0.006):
				continue;
			valueLabels=np.zeros((len(values), len(labels)))
			
			for k in range(0, len(data)):
				rid=self.get_id(data[k][i], values)
				if (rid!=-1): 
					cid=self.get_id(data[k][t_attr], labels)					
				if ((rid!=-1) and (cid!=-1)):
					valueLabels[rid,cid]=valueLabels[rid,cid]+1
			
			entrs=np.zeros((1,valueLabels.shape[0]))
			weightings=np.zeros((1,valueLabels.shape[0]))
			
			for m in range(0, valueLabels.shape[0]):
				entrs[0,m]=self.get_entr(valueLabels[m])
			
			ss=0		
			for k in range(0, valueLabels.shape[0]):
				ss=ss+sum(valueLabels[k])		
			for l in range(0, valueLabels.shape[0]):
				weightings[0, l]=float(sum(valueLabels[l])/ss)
			
			sum_entr=0			
			for m in range(0, valueLabels.shape[0]):
				sum_entr=sum_entr+weightings[0,m]*entrs[0,m]			

			print "information gain of",attr[attr_list[i]],"= %.4f" % sum_entr				
			if (sum_entr<min_entr):
				b_id=i
				min_entr=sum_entr		

		attribIndex=attr_list.pop(b_id)		
		toStop=False
		return attribIndex,toStop;

	tree={}
	defaultLabel="NULL"	

	def buildtree(self, data, attr_list, t_attr, ts_length):	
		classCounts=Counter([instance[t_attr] for instance in data])
		default=classCounts.most_common(1)[0][0]
		labels=[]
		for i in range(0, len(data)):
			labels.append(data[i][t_attr])

		if ((len(data)*1.0)/ts_length<=0.02): 
			return default
		
		if (data is None) or len(attr_list)<=0:
			return default

		if (labels.count(labels[0])==len(labels)):
			return labels[0]
	
		toStop=False
		if (len(attr_list)>-1):
			b_id_attr, toStop=self.ig_calculation(data, attr_list, t_attr)
			print "Best attribute:", attr[b_id_attr]

		else:
			random.shuffle(attr_list)
			b_id_attr=attr_list.pop(-1)
		
		tree={b_id_attr:{}}
        	unique=[]
        	checked=[]
        	for i in range(0, len(data)):
            		if data[i][b_id_attr] not in checked:
                		checked.append(data[i][b_id_attr])
                		unique.append(data[i][b_id_attr])
        	print "#b=",len(unique)
        	for value in unique:
			dataSubset=[]
			for i in range(0, len(data)):
				if (data[i][b_id_attr]==value):
					dataSubset.append(data[i])
            		subTree=self.buildtree(dataSubset, attr_list, t_attr, ts_length)
            		tree[b_id_attr][value]=subTree
        	return tree
		
	def cf(self, tree, instance, max, defaultLabel=None):		
		if (tree is None):
			return None 
	
		if (not isinstance (tree, dict)):
			return tree
	
		root=tree.keys()[0]
		subTrees=tree.values()[0]		 
		branch=len(tree.values()[0])	
		if (branch>max[0]):max[0]=branch	
		if instance[root] not in subTrees:
			return None
		return self.cf(subTrees[instance[root]], instance, max)
		
	def classify(self, test_instance):		
		max=[0]
		result=self.cf(self.tree, test_instance, max)
		if result is None:
			result=self.defaultLabel
		return result

def implement():	
	with open("hw4-data.csv") as ff:
	    data=[tuple(line) for line in csv.reader(ff, delimiter=",")]
	
	#print data	
    	
	ss=0
	for j in range (0,10):
		training=[x for i, x in enumerate(data) if i % 10!=j]
		test=[x for i, x in enumerate(data) if i % 10==j]	
		tree=DecisionTree()
		attribs=range(0,(len(training[0])-1))
		t_attrib=len(training[0])-1
		exattrs=[]
		classCounts=Counter([instance[t_attrib] for instance in training])
		tree.defaultLabel=classCounts.most_common(1)[0][0]
		labels=[]
		for k in range(0, len(training)):
			if training[k][t_attrib] not in labels:
				labels.append(data[k][t_attrib])
		for k in range(0, len(attribs)):
			values=[]
			checked=[]
			for l in range(0, len(training)):
				if training[l][k] not in checked:
					checked.append(data[l][k])
					values.append(data[l][k])
			if ( (len(labels)*1.0/len(values))<0.001):		
				exattrs.append(attribs[k])
		
		notexattrs=[]
		
		for i in range(0,len(attribs)):
			if(attribs[i] not in exattrs):
				notexattrs.append(attribs[i])		
		tree.tree=tree.buildtree(training, notexattrs, t_attrib, len(training))

		results=[]
		for k in test:
			result=tree.classify(k[:-1])
			results.append(result==k[-1])
	
		# Accuracy
		accuracy=float(results.count(True))/float(len(results))
		ss += accuracy
		print "cv=",j," accuracy: %.4f" % accuracy
		print "\n"

	accuracy=ss/10
	print "The accuracy is %.4f" % accuracy
	# Writing results to a file (DO NOT CHANGE)
	f=open("result.txt", "w")
	f.write("accuracy: %.4f" % accuracy)
	f.close()	
	
if __name__=="__main__":
	implement()