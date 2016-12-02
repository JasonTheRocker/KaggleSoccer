import numpy as np

def replaceStrFeature(word):
	new = word.replace("(","").replace(")","").replace("[","").replace("]","").replace("u'", "").replace("'", "").replace("00:00:00", "").replace("-","").replace("/","").strip()
	try:
		convert = float(new)
	except ValueError:
		if "high" in new:
			#convert = 1.0
			convert = 100.0
		elif "low" in new:
			#convert = -1.0
			convert = 50.0
		elif "le" in new:
			convert = 100.0
		elif "right" in new:
			convert = 50.0
		elif "None" in new:
			convert = 0.0
		elif "medium" in new:
			convert = 0.0
		elif "o" in new:
			convert = 0.0
		elif "y" in new:
			convert = 100.0
		elif "es" in new:
			convert = 0.0
		elif "ean" in new:
			convert = 0.0
		else:
			print new
			convert = new
		
	return convert

def replaceStrV2(w):
	new = word.replace("[","").replace("]","").strip()
	convert = float(new)
	return convert
	

def getDataAndLabels():
	f = open('FeaturesV1.txt', 'r')
	line = f.readline()
	labels = []
	data = []	
	feature_size = len(line.split(",")) - 2	

	while line != "":
		record = line.split(",")
		home = int(record[9].strip())
		away = int(record[10].strip())
		if home > away:
			labels.append([float(1),float(0),float(0)])
		elif home == away:
			labels.append([float(0),float(1),float(0)])
		else:
			labels.append([float(0),float(0),float(1)])
		record = map(replaceStrFeature, record)
		
		# need to pop out player id, home, away, 
		for i in range(56, 78):
			record.pop(55)
		for i in range(0, 11):
			record.pop(0)	
		for i in range(0, 22):
			record.pop(0)
		for i in range(0, 22):
			record.pop(0)		
			
		data.append(record)
		line = f.readline()
	f.close()
	return data, labels

def PCAreduce(data):
	ret = []
	for i in data:
		b = [sum(i[current: current+40]) for current in xrange(0, len(i), 40)]
		ret.append(b)
	return ret

def getDataAndLabelsV2():
	f = open('FeaturesV2.txt', 'r')
	line = f.readline()
	labels = []
	data = []	
	feature_size = len(line.split(",")) - 2
	
	while line != "":
		record = line.split(",")
		record = map(replaceStrFeature, record)
		data.append(record)
		line = f.readline()
	f.close()
	
	f = open('Label.txt', 'r')
	line = f.readline()
	while line != "":
		record = float(line)
		labels.append(record)
		line = f.readline()
	f.close()	
	
	return data, labels, feature_size	


if __name__ == "__main__":
	data, labels = getDataAndLabels()
	f = open('Label.txt', 'w+')
	for i in labels:
        	print >>f,  i	
		