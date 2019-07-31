import csv
import sys
import math

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def __init__(self,k=2):
	self.k=k
	self.centroids={}

def read_data(file_path, num_ex=-1):
	if num_ex == -1:
		num_ex = 1000
	file = open(file_path, 'r')
	raw_data = csv.DictReader(file)
	train_data = list(raw_data)
	for row in train_data:
		row['latitude'] = float(row['latitude'])
		row['longitude'] = float(row['longitude'])
		row['reviewCount'] = float(row['reviewCount'])
		row['checkins'] = float(row['checkins'])

	return train_data

def euclidean_distance(point, centroid):
	res = 0

	res += math.pow(point[0] - centroid[0])
	res += math.pow(point[1] - centroid[1])
	res += math.pow(point[2] - centroid[2])
	res += math.pow(point[3] - centroid[3])
	res = math.sqrt(res)

	return res

def manhattan_distance(point, centroid):
	res = 0
	res += math.abs(point[0] - centroid[0])
	res += math.abs(point[1] - centroid[1])
	res += math.abs(point[2] - centroid[2])
	res += math.abs(point[3] - centroid[3])	
	return res

def init_centroids(dataset, k):
    #dataset length:21092
	for i  in range k:
		self.centroids[i]=dataset[i]

	


def get_centroids(dataset, k):

	for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break



if __name__ == '__main__':
	if len(sys.argv) < 4:
		print("invalid args")
		exit(1)

	file_path = sys.argv[1]
	if "/given/" not in file_path:
		file_path = "../data" + "/given/" + file_path
	else:
		file_path = "../data" + file_path

	train_data = read_data(file_path)
	# k value is argv 2

	# 4 continuous attributes: latitude, longitude, reviewCount, checkin
	# cluster have five options
	options = int(sys.argv[3])

	if options == 1:
		# original data
		print(options, k_value)
	elif options == 2:
		# log transform to reviewCount and checkins
		for row in train_data:
			row['reviewCount'] = math.log(row['reviewCount'])
			row['checkins'] = math.log(row['checkins'])

		print(options, k_value)
	elif options == 3:
		# standerized four attributes

		print(options, k_value)
	elif options == 4:
		# standardized four attributes find manhattan distance for clustering

		print(options, k_value)
	elif options == 5:
		# use 3% random samples data for clustering
		pec = 0.03
		print(options, k_value)
	else:
		print("invalid option")
		exit(1)

	# using distance mode, k_value, and dataset to find centroid



	




