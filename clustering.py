import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import csv

print("start")

class SOM:
    def __init__(self, height, width, inp_dimension):
        #inisialisasi height, width, input dimensionnya
         self.height = height
         self.width = width
         self.inp_dimension = inp_dimension

         #inisialisasi node, placeholder input, variable weight
         self.node = [ tf.to_float([i,j]) for i in range(height) for j in range(width) ]
         self.input = tf.placeholder(tf.float32, [inp_dimension])
         self.weight = tf.Variable(tf.random_normal([height*width, inp_dimension]))

         #buat fungsi cari node terdekat, update weight
         self.best_matching_unit = self.get_bmu(self.input)
         self.updated_weight = self.get_update_weight(self.best_matching_unit, self.input)

    def get_bmu(self, input):
        expand_input = tf.expand_dims(input, 0)
        euclidean = tf.square(tf.subtract(expand_input, self.weight))
        distances = tf.reduce_sum(euclidean, 1) #reduce_sum(euclidean, [1 atau 0]), klo 1 brarti jumlahin matrix per baris (ke kanan), klo 0 jumlahin per kolom (ke bawah)

        min_index = tf.argmin(distances, 0)
        bmu_location = tf.stack([tf.mod(min_index, self.width), tf.div(min_index, self.width)]) #x: min_index dimodulus width, y: min_index dibagi width
        return tf.to_float(bmu_location)
        
    def get_update_weight(self, bmu, input):
        #inisialisasi learning rate & sigma
        learning_rate = .5
        sigma = tf.to_float(tf.maximum(self.height, self.width)/2)

        #hitung perbedaan jarak bmu ke node lain pake euclidean
        expand_bmu = tf.expand_dims(bmu, 0)
        euclidean = tf.square(tf.subtract(expand_bmu, self.node))
        distances = tf.reduce_sum(euclidean, 1)
        ns = tf.exp(tf.negative(tf.div(distances**2, 2 * sigma**2)))
        
        #hitung rate
        rate = tf.multiply(ns, learning_rate)
        numofnode = self.height * self.width
        rate_value = tf.stack( [ tf.tile(tf.slice(rate, [i], [1]), [self.inp_dimension]) for i in range(numofnode) ] )

        #hitung update weight nya
        weight_diff = tf.multiply( rate_value, tf.subtract( tf.stack([input for i in range(numofnode)]), self.weight ) )

        updated_weight = tf.add(self.weight, weight_diff)

        return tf.assign(self.weight, updated_weight)

    def train(self, dataset, numofepoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(numofepoch):
                for data in dataset:
                    feed = { self.input : data }
                    sess.run([self.updated_weight], feed)
            
            self.weight_value = sess.run(self.weight)
            self.node_value = sess.run(self.node)
            cluster = [ [] for i in range(self.width) ]
            
            for i, location in enumerate(self.node_value): #enumerate return index nya juga
                cluster[int(location[0])].append(self.weight_value[i])
            
            self.cluster = cluster

#load data
def load_data(path):
    dataset = []
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            feature = row[0:23]
            dataset.append(feature)
    return dataset


dataset = load_data("clustering.csv")

#data preprocessing
def preprocessing(dataset):
    preprocess = []

    for feature in dataset:
        
        calorie = (float(feature[3]) + float(feature[4])) / 100 

        dividend = ( float(feature[7]) + float(feature[8])) 
        divisor = ( float(feature[6]) + float(feature[10]))

        if (divisor == 0):
            divisor = 1

        fat_ratio = dividend / divisor
        sugar = float(feature[12])
        protein = float(feature[14])
        salt = float(feature[15]) 

        feature2 = [calorie, fat_ratio, sugar, protein, salt]

        preprocess.append(feature2)

    return preprocess

dataset = preprocessing(dataset)

#normalize feature
def normalize(dataset, min, max):
    normalized=[]
    for i in range(0, len(dataset)):
        temp=[]
        for j in range(0, len(dataset[i])):
            dataset[i][j] = (dataset[i][j] - min[j]) / (max[j] - min[j])
            temp.append(dataset[i][j])
        normalized.append(temp)
    return normalized

#max min per fitur (inisialisasi infinite)
maxval = [-100000 for i in range(0,5)]
minval = [100000 for i in range(0,5)]

def MaxMinFeature(dataset):
    for i in range (0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if(dataset[i][j]>maxval[j]):
                maxval[j] = dataset[i][j]
            if(dataset[i][j]<minval[j]):
                minval[j] = dataset[i][j]  

MaxMinFeature(dataset)
dataset = normalize(dataset, minval, maxval)

def applyPCA(data):
    pca = PCA(n_components=3) 
    dataset_pca = pca.fit_transform(data)
    return dataset_pca

dataset = applyPCA(dataset)
print(dataset)
print(np.shape(dataset))
dataset = dataset[:6]

som = SOM(5,5,3)
som.train(dataset, 5000)

plt.imshow(som.cluster)
plt.show()