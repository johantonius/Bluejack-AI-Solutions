import tensorflow as tf
import numpy as np
import csv
import random
from sklearn.decomposition import PCA

#load label
def load_result(path):
    dataset = []
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            result = row[4]
            dataset.append(result)
    return dataset

result_list = ['a', 'b', 'c', 'd', 'e']
#label 
def preprocessing(label):
    preprocess = [] #bikin dataset kosong dulu karena dataset yg mau di proses
    
    for result in label:
    #     features = [float(x) for x in features] #ini dijadiin float semua karena ada feature yg float, jadi biar ga redundant di jadiin float semua
        
        #proses si targetnya biar jd matrix
        result_index = result_list.index(result) #ini untuk tau index mana siresult misal high itu di index 0 maka resultnya itu 0
        new_result = np.zeros(len(result_list), 'int') #bikin array [1x3] yang isinya 0 semua
        new_result[result_index] = 1 #nah ini buat jadiin misal high jadi [1 0 0] 1nya sesuai dimana index mereka medium [0 1 0]
        preprocess.append(new_result)
    return preprocess

label = load_result("classification.csv")
label = preprocessing(label)
#print(label)


#load feature
def load_feature(path):
    dataset = []
    predataset=[]
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        #dari setiap baris yang dibaca
        for row in reader:
            feature = row[5:11]
            feature.append(row[12])
            feature.append(row[14])
            feature.append(row[15])
            feature.append(row[17])
            feature.append(row[18])
            feature.append(row[19])
            feature.append(row[20])
            feature.append(row[23])
            result = row[4]
            # dataset.append((feature,result))
            predataset.append(feature)
        for features in predataset:
            features = [float(x) for x in features] 
            dataset.append(features)
    return dataset

features = load_feature("classification.csv")
 
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
#max min per fitur
maxval = [-10000000 for i in range(0,14)]
minval = [10000000 for i in range(0,14)]
def MaxMinFeature1(features):
    for i in range (0, len(features)):
        for j in range(0, len(features[i])):
            if(features[i][j]>maxval[j]):
                maxval[j] = features[i][j]
            if(features[i][j]<minval[j]):
                minval[j] = features[i][j]

features = MaxMinFeature1(features)
features = normalize(features,minval, maxval)

def denormalize(dataset, min, max):
    denormalized=[]
    for i in range(0, len(dataset)):
        temp=[]
        for j in range(0, len(dataset[i])):
            dataset[i][j] = ((dataset[i][j] * (max[j] - min[j])) + min[j])
            temp.append(dataset[i][j])
        denormalized.append(temp)
    return denormalized

#PCA
def applyPCA(data):
    pca = PCA(n_components=5)
    dataset_pca = pca.fit_transform(data)
    return dataset_pca

features = applyPCA(features)
def combineDataset(feature, target):
    dataset =[]
    for i in range(0, len(feature)):
        dataset.append((feature[i], target[i]))
    return dataset

dataset = combineDataset(features, label)
random.shuffle(dataset) 
counter = int(.7 *len(dataset))
train_dataset = dataset[:counter]
val_dataset = dataset[counter:]
counter_test_dataset = int(.1*len(dataset))
test_dataset = dataset[:counter_test_dataset]
#print(test_dataset)
#BPNN
num_of_input = 5
num_of_ouput = 5
num_of_hidden = [5,6]

learning_rate = .7
num_of_epoch = 5000
report_between = 100
report_between_valsave = 500

def fully_connected(input, numinput, numoutput):
    w = tf.Variable(tf.random_normal([numinput, numoutput]))
    b = tf.Variable(tf.random_normal([numoutput]))

    wx_b = tf.matmul(input, w) + b
    act = tf.nn.sigmoid(wx_b)
    return act

def build_model(input):
    layer1 = fully_connected(input, num_of_input, num_of_hidden[0])
    layer2 = fully_connected(layer1, num_of_hidden[0], num_of_hidden[1])
    layer3 = fully_connected(layer2, num_of_hidden[1], num_of_ouput)
    return layer3

#variable buat training input & output
trn_input = tf.placeholder(tf.float32,[None, num_of_input]) 
trn_target = tf.placeholder(tf.float32,[None, num_of_ouput]) 
save_dir = "./BPNN-model/" #. untuk cek kalo gaada folder tersebut maka akan dibikinin
filename = "bpnn.ckpt"

#training, model = hasil activation function
def optimize(model, dataset, valDataset, testDataset):
    # hitung E1, E2 dan langsung diambil averagenya
    error = tf.reduce_mean(.5 * (trn_target - model)**2)
    #ini untuk update weightnya
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    correct_prediction = tf.equal(tf.arg_max(model,1), tf.arg_max(trn_target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        maxError = -9999999
        for epoch in range(num_of_epoch):
            #training
            feature = [data[0] for data in dataset]
            result = [data[1] for data in dataset]
            feed = {trn_input: feature, trn_target:result}
            _, error_value, accuracy_value = sess.run([optimizer, error, accuracy], feed)

           #validating
            feature2 = [data[0] for data in valDataset]
            result2 = [data[1] for data in valDataset]
            feed2 = {trn_input: feature2, trn_target:result2}
            error_value2, accuracy_value2 = sess.run([error, accuracy], feed2)
            

            if epoch % report_between ==0:
                print("Training | Epoch : {}  | Error : {}".format(epoch, error_value*100)) 
            if epoch >=500: 
                if epoch % report_between ==0:
                    tot_error = error_value2*100
                    print("Validating | Epoch | Epoch : {}  | Error : {}".format(epoch, tot_error) )
                    if epoch % report_between_valsave==0:
                        if tot_error>maxError:
                            maxError = tot_error
                        else:
                            saver.save(sess, save_dir + filename, epoch)
        feature_test = [data[0] for data in testDataset]
        result_test = [data[1] for data in testDataset]
        feed_test = {trn_input: feature_test, trn_target: result_test}
        _, accuracy_value_test = sess.run([error ,accuracy], feed_test)
        print("Testing Accuracy: ", accuracy_value_test*100)
                
def testing(model, testing_data):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)

        
        saver.restore(sess, ckpt.model_checkpoint_path)

        state = sess.run(model.initial_state)
        for i in testing_data:
            x = np.array(i).reshape(1,1,1)
            dictionary_feed = {model.input : x , model.initial_state:state}
            prediction, state = sess.run([model.act, model.final_state], dictionary_feed)

            total_prediction = denormalize(prediction, min_val, max_val)
            accuracy = total_prediction/len(dataset)*100
            print("Prediction : {}".format(total_prediction))    
            print("Accuracy: {}".format(accuracy))
model = build_model(trn_input)
optimize(model, train_dataset, val_dataset, test_dataset)