import collections
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------- #
# INPUT: 
#		f_name_1: name of the first file
#		f_name_2: name of the second file
#		dictionary_size: dimension of the recognized words
# OUTPUT:
#		dictionary: dict variable that maps each word into its code
#		reverse_dictionary: dict variable that maps each code into its word
#
# We open each file and count the number of each word appears 
# storing it in word_counter (collections.Counter() datatype), then we put in dictionary
# the "dictionary_size" most frequent appearing words.
# We assign to each word an increasing number. (A lower code corresponds to a higher frequency)
# We reserve #0 to unknown words.
# ---------------------------------------------------------- #
def create_dictionary(f_name_1, f_name_2, dictionary_size):	
	dictionary = dict()
	reverse_dictionary = dict()

	word_counter = collections.Counter()
	word_code = 1
	
	with open(f_name_1) as f:
		for line in f:
			word_counter.update(line.split())

	with open(f_name_2) as f:
		for line in f:
			word_counter.update(line.split())

	for word, _ in word_counter.most_common(dictionary_size):
		dictionary[word] = word_code
		reverse_dictionary[word_code] = word
		word_code += 1
	
	reverse_dictionary[0] = ""
	return dictionary, reverse_dictionary

# ---------------------------------------------------------- #
# INPUT: 
#		dataset: current dataset containing the integer values of the triplets
#		dictionary: dict variable that maps each word into its code
#		author: DANTE = 0, PETRARCA = 1
#		file_name: name of file 
# OUTPUT:
#		dataset_: updated dataset containing the integer values of the triplets
#
# We read each word in the file mapping it into its code. 
# In order to create a triplet we append to "triplet" the codes corresponding to three 
# consecutive lines. Then we insert in the first position of "triplet" the author value.
# Eventiually we append triplet to "dataset". We repeat this procedure untill the end of the file.
# If the read word is not present in the dictionary, then we skip it.
# ---------------------------------------------------------- #
def divide_in_triplets(dataset, dictionary, author, file_name):
	dataset_ = dataset
	triplet = []

	with open(file_name) as f:
		count_lines = 0
		for line in f:
			count_lines += 1
			for word in line.split():
				if word in dictionary:
					triplet.append(dictionary[word]) 
			if count_lines == 3:
				triplet.insert(0, author)
				y = []
				for j in triplet:
					y.append(j)

				dataset_.append(y)
				count_lines = 0
				triplet.clear()
	return dataset_

# ---------------------------------------------------------- #
# INPUT: 
#		dataset: current dataset containing the integer values of the triplets
# OUTPUT:
#		dataset_: updated dataset containing the integer values of the triplets
#		triplet_size-1: lenght of the triplets
#
# We first look for the maximum lenght between the triplets.
# Then for each triplet with lenght less than the maximum one, we fill it up with zeros
# untill the dimensions match appending it to "dataset_".
# Then in order to randomize things, we shuffle the rows of the created "dataset_".
# ---------------------------------------------------------- #
def dataset_processing(dataset):
	dataset_ = []
	data_ = []
	triplet_size = 0

	for data in dataset:
		if len(data) > triplet_size:
			triplet_size = len(data)
		
	for data in dataset:
		data_ = data
		for i in range(len(data), triplet_size):
			data_.append(0)
		dataset_.append(data_)

	np.random.shuffle(dataset_)

	return dataset_, (triplet_size-1)

# ---------------------------------------------------------- #
# INPUT: 
#		dataset: current dataset containing the integer values of the triplets
#		test_percent: percentage of the whole dataset that we assing to testing
#		validation_percent: percentage of the whole dataset that we assing to validation
# OUTPUT:
#		training: training set
# 		validation: validation set
# 		test: test set
# In "test_size", "validation_size", "training_size" we ce calculate the number of "dataset" 
# to assing to test, validation, training sets respectively.
# We append the fist "training_size" rows to "training", the next "validation_size" 
# rows  to "validation" and the last "test_size" to test.
# ---------------------------------------------------------- #
def split_dataset(dataset, test_percent, validation_percent):
	training = []
	validation = []
	test = []

	test_size = int(len(dataset)*test_percent)
	validation_size = int(len(dataset)*validation_percent)
	training_size = int(len(dataset)-(test_size + validation_size))

	i = 0
	for data in dataset:
		if(i < training_size):
			training.append(data)
		if (i >= training_size) and (i < (training_size + validation_size)):
			validation.append(data)
		if i >= (training_size + validation_size):
			test.append(data)

		i += 1 

	print("\n---------------------------------------------------------")
	print("Size Dataset:", len(dataset),"\nSize Training ", len(training),"\nSize Validation:", len(validation), "\nSize Test:", len(test))
	
	return training, validation, test

# ---------------------------------------------------------- #
# INPUT: 
#		dataset: current dataset containing the integer values of the triplets
#		min_perc: minimum percentage of "dataset" that "batch_size" can have.
# OUTPUT:
#		batch_size: batch size.
#
# We look for the minumum integer that divides without residue the number of rows
# of "dataset" that's greater or equal than the minimum number of rows calculated with "min_perc".
# ---------------------------------------------------------- #
def find_batch_size(dataset, min_perc):
	batch_size = int(len(dataset)*min_perc)

	while(len(dataset)%batch_size != 0):
		batch_size += 1
	
	return batch_size

# ---------------------------------------------------------- #
# INPUT: 
#		dataset: current dataset containing the integer values of the triplets
#		actual_batch_nr: current number of the batch to be returned 
#		batch_size: size of the batch
#		triplet_size: lenght of the triplets
# OUTPUT:
#		x: processed input batch
#		y: processed label batch
#
# We store in "x" each line of "dataset" (belonging to the current batch) without
# the first element (author label).
# We store in "y" the one hot encoding vector: [1 0] -> DANTE or [0 1] -> PETRARCA.  
# ---------------------------------------------------------- #
def get_next_batch(dataset, actual_batch_nr, batch_size, triplet_size):
	x = np.ndarray(shape=(batch_size, triplet_size), dtype=np.int32)
	y = np.ndarray(shape=(batch_size, 2), dtype=np.float32)

	row_x = []
	row_y = []
	out = []
	for i in range(0, batch_size):
		for j in range(1, triplet_size+1):
			row_x.append(dataset[actual_batch_nr*batch_size+i][j])
		
		if dataset[actual_batch_nr*batch_size+i][0] == 0:
			out.append(1)
			out.append(0)
		else:			
			out.append(0)
			out.append(1)

		x[i] = row_x
		y[i] = out
		out.clear()
		
		row_x.clear()
		row_y.clear()
	return x, y 

# ---------------------------------------------------------- #
# INPUT: 
#		vocabulary_size:
#		train: 
#		test: 
#		validation:
#		triplet_size: lenght of the triplets
#		batch_size:
# OUTPUT:
# 
# This function trains a neueal network composed by a Recurrent Neural Network (RNN)
# and a Single Layer Perceptron (SLP). 
# The first step performs an embedding in which each word of the dictionary is transformed into
# a numeric vector (randomly initialized) of size "embedding_size". Each row of this matrix 
# corresponds to one word in the dictionary.
# Then the RNN takes an entire triplet word by word and the state corresponding to each triplet
# is feeded to the SLP that classifies it to the correct author.
# ---------------------------------------------------------- #
def train_model(vocabulary_size, train, test, validation, reverse_dictionary,triplet_size, batch_size):
	
	# DEFINITIONS:
	numClasses = 2
	nMaxEpochs = 1

	# Used to create the plot
	loss_history = np.empty(shape=[1], dtype = float)
	accuracy_history = np.empty(shape=[1], dtype = float)

	# Arrays used for storing inputs & labels 
	x = np.ndarray(shape=(batch_size, triplet_size), dtype=np.int32)
	y = np.ndarray(shape=(batch_size, numClasses), dtype=np.float32)
	
	# EMBEDDING:
	
	# Dimension of the embedded word
	embedding_size = 300
	
	# Tensorflow placeholders used to feed the input & targets to the net
	train_inputs = tf.placeholder(tf.int32, shape=[None, triplet_size])
	train_labels = tf.placeholder(tf.float32, shape=[None, numClasses])

	# Tensorflow variable corresponding to the embedding matrix
	embeddings = tf.Variable(tf.zeros([vocabulary_size+1, embedding_size]),dtype=tf.float32)

	# Embedding vectors correspondg to the words contained in "train_inputs"
	embed = tf.nn.embedding_lookup(embeddings, train_inputs)

	# RECURRENT NEURAL NETWORK:

	# The number of units in the LSTM cell
	lstmUnits = 20

	# Way to create a RNN in Tensorflow, "state" is the current state of our RNN
	lstmCell = tf.nn.rnn_cell.LSTMCell(lstmUnits)
	lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
	_, state = tf.nn.dynamic_rnn(lstmCell, embed, dtype=tf.float32)

	# SINGLE LAYER PERCEPTRON:
	
	# Creation of one layer with input "state.h" and output dimension "numClasses". 
	# The activation function is linear
	yOut = tf.layers.dense(state.h, numClasses)


	# Computes softmax cross entropy between logits and labels.
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yOut, labels=train_labels))

	# Optimizer that implements the Adam algorithm.
	optimizer = tf.train.AdamOptimizer().minimize(loss)
	
	# TRAINING
	print("\n---------------------------------------------------------")
	print("TRAINING:")

	# Initialization of the global variables in the graph
	tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		# Beginning of the training
		for j in range (0, nMaxEpochs):
			for i in range(0, int(len(train)/batch_size)):
				
				# The current batch fetched
				x, y = get_next_batch(train,i,batch_size,triplet_size)

				# Training step & loss calculation all over the batch
				_, loss_= session.run([optimizer, loss], {train_inputs: x, train_labels: y})

				# The value of "loss_" is stored for the plot
				loss_history = np.append(loss_history, loss_)

			# VALIDATION

			# The validation set is fetched (entirely)
			x, y = get_next_batch(validation,0,len(validation),triplet_size)

			# We put the output of the network in the one hot encoding form
			prediction = tf.argmax(yOut,1) 

			# We check if the output of the network is equal to the target & we average the result
			correct_prediction = tf.equal(prediction, tf.argmax(y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			accuracy_ = session.run([accuracy],{train_inputs: x, train_labels: y})
			accuracy_history = np.append(accuracy_history, accuracy_)

			progress = j / nMaxEpochs
			print("Progress:", progress*100, "%")
		print("Progress: 100 % -> TRAINING COMPLETED")

		#TEST:

		# The test set is fetched (entirely)
		x, y = get_next_batch(test,0,len(test),triplet_size)

		# We compute the same  steps as above
		prediction = tf.argmax(yOut,1) 
		correct_prediction = tf.equal(prediction, tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		accuracy_, yOut_ = session.run([accuracy, yOut],{train_inputs: x, train_labels: y})

		print("\n---------------------------------------------------------")
		print("ACCURACY ON VALIDATION SET:")
		for p in range(0, len(accuracy_history)):
			pr = accuracy_history[p] * 100
			print("Epoch:", p+1, ": %.2f" % pr, " %")

		print("\n---------------------------------------------------------")
		print("ACCURACY ON TEST SET:")
		pr = accuracy_ * 100
		print("%.2f" % pr, " %")

		# Prediction Examples
		reverse_triplet(x,y,np.array(yOut_), reverse_dictionary)
		
		# PLOTTING LOSS:
		loss_history = np.delete(loss_history, 0, 0)
		
		plt.plot(range(len(loss_history)),loss_history)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('Loss function')
		plt.show()

# ---------------------------------------------------------- #
# INPUT: 
#		x: array of inputs of the network (test set)
#		y: array of labels (test set)
#		yOut: Network outputs corresponding to x
#		reverse_dictionary: dict variable that maps each code into its word
#
# Here we show one example for each poet. We store the first triplet of Dante
# and Petrarca encountered in the test set. We print the recognised words that 
# compose each one of them throgh "reverse_dictionary". Then we display the label
# and the output of the network corresponding to the two triplets.
# ---------------------------------------------------------- #
def reverse_triplet(x, y, yOut, reverse_dictionary):
	dante_index = -1
	petrarca_index = -1

	dante_triplet = ""
	petrarca_triplet = ""


	for i in range(0, len(y)):
		if y[i][0] == 1 and y[i][1] == 0 and dante_index == -1:
			dante_index = i
		if y[i][0] == 0 and y[i][1] == 1 and petrarca_index == -1:
			petrarca_index = i
	
	for word in x[dante_index]:
		#dante_triplet.append(reverse_dictionary[word])
		dante_triplet = dante_triplet + reverse_dictionary[word] + " "
		
	for word in x[petrarca_index]:
		petrarca_triplet = petrarca_triplet + reverse_dictionary[word] + " "
	
	if(np.argmax(y[dante_index], axis=0) == np.argmax(yOut[dante_index], axis=0)):
		dante_label = "Label: DANTE, Network: DANTE"
	else:
		dante_label = "Label: DANTE, Network: PETRARCA"

	if(np.argmax(y[petrarca_index], axis=0) == np.argmax(yOut[petrarca_index], axis=0)):
		petrarca_label = "Label: PETRARCA, Network: PETRARCA"
	else:
		petrarca_label = "Label: PETRARCA, Network: DANTE"

	print("\n---------------------------------------------------------")
	print("PREDICTION EXAMPLES:")
	print("\nTriplet:", dante_triplet)
	print(dante_label)
	print("\nTriplet:", petrarca_triplet)
	print(petrarca_label)
	print("---------------------------------------------------------")

def main():
	# VARIABLES:

	dictionary = dict()
	reverse_dictionary = dict()

	dataset = []
	training = []
	validation = []
	test = []

	# COSTANTS:
	DICTIONARY_SIZE = 20000
	TRIPLET_SIZE = -1

	DANTE = 0
	PETRARCA = 1

	TEST_PERCENT = 0.2
	VALIDATION_PERCENT= 0.16

	# Creation of the dictionary & reverse dictionary
	dictionary, reverse_dictionary = create_dictionary('src_text/divina_commedia.txt','src_text/canzoniere.txt', DICTIONARY_SIZE)

	# Division in triplets	
	dataset = divide_in_triplets(dataset, dictionary, DANTE, 'src_text/divina_commedia.txt')
	dataset = divide_in_triplets(dataset, dictionary, PETRARCA, 'src_text/canzoniere.txt')
	
	# Standardizing the triplet lenghts
	dataset, TRIPLET_SIZE = dataset_processing(dataset)

	# Split Dataset in Training, Validation and Test Set
	training, validation, test = split_dataset(dataset, TEST_PERCENT, VALIDATION_PERCENT)

	# Computation of the size of the training batch	
	BATCH_SIZE = find_batch_size(training,0.02)

	# Creation & training of the Neural Network
	train_model(DICTIONARY_SIZE, training, validation, test, reverse_dictionary, TRIPLET_SIZE, BATCH_SIZE)
	
if __name__ == "__main__":
	main()