#This program is used for giving custom user input to the NN and see the output.

# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import prediction as pred
import numpy as np
import pickle

def get_string_from_nn(all_letters):
	# net = network2.Network([1024, 30, 66], cost=network2.CrossEntropyCost)
	
	# biases_saved = np.load('biases.npy', encoding = 'latin1')
	# weights_saved = np.load('weights.npy', encoding = 'latin1')

	# #all_letters = np.load('all_letters.npy')
	# #all_letters = all_letters.tolist()

	# word_string = ""
	
	# i = 0
	# for x in all_letters:
	# 	output = np.argmax(net.feedforward(x, biases_saved = biases_saved, weights_saved = weights_saved))
		
	# 	#second stage classification below
	# 	if (output in (18, 19, 21, 29, 44, 47, 1)):
	# 		output = get_let_from_2nd_nn_ijltIL1(x)
	# 	elif (output in (12, 14, 42)):
	# 		output = get_let_from_2nd_nn_ceg(x)
			
	# 	word_string = word_string + get_letter(output)
	# 	i = i + 1

	# return word_string

	model = pred.load_model('bin/')
	mapping = pickle.load(open('%s/mapping.p' % 'bin/', 'rb'))

	word_string = ""
	
	i = 0
	for x in all_letters:
		output = pred.predict(x, model, mapping)			
		word_string = word_string + output['prediction']
		i = i + 1

	return word_string

	#print np.argmax(net.feedforward(test_data[502][0], biases_saved = biases_saved, weights_saved = weights_saved))