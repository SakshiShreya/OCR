#This program is used for giving custom user input to the NN and see the output.

# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import prediction as pred
import numpy as np
import pickle

def get_string_from_nn(all_letters):

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