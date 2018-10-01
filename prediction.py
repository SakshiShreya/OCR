
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from keras.models import model_from_yaml

# In[2]:


def load_model(bin_dir):
    ''' Load model from .yaml and the weights from .h5

        Arguments:
            bin_dir: The directory of the bin (normally bin/)

        Returns:
            Loaded model from file
    '''

    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model


# In[24]:


def predict(x, model, mapping):

    # Visualize new array

    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    out = model.predict(x)

    # Generate response
    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]}

    return response

# model = load_model('bin/')
# mapping = pickle.load(open('%s/mapping.p' % 'bin/', 'rb'))

# # In[28]:
# x = imread('a.jpg', mode='L')


# # In[29]:
# print(predict(x))

