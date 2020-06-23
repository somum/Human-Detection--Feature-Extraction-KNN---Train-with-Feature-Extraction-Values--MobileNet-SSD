from keras import applications
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model

def intermediate_output(model,img):


    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-2].output)

    img_path = 'frame0.jpg'
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    intermediate_output = intermediate_layer_model.predict(x)

    return intermediate_output
