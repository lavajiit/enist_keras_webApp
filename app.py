from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re, sys, os, base64
from io import StringIO, BytesIO
from PIL import Image

sys.path.append(os.path.abspath("./model"))
from load import * 
app = Flask(__name__)
global model, graph
model, graph = init()
	

@app.route('/')
def index():
	return render_template("index.html")

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	imgstr = re.search(b'base64,(.*)',imgData).group(1)
	img_bytes = base64.b64decode(imgstr)
	img = Image.open(BytesIO(img_bytes)).convert('L')
	x  = np.array(img)

	x = imresize(x,(28,28))
	x = np.invert(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		# print(out)
		index = np.argmax(out,axis=1)[0]
		print(index)
		return class_mapping[index]	
	

if __name__ == "__main__":
	# port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=5000)
