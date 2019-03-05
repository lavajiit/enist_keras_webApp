from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re, sys, os, base64

sys.path.append(os.path.abspath("./model"))
from load import * 
app = Flask(__name__)
global model, graph
model, graph = init()

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		# output.write(imgstr.decode('base64'))
		output.write(base64.b64decode(imgstr))
	

@app.route('/')
def index():
	return render_template("index.html")

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	#read the image into memory
	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		index = np.argmax(out,axis=1)[0]
		print(index)
		return class_mapping[index]	
	

if __name__ == "__main__":
	# port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=5000)
