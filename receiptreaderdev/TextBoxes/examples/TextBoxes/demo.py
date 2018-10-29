import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import xml.dom.minidom
sys.path.append('/home/nhutsam/')
# %matplotlib inline
from nms import nms
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/nhutsam/receiptreaderdev/TextBoxes/'  # this file is expected to be in {caffe_root}/examples
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
#caffe.set_device(0)
#caffe.set_mode_gpu()
caffe.set_mode_cpu()
model_def = '/home/nhutsam/receiptreaderdev/TextBoxes/examples/TextBoxes/deploy.prototxt'
model_weights = '/home/nhutsam/receiptreaderdev/TextBoxes/examples/TextBoxes/TextBoxes_icdar13.caffemodel'

use_multi_scale = False

if not use_multi_scale:
    scales=((700,700),)
else:
	scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))

# In[1]:
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# In[2]:7
dt_results=[]
image_path='/home/nhutsam/receiptreaderdev/TextBoxes/examples/img/demo18.jpg'
image_path
image=caffe.io.load_image(image_path)
image_height,image_width,channels=image.shape
plt.clf()
plt.imshow(image)
currentAxis = plt.gca()
for scale in scales:
	print(scale)
	image_resize_height = scale[0]
	image_resize_width = scale[1]
	transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_mean('data', np.array([104,117,123])) # mean pixel
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
	
	net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)		
	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image
	# Forward pass.
	detections = net.forward()['detection_out']
	# Parse the outputs.
	det_label = detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]
	top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
	top_conf = det_conf[top_indices]
	top_xmin = det_xmin[top_indices]
	top_ymin = det_ymin[top_indices]
	top_xmax = det_xmax[top_indices]
	top_ymax = det_ymax[top_indices]

	for i in xrange(top_conf.shape[0]):
		xmin = int(round(top_xmin[i] * image.shape[1]))
		ymin = int(round(top_ymin[i] * image.shape[0]))
		xmax = int(round(top_xmax[i] * image.shape[1]))
		ymax = int(round(top_ymax[i] * image.shape[0]))
		xmin = max(1,xmin)
		ymin = max(1,ymin)
		xmax = min(image.shape[1]-1, xmax)
		ymax = min(image.shape[0]-1, ymax)
		score = top_conf[i]
		dt_result=[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,score]
		dt_results.append(dt_result)
		
		
# In[]:
from math import sqrt
nms_flag = nms(dt_results,0.4)
def sortBycoordinate(dt_results):
    coordinate_result = [dt_results[k] for k,dt in enumerate(dt_results) if nms_flag[k]==True]
    coordinate_result = sorted(coordinate_result, key=lambda x: x[1])
    n = len(coordinate_result)
    flag_res=[False]*len(coordinate_result)  
    j=0
    final_res=[]
    while j!=-1:
        res=[]
        temp = (coordinate_result[j][5]+coordinate_result[j][1])/2
        for i in range(j,n):
            if coordinate_result[i][5]>temp and coordinate_result[i][1]<temp:
                res.append(coordinate_result[i])
                flag_res[i]=True
        final_res.append(res)
        for i in range(0,n):
            if flag_res[i]==False:
                j=i
                break
            else:
                j=-1
    for i in range(len(final_res)):
        final_res[i]=sorted(final_res[i], key=lambda x: x[0])
    return final_res
# In[]:
dt_results=sortBycoordinate(dt_results)
dt_results=sum(dt_results,[])
k=0
for dt in dt_results:
    name = '%d ' %(k)
    xmin = dt[0]
    ymin = dt[1]
    xmax = dt[2]
    ymax = dt[5]
    coords = (xmin, ymin), xmax-xmin+5, ymax-ymin+5
    color = 'b'
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.5})
    k=k+1

#plt.savefig('/home/nhutsam/TextBoxes/examples/results/demo_result.jpg')
plt.show()
print('success')

# In[]:


# Using pytesseract framework recognize text in image
import cv2
sys.path.append('/home/nhutsam/receiptreaderdev/neuralCRNN/')
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = '/home/nhutsam/receiptreaderdev/neuralCRNN/models/crnn.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

img=cv2.imread(image_path)
transformer = dataset.resizeNormalize((100, 32))

results =''
for dt_i_th in dt_results:
    dt_res=dt_i_th
    if dt_res[8] < 0.8 :
        continue
    xmin=dt_res[0]
    ymin=dt_res[1]
    xmax=dt_res[2]
    ymax=dt_res[5]
    drop_img=img[ymin:ymin+(ymax-ymin+5), xmin:xmin+(xmax-xmin+5)]
    
    image = Image.fromarray(drop_img).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    results+=sim_pred+' '










