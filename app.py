from flask import Flask,jsonify,request,render_template
from flask_restful import reqparse, Resource , Api
import PIL.Image
import json
import torchvision.transforms as transforms
from torchvision import models
import io
from flask_cors import CORS

#init 
app = Flask(__name__)
CORS(app)

#wrap Api to the app
api= Api(app)

parser = reqparse.RequestParser()

#path to labels mapping id - label
class_index = json.load(open('imagenet_class_index.json'))

#Define pre-trained model
model = models.densenet121(pretrained=True)
#using eval mode
model.eval()

# Transform the image to Pytorch format
# image -> bytes -> transform to tensor
def tfm_img(img):
    img_torch = transforms.Compose([ 
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = PIL.Image.open(io.BytesIO(img))
    return  img_torch(img).unsqueeze(0)

def predict(img_bytes):
    img_tensor = tfm_img(img=img_bytes)
    outputs = model.forward(img_tensor)
    #get the best prediction in all class
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    #get the value in dict -> return array 
    return class_index[predicted_idx]


#Resources
class Image(Resource):
    def post(self):
        #receive th request
        file = request.files['img']
        # Save the image 
        img_bytes = file.read()
        # do prediction
        class_id , class_name = predict(img_bytes=img_bytes)
        return jsonify({'class':class_name})
        # return jsonify({"t":"v"})
#endpoint
api.add_resource(Image,'/predict')

class Quotes(Resource):
    def get(self):
        return {
            'William Shakespeare': {
                'quote': ['Love all,trust a few,do wrong to none',
		'Some are born great, some achieve greatness, and some greatness thrust upon them.']
        },
        'Linus': {
            'quote': ['Talk is cheap. Show me the code.']
            }
        }

api.add_resource(Quotes, '/')

if __name__ == '__main__':
    app.run(debug=True)

