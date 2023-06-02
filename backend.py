from flask import Flask, jsonify, request
import werkzeug
import os
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras
import  nltk
from nltk.stem import WordNetLemmatizer
import json
from flask import Flask, request, jsonify, json
from torchvision import transforms
import torch
from PIL import Image
from keras.models import load_model
import random
import pandas as pd
import numpy as np

lemmatizer = WordNetLemmatizer()



intents = json.loads(open('chat_bot/intents.json').read())

words = pickle.load(open('chat_bot/words.pkl', 'rb'))
classes = pickle.load(open('chat_bot/classes.pkl', 'rb'))
model = load_model('chat_bot/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag =[0]*len(words)
    
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res= model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results =[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    results.sort(key= lambda x:x[1], reverse=True)
    
    return_list =[]
    
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list
    

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result= random.choice(i['responses'])
            break
    return result




app = Flask(__name__)

@app.route('/hello', methods=['PUT', 'POST'])
def home():
    image_file = request.files['image']
    filename = werkzeug.utils.secure_filename(image_file.filename)
    path = "uploaded_images/"+filename
    image_file.save("./uploaded_images/"+filename)

    img_random = Image.open(path).convert("RGB")

    preprocess = transforms.Compose([
    transforms.Resize(265),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
        )
    ])

    img_random_preprocessed = preprocess(img_random)

    batch_img_random_tensor = torch.unsqueeze(img_random_preprocessed, 0)

    model = torch.load("leaf_dense_18.pt")

    out = model(batch_img_random_tensor)
    direct = "archive/New Plant Diseases Dataset(Augmented)/disease.txt"

    with open(direct) as f:
        disease = f.readlines()

    clean=[]
    for dis in disease:
        clean.append(dis.strip())

    print(clean)

    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0]*100

    _,indices = torch.sort(out, descending=True)

    top_5 = [(clean[idx], percentage[idx].item()) for idx in indices[0][:5]]
    print("...........................")
    print(top_5)
    list_of=[]

    for i in range(0,5):
        list_of.append(top_5[i][0])

    f = open("mappings.json")
    data = json.load(f)

    names=[]
    causes=[]
    fertilisers=[]

    for i in range(0, 38):
        if data['crops'][i]['name'] in list_of:
            names.append(data['crops'][i]['name'])
            causes.append(data['crops'][i]["cause"])
            fertilisers.append(data['crops'][i]['fertilizsers'])

    print(names)
    print(causes)
    print(fertilisers)

    if request.method=="PUT":
           return jsonify({
        "probability_1":{
            "name":names[0],
            "cause":causes[0],
            "fertiliser":fertilisers[0]
        },
         "probability_2":{
            "name":names[1],
            "cause":causes[1],
            "fertiliser":fertilisers[1]
        }, "probability_3":{
            "name":names[2],
            "cause":causes[2],
            "fertiliser":fertilisers[2]
        }, "probability_4":{
            "name":names[3],
            "cause":causes[3],
            "fertiliser":fertilisers[3]
        }, "probability_5":{
            "name":names[4],
            "cause":causes[4],
            "fertiliser":fertilisers[4]
        },

    })
    elif request.method=="POST":
        fertis =""
        for ferts in fertilisers[0]:
            fertis+=ferts

        # print(fertilisers[0])
        #string= "the disease your plant might be facing is "+names[0]+" and the cause of the diseases might be " + causes[0] + " the fertilizers that can be used are "+fertis

        string = "the disease your plant might be facing is"
        string+=names[0]
        string+="and the cause of the diseases might be "
        string+=causes[0]
        string+="the fertilizers that can be used are "
        string+=fertis

        return jsonify({
            "message":string
        })
        #return "the disease your plant might be facing is "+names[0]+" and the cause of the diseases might be "+causes[0]+" the fertilizers that can be used are "+fertis
    
    

    



    # return jsonify({
    #     "message":"saved successfully"
    # })



@app.route("/chat", methods=['PUT'])
def chat():
    body = request.get_json(force=True)
    text = body['input']
    ints = predict_class(text)
    ans = get_response(ints, intents)

    return jsonify({
        "message": ans
    })


@app.route("/scheme", methods=['POST', 'PUT'])
def scheme():
    body = request.get_json(force= True)
    example_array = body['input']
    print(example_array)
    data = pd.read_csv("schemes.csv")
    values= data.iloc[:, 1:].values
    count =[]
    for x in values:
        c=0
        for k in range(0,8):
            if x[k]==example_array[k]:
                c+=1
        count.append(c)
    count = np.array(count)
    ezAns = count.argsort()[-3:][::-1]

    schemes_list = [data['scheme_names'][ezAns[0]],data['scheme_names'][ezAns[1]],data['scheme_names'][ezAns[2]]]

    if request.method=="PUT":
           return jsonify({
        "1st most relevant": data['scheme_names'][ezAns[0]],
        "2nd most relevant": data['scheme_names'][ezAns[1]],
        "3rd most relevant ": data['scheme_names'][ezAns[2]]
    })
    elif request.method=="POST":
        f= open("schemes.json")

        data = json.load(f)

        output_string =""

        for i in range(0,20):
            if data['schemes'][i]['name'] in schemes_list:
                output_string+= "you can have a look at the scheme "
                output_string+=data['schemes'][i]['name']
                output_string+=" this scheme is about "
                output_string+=data['schemes'][i]['description']
                output_string+="\n"
                output_string+="\n"
        print(output_string)
        return jsonify({
            "message": output_string
        })




 



@app.route("/test", methods=['GET'])
def test():
    return jsonify({
        "message":"this is working "
    })



if __name__ == "__main__":
    app.run(debug=True, port=5050)
