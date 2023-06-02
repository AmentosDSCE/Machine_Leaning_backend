import torch

from PIL import Image
from torchvision import transforms
import os
import json
model = torch.load("leaf_dense_18.pt")
# print(model.eval())


img_random = Image.open("archive/test/test/AppleCedarRust1.JPG").convert("RGB")

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

out = model(batch_img_random_tensor)

direct = "archive/New Plant Diseases Dataset(Augmented)/disease.txt"

with open(direct) as f:
    disease = f.readlines()

clean =[]

for dis in disease:
    clean.append(dis.strip())

_, index = torch.max(out,1)

percentage = torch.nn.functional.softmax(out, dim=1)[0]*100

_,indices = torch.sort(out, descending=True)

top_5 = [(clean[idx], percentage[idx].item()) for idx in indices[0][:5]]

# print(top_5[0][0])

list_of=[]

for i in range(0,5):
    list_of.append(top_5[i][0])

# print(list_of)

f= open("mappings.json")
data= json.load(f)

disss=[]


names=[]
causes=[]
fertilisers=[]

for i in range(0,38):
    disss.append(data['crops'][i]['name'])
    
    if data['crops'][i]['name'] in list_of:
        names.append(data['crops'][i]['name'])
        causes.append(data['crops'][i]['cause'])
        fertilisers.append(data['crops'][i]['fertilizsers'])

print(len(names))
print(len(causes))
print(len(fertilisers))


