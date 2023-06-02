from flask import Flask, jsonify, request
import werkzeug
from PIL import Image
from torchvision import transforms
import torch


app = Flask(__name__)

@app.route('/hello', methods=['PUT'])
def home():
    image_file = request.files['image']
    filename = werkzeug.utils.secure_filename(image_file.filename)
    path = "uploaded_images/"+filename
    image_file.save("./uploaded_images/"+filename)

    img_random = Image.open(path).convert("RGB")

    model = torch.load("leaf_dense_18.pt")

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

    clean=[]
    for dis in disease:
        clean.append(dis.strip())

    print(clean)

    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0]*100

    _,indices = torch.sort(out, descending=True)

    top_5 = [(clean[idx], percentage[idx].item()) for idx in indices[0][:5]]

    return jsonify({
        "probability_1": top_5[0],
        "probability_2": top_5[1],
        "probability_3": top_5[2],
        "probability_4": top_5[3],
        "probability_5": top_5[4]
    })


if __name__=="__main__":
    app.run(debug=True, port=4040)

