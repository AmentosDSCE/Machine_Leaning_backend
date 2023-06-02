import os

print(os.listdir("New Plant Diseases Dataset(Augmented)/train"))
diseases = os.listdir("New Plant Diseases Dataset(Augmented)/train")

with open("disease.txt", 'w') as f:
    for disease in diseases:
        f.writelines(disease)
        f.write("\n")
        

print("done")