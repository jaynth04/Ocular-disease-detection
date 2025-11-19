from django.shortcuts import render
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import os
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import io
import sys
# Create your views here.

def index(request):
    return render(request,'myapp/index.html')

def login(request):
    return render(request,'myapp/login.html')

def homepage(request):
    if request.method == "POST":
        username = request.POST['uname']
        password = request.POST['pwd']
        print(username, password)
        if username == 'admin' and password == 'admin':
            return render(request, 'myapp/homepage.html')
        else:
            return render(request, 'myapp/login.html')
    return render(request,'myapp/homepage.html')



data = pd.read_csv("D:/2023-24/finalprojects/glaumetric/glaumetric/dataset/full_df.csv")
data.head(20)

def has_condn(term,text):
    if term in text:
        return 1
    else:
        return 0

def process_dataset(data):

        # create 2 more columns labelling them whether right or left cataract
    data["left_cataract"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("cataract", x))
    data["right_cataract"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("cataract", x))

    data["LD"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy", x))
    data["RD"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy", x))

    data["LG"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("glaucoma", x))
    data["RG"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("glaucoma", x))

    data["LH"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive", x))
    data["RH"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive", x))

    data["LM"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("myopia", x))
    data["RM"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("myopia", x))

    data["LA"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration", x))
    data["RA"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration", x))

    data["LO"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("drusen", x))
    data["RO"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("drusen", x))

    # store the right/left cataract images ids in a array
    left_cataract_images = data.loc[(data.C == 1) & (data.left_cataract == 1)]["Left-Fundus"].values
    right_cataract_images = data.loc[(data.C == 1) & (data.right_cataract == 1)]["Right-Fundus"].values

    # store the left/right normal image ids in a array
    left_normal = data.loc[(data.C == 0) & (data["Left-Diagnostic Keywords"] == "normal fundus")][
        'Left-Fundus'].sample(350, random_state=42).values
    right_normal = data.loc[(data.C == 0) & (data["Right-Diagnostic Keywords"] == "normal fundus")][
        'Right-Fundus'].sample(350, random_state=42).values

    # store the left/right diabetes image ids
    left_diab = data.loc[(data.C == 0) & (data.LD == 1)]["Left-Fundus"].values
    right_diab = data.loc[(data.C == 0) & (data.RD == 1)]["Right-Fundus"].values

    # store the left/right glaucoma image ids
    left_glaucoma = data.loc[(data.C == 0) & (data.LG == 1)]["Left-Fundus"].values
    right_glaucoma = data.loc[(data.C == 0) & (data.RG == 1)]["Right-Fundus"].values

    # store the left/right diabetes image ids
    left_hyper = data.loc[(data.C == 0) & (data.LH == 1)]["Left-Fundus"].values
    right_hyper = data.loc[(data.C == 0) & (data.RH == 1)]["Right-Fundus"].values

    # store the left/right diabetes image ids
    left_myopia = data.loc[(data.C == 0) & (data.LM == 1)]["Left-Fundus"].values
    right_myopia = data.loc[(data.C == 0) & (data.RM == 1)]["Right-Fundus"].values

    # store the left/right diabetes image ids
    left_age = data.loc[(data.C == 0) & (data.LA == 1)]["Left-Fundus"].values
    right_age = data.loc[(data.C == 0) & (data.RA == 1)]["Right-Fundus"].values

    # store the left/right diabetes image ids
    left_other = data.loc[(data.C == 0) & (data.LO == 1)]["Left-Fundus"].values
    right_other = data.loc[(data.C == 0) & (data.RO == 1)]["Right-Fundus"].values

    normalones = np.concatenate((left_normal, right_normal), axis=0);
    cataractones = np.concatenate((left_cataract_images, right_cataract_images), axis=0);
    diabones = np.concatenate((left_diab, right_diab), axis=0);
    glaucoma = np.concatenate((left_glaucoma, right_glaucoma), axis=0);
    hyper = np.concatenate((left_hyper, right_hyper), axis=0);
    myopia = np.concatenate((left_myopia, right_myopia), axis=0);
    age = np.concatenate((left_age, right_age), axis=0);
    other = np.concatenate((left_other, right_other), axis=0);

    return normalones, cataractones, diabones, glaucoma, hyper, myopia, age, other;



def dataupload(request):
    normal, cataract, diab, glaucoma, hyper, myopia, age, other = process_dataset(data);

    print("Dataset stats::")
    print("Normal ::", len(normal))
    print("Cataract ::", len(cataract))
    print("Diabetes ::", len(diab))
    print("Glaucoma ::", len(glaucoma))
    print("Hypertension ::", len(hyper))
    print("Myopia ::", len(myopia))
    print("Age Issues ::", len(age))
    print("Other ::", len(other))
    print("*" * 50)
    print("Normal::\n")
    print(normal)
    print("*" * 50)
    print("*" * 50)
    print("Cataract::\n")
    print(cataract)
    print("*" * 50)
    print("*" * 50)
    print("Diabetes::\n")
    print(diab)
    print("*" * 50)
    print("*" * 50)
    print("Glaucoma::\n")
    print(glaucoma)
    print("*" * 50)
    print("*" * 50)
    print("HyperTension::\n")
    print(hyper)
    print("*" * 50)
    print("*" * 50)
    print("Myopia::\n")
    print(myopia)
    print("*" * 50)
    print("*" * 50)
    print("Age Related::\n")
    print(age)
    print("*" * 50)
    print("*" * 50)
    print("Other ::\n")
    print(other)
    print("*" * 50)
    content = {

        "Normal": len(normal),
        "Cataract": len(cataract),
        "Diabetes": len(diab),
        "Glaucoma": len(glaucoma),
        "Hypertension": len(hyper),
        "Myopia": len(myopia),
        "AgeIssues": len(age),
        "Other": len(other),

    }
    return render(request,'myapp/dataupload.html',content)

def createmodel(request):
    # Load saved model
    model = load_model('D:/2023-24/finalprojects/glaumetric/glaumetric/model/Ocular Conditions_model.h5')
    print(model.summary())
    # Model summary
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Print the model summary to the redirected stdout
    # model_lstm.summary()
    model.summary()
    # Get the model summary as a string
    summary_string = sys.stdout.getvalue()

    # Reset stdout to its original value
    sys.stdout = original_stdout

    # Now, `summary_string` contains the model summary
    print(summary_string)
    content1 = {
        'data': summary_string
    }
    return render(request,'myapp/createmodel.html',content1)

def predictdata(request):
    if request.method=='POST':
        imgname = request.POST['myFile']
        imgpath = 'D:/2023-24/finalprojects/glaumetric/glaumetric/Testing Images/'
        print(imgname)
        imgafile = imgpath + imgname
        # Load the model
        model = load_model('D:/2023-24/finalprojects/glaumetric/glaumetric/model/Ocular Conditions_model.h5')

        # Define classes
        classes = {
            0: {
                "label": "Normal",
                "description": "Normal eye health without any significant conditions.",
                "prevention": "Maintain a healthy lifestyle, regular eye check-ups.",
                "medicines": "Typically none for normal eye health.",
                "risk_factors": "Aging is a general risk factor, but otherwise, no specific risk factors for normal eye health.",
                "recommended_test": "Routine eye examination."
            },
            1: {
                "label": "Cataract",
                "description": "Clouding of the lens in the eye, leading to blurred vision.",
                "prevention": "Protect eyes from UV radiation, quit smoking, control diabetes.",
                "medicines": "Typically none for early-stage cataracts; surgery may be required in advanced cases.",
                "risk_factors": "Aging, diabetes, excessive sunlight exposure, smoking, certain medications.",
                "recommended_test": "Comprehensive eye exam including visual acuity test and dilated eye exam."
            },
            2: {
                "label": "Diabetes",
                "description": "Elevated blood sugar levels leading to various health complications, including diabetic retinopathy affecting the eyes.",
                "prevention": "Maintain healthy blood sugar levels, regular exercise, balanced diet, routine eye exams.",
                "medicines": "Insulin or oral medications to manage blood sugar levels; specific eye medications may be prescribed for diabetic retinopathy.",
                "risk_factors": "Poorly controlled blood sugar levels, obesity, family history of diabetes, sedentary lifestyle.",
                "recommended_test": "Comprehensive diabetic eye exam including dilated eye exam, visual acuity test, and intraocular pressure measurement."
            },
            3: {
                "label": "Glaucoma",
                "description": "Group of eye conditions resulting in optic nerve damage, often associated with elevated intraocular pressure.",
                "prevention": "Regular eye check-ups, avoiding smoking, maintaining healthy blood pressure.",
                "medicines": "Eye drops to reduce intraocular pressure, oral medications in some cases, surgery or laser therapy in advanced cases.",
                "risk_factors": "Elevated intraocular pressure, family history of glaucoma, older age, certain medical conditions like diabetes or hypertension.",
                "recommended_test": "Comprehensive eye exam including tonometry, visual field test, and examination of the optic nerve."
            },
            4: {
                "label": "Hypertension",
                "description": "High blood pressure, which can affect blood vessels in the eyes and lead to various eye conditions.",
                "prevention": "Maintain healthy blood pressure levels, balanced diet, regular exercise, limit alcohol intake.",
                "medicines": "Antihypertensive medications prescribed by a healthcare professional.",
                "risk_factors": "High salt intake, obesity, sedentary lifestyle, family history of hypertension.",
                "recommended_test": "Routine eye examination to assess blood vessel health."
            },
            5: {
                "label": "Myopia",
                "description": "Nearsightedness, where distant objects appear blurry.",
                "prevention": "Limit screen time, take breaks during close work, spend time outdoors.",
                "medicines": "Typically corrective lenses (glasses or contact lenses) for vision correction.",
                "risk_factors": "Genetics, prolonged close work (such as excessive screen time), lack of outdoor activities during childhood.",
                "recommended_test": "Comprehensive eye exam including refraction test."
            },
            6: {
                "label": "Age Issues",
                "description": "Various age-related changes affecting vision, such as presbyopia (loss of near vision), macular degeneration, etc.",
                "prevention": "Regular eye check-ups, maintain overall health, balanced diet rich in antioxidants.",
                "medicines": "Depends on the specific age-related eye condition; for example, certain supplements may be recommended for macular degeneration.",
                "risk_factors": "Aging is the primary risk factor, but genetics and lifestyle factors can also play a role in specific conditions.",
                "recommended_test": "Comprehensive eye examination to detect age-related changes and conditions."
            }
        }

        # Load the image
        input_img = imgafile
            # 'Testing Images/937_left.jpg'
        img = load_img(input_img, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predict the class
        pred = model.predict(img)
        pred_class = np.argmax(pred)

        # Print the prediction and its description
        if pred_class in classes:
            predicted_class_details = classes[pred_class]
            print(f"Predicted Class: {predicted_class_details['label']}")
            print(f"Description: {predicted_class_details['description']}")
            print(f"Prevention: {predicted_class_details['prevention']}")
            print(f"Recommended Medicines: {predicted_class_details['medicines']}")
            print(f"Risk Factors: {predicted_class_details['risk_factors']}")
            print(f"Recommended Test: {predicted_class_details['recommended_test']}")
        else:
            print("Invalid prediction")
        res1="Predicted Class:"+ str({predicted_class_details['label']})
        res2="Description:"+ str({predicted_class_details['description']})
        res3="Prevention:"+ str({predicted_class_details['prevention']})
        res4="Recommended Medicines:"+str( {predicted_class_details['medicines']})
        res5="Risk Factors:"+ str({predicted_class_details['risk_factors']})
        res6="Recommended Test:"+str( {predicted_class_details['recommended_test']})
        content={
            'data1':res1,
            'data2':res2,
            'data3':res3,
            'data4':res4,
            'data5':res5,
            'data6':res6,
        }
        return render(request, 'myapp/predictdata.html',content)

    return render(request,'myapp/predictdata.html')

def viewgraph(request):
    return render(request,'myapp/viewgraph.html')




