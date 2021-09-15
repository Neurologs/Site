import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import imutils
from PIL import Image
import os
import random
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_yaml
import h5py


# FUNCTIONS

@st.cache(allow_output_mutation=True)
def load_model_is_mri():
    model = load_model("model_is_mri")
    model.make_predict_function()
    return model


@st.cache(allow_output_mutation=True)
def load_model_classifier():
    model = load_model("model_classifier_brain_tumors")
    model.make_predict_function()
    return model


@st.cache(allow_output_mutation=True)
def alzheimer_model():
    yaml_file = open('model_alzheimer/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('model_alzheimer/adweights.h5')
    model.make_predict_function()
    return model
    
    
def random_scan_tumors(file):
    
    img = open(os.path.join(file, random.choice(os.listdir(file))), 'rb').read()
    rand_image = cv2.imdecode(np.fromstring(img, dtype='uint8'), cv2.IMREAD_ANYCOLOR)
    rand_image = cv2.cvtColor(rand_image, cv2.COLOR_BGR2RGB)
    resize_image = cv2.resize(rand_image, (350, 350))
    st.image(resize_image)
    rand_image = crop_image(rand_image)
    rand_image = cv2.resize(rand_image, (224, 224))
    rand_image = np.expand_dims(rand_image, axis=0)
    return rand_image


def crop_image(image):

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    img_thresh = cv2.threshold(img_gray, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)

    contours = cv2.findContours(
        img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return new_image


def random_scan_alzheimer(file):
    img = open(os.path.join(file, random.choice(os.listdir(file))), 'rb').read()
    rand_image = cv2.imdecode(np.fromstring(img, dtype='uint8'), cv2.IMREAD_ANYCOLOR)
    rand_image = cv2.cvtColor(rand_image, cv2.COLOR_BGR2RGB)
    return rand_image
       

def verif_is_mri(test_mri): 
    model = load_model_is_mri()
    verif = model.predict(test_mri)
    result_verif = np.argmax(verif)
    return result_verif


def analyse_mri_tumors(test_image):

    verif = load_model_is_mri()
    verif_ismri = verif.predict(test_image)
    verif = np.argmax(verif_ismri)

    if verif == 1:
        components.html("""<div><br>
                        <p style="background-color:#F63366; text-align:center; font-size:120%; color:white"><br>
                        Are you sure it is<br> a brain scan image ?<br><br>Please upload another file.<br><br></p>
                        </div>""", height=200)
    if verif == 0:
        loaded_model = load_model_classifier()
        prediction = loaded_model.predict(test_image)
        pred = round(np.max(prediction) * 100, 2)
        predict_proba = " [Probability prediction : " + str(pred) + " %] "
        class_prediction = np.argmax(prediction)

        if class_prediction == 0:
            st.write("""<p style="text-align:center; font-size:120%; color:#115764"><br><b>--  RESULT  --</b></p><br>
                     <p style="background-color:#F63366; text-align:center; font-size:130%; color:white"><br><b>Tumor detected</b><br><br></p>
                     <p style="text-align:center; font-size:110%; color:#115764">Classified as <span style="color:red"><b>glioma</b></span> tumor.</p><br>
                     """, unsafe_allow_html=True)
            st.write(predict_proba)
            st.write("")
          
        elif class_prediction == 1:
            st.write("""<p style="text-align:center; font-size:120%; color:#115764"><br><b>--  RESULT  --</b></p><br>
                     <p style="background-color:#F63366; text-align:center; font-size:130%; color:white"><br><b>Tumor detected</b><br><br></p>
                     <p style="text-align:center; font-size:110%; color:#115764">Classified as <span style="color:#1848b3"><b>meningioma</b></span> tumor.</p><br>
                     """, unsafe_allow_html=True) 
            st.write(predict_proba)
            st.write("")
           
        elif class_prediction == 2:
            st.write("""<p style="text-align:center; font-size:120%; color:#115764"><br><b>--  RESULT  --</b></p><br>
                      <p style="background-color:#9CFF8B; text-align:center; font-size:130%; color:#115764"><br><b>There is no tumor</b><br><br></p><br>
                     """, unsafe_allow_html=True)
            st.write(predict_proba)
            st.write("")

        elif class_prediction == 3:
            st.write("""<p style="text-align:center; font-size:120%; color:#115764"><br><b>--  RESULT  --</b></p><br>
                     <p style="background-color:#F63366; text-align:center; font-size:130%; color:white"><br><b>Tumor detected</b><br><br></p>
                     <p style="text-align:center; font-size:110%; color:#115764">Classified as <span style="color:#1dbf4f"><b>pituitary</b></span> tumor.</p><br>
                     """, unsafe_allow_html=True)
            st.write(predict_proba)
            st.write("")
  
        else:
            st.write("""<p style="text-align:center; font-size:120%; color:#115764"><br><b>--  RESULT  --</b></p><br>
                     <p style="text-align:center; font-size:100%; color:#115764"><b>Error. The algorithm failed to predict the outcome. Please try again.</p><br>
                     """, unsafe_allow_html=True)
          
def analyse_alzheimer(test_img):
      
    test_norm = cv2.normalize(test_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    test_resize = cv2.resize(test_norm, (208, 176))
    test_reshape = test_resize.reshape((1, test_resize.shape[0], test_resize.shape[1], test_resize.shape[2]))
    loaded_model = alzheimer_model()
    prediction = loaded_model.predict(test_reshape)
    pred = round(np.max(prediction) * 100, 2)
    predict_proba = " [Probability prediction : " + str(pred) + " %] "
    prediction = np.argmax(prediction, axis=1)
    
    if prediction == 0:
        st.write("""<p style="text-align:center; font-size:130%; color:#3b4a46"><br><b>--  RESULT  --</b></p><br>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Stage of Alzheimer's</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>Stage 3. Mild decline</b></p>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Diagnosis</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>Preclinical AD</b></p>
                 """, unsafe_allow_html=True) 
        st.write("")
        st.write(predict_proba)
       
        
    if prediction == 1:
        st.write("""<p style="text-align:center; font-size:130%; color:#3b4a46"><br><b>--  RESULT  --</b></p><br>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Stage of dementia</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>Stage 4. Moderate decline</b></p>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Diagnosis</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>Early-stage AD</b></p>
                 """, unsafe_allow_html=True) 
        st.write("")
        st.write(predict_proba)
     
        
    if prediction == 2:
         st.write("""<p style="text-align:center; font-size:130%; color:#3b4a46"><br><b>--  RESULT  --</b></p><br>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Stage of dementia</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>Stage 1. No impairment</b></p>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Diagnosis</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>No dementia</b></p>
                     """, unsafe_allow_html=True)
         st.write("")
         st.write(predict_proba)
        
    if prediction == 3:
        st.write("""<p style="text-align:center; font-size:130%; color:#3b4a46"><br><b>--  RESULT  --</b></p><br>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Stage of dementia</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>Stage 2. Very mild decline</b></p>
                 <p style="text-align:center; font-size:120%; color:#12A67F"><u>Diagnosis</u></p>
                 <p style="text-align:center; font-size:120%; color:#115764"><b>No dementia</b></p>
                 """, unsafe_allow_html=True) 
        st.write("")
        st.write(predict_proba)

# NAVIGATION BAR

with st.sidebar:
    logo = Image.open("images/logo_app.jpg")
    st.image(logo)
    signature = """<h2 style = "text_align:center; color:white; font-family:Gadugi;"><b>AI for brain health<b></h2>
                   <hr>"""
    st.markdown(signature, unsafe_allow_html=True)
    st.sidebar.text('')
    st.sidebar.text('')

def main():
    pages = {
        "Home": homepage,
        "Alzheimer's stages": page_alzheimer,
        "Brain tumors classifier": page_tumors,
        "Mini-Mental State Examination": MMSE_page,
        "Ongoing projects": ongoing_page}

    page = st.sidebar.selectbox("", tuple(pages.keys()))
    pages[page]()
    
    st.markdown(
        """<style>.st-c5.st-bc.st-c6.st-c7.st-c8.st-be.st-c9.st-ca.st-cb, .css-1d0tddh.e1wbw4rs0, .st-br .st-bq {font-weight: bold; color: #26ca9f;} </style>""", unsafe_allow_html=True)


# HOMEPAGE

def homepage():
    st.write("")
    st.write("")
    st.write("""<h1 style = "text-align:center; color:#12A67F; font-family:Gadugi;"><b>Brain disorders : a global epidemic</b><br><br></h1>""", unsafe_allow_html=True)
    
    image_map = "images/home_map.jpg"
    st.image(image_map, use_column_width=True)

    st.write("""<br><p style="font-size:105%; color:#3b4a46; font-weight:450; text-align:left">Brain disorders are a major public health problem and addressing their enormous social and economic burden is an absolute emergency. As well as a formidable challenge.<br><br>
    At Neurologs, we are convinced that Artificial Intelligence technologies could revolutionize the medicine.<br><br>
    Our teams works to provide healthcare professionals a comprehensive cloud-based Platform with AI-assisted diagnostics solutions, personalized treatment recommendation systems and tools for clinical research in the field of brain disorders.</p>""", unsafe_allow_html=True)
    st.write("")
      
    platform = "images/Platform_slide.jpg"
    st.image(platform, use_column_width=True)
    st.write("")
    
    st.write("""<p style = "color:#12A67F; font-family:Gadugi; font-weight:bold">Please use the navigation bar (on the left) to discover and test some of the diagnostics algorithms we are developing.</p>""", unsafe_allow_html=True)

    
# PAGE ALZHEIMER

def page_alzheimer():
     
    st.write("""<style>
                .st-bq, .st-br {color:#10515C} .st-ag {font-weight: bold} .st-af {font-size: 1rem} .st-ek, .st-el {padding-left: 6px; padding-top:5px}                                        
                .st-e4 .st-e5 .st-e6, .st-e7, .st-d7, .st-d8, .st_d9, st-da {border-color:yellow}
                .st-dx.st-b2.st-bp.st-dy.st-dz.st-e0.st-e1.st-e2.st-bc.st-bk.st-bl.st-bm.st-bn.st-bg.st-bh.st-bi.st-bj.st-e3.st-e4.st-e5.st-e6.st-av.st-aw.st-ax.st-ay.st-e7.st-cc.st-e8.st-e9.st-ea.st-eb.st-ec.st-ed.st-ee.st-c6.st-ef.st-eg
                 {border-color: yellow; margin-left: 19px; margin-right: 14px; height: 30px; width: 30px; border-width: 3px; transition-property: none;}
                .st-e0 {width: 30px} .st-e1 {height: 30px} .st-fo, .st-en {background-color: #12A67F} .st-eg {margin-left: 19px; margin-right: 14px;} 
                #.st-el, .st-ep, .st-ey, .st-ew {background-color:#ffe4e1}
                .css-9ycgxx.exg6vvm3 {color:white}
                .css-113fe4q.euu6i2w0 {color:gray}
                .css-1op0mqd.exg6vvm2 {color:yellow} 
                .css-1qpos38 {background-color:#12A67F; color:white; font-size:18px} 
                .row-widget.stButton {text-align:center}
                label {display: inline-flex}
                .uploadedFileData.css-1l4firl.exg6vvm8 {color:#10515C}
                .css-rncmk8 > * {margin:0px}
                 p {text-align:center; font_size:18px; color:#3b4a46; font-weight:bold}
                .css-1p9wfzo {color: #0c113899; text-align: left; margin-top: 1.5em; font-weight:400; font-size:85%}
                .css-rncmk8 > * {margin:0px}
                .css-rncmk8 {display: flex; flex-flow : row wrap; justify-content: space-around; width:698px;} 
                .css-1w0ubgf.e1tzin5v2 {background-color: lightyellow; height: 350px; margin:0px}
                @media screen and (max-width: 450px){.css-rncmk8{flex-flow: column wrap;}} 
                </style>""", unsafe_allow_html=True)
    
    st.write("")
    title = """<h1 style = "text-align:center; color:#12A67F; font-family:Gadugi;"><b>Alzheimer's stages</b></h1><br>"""
    st.markdown(title, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("""<p style="font-size:105%; color:#3b4a46; font-weight:450; text-align:left">In 2020, there are over 50 million people worldwide living with dementia. 
             With population ageing and growth, this number will almost double every 20 years, reaching 82 million in 2030 and 152 million in 2050. 
             Moreover, the economic impact of dementia is already enormous, with total estimated worldwilde costs greater than US$ 1 trillion every year.<br><br>
             Alzheimer’s disease (AD) is the most common form of dementia accounting for an estimated 60% to 80% of cases. Since there is no effective Alzheimer’s treatment to date, 
             early diagnosis and measures to reduce or prevent further progression of disease are crucial.<hr></p>
             <p style="font-size:105%; color:#3b4a46; font-weight:450; text-align:left">AD is characterized by the formation of amyloid plaques and neurofibrillary tangles, resulting from accumulation of abnormal amounts of amyloid-β [Aβ] and hyperphosphorylated tau proteins, 
             respectively outside and inside brain neurons. These plaques and tangles cause various synaptic, neuronal and axonal damage which lead to progressive cognitive and functional decline until death.<br><br>
             Multiple imaging techniques with associated biomarkers are used to identify and monitor AD. For example, cerebral atrophy, which is considered a characteristic feature of neurodegeneration, 
             can be visualized with structural MRI. Indeed, degrees and rate of brain volume loss (especially hippocampus and medial temporal lobe volumes) as well as shrinkage of cerebral cortex and ventricular enlargement 
             correlate closely with changes in cognitive performance, supporting their validity as biomarkers of AD.</p>""", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    image_alzheimer = Image.open("images/brain_alzheimer.jpg")
    st.image(image_alzheimer, caption = "van Oostveen, de Lange ; “Imaging Techniques in Alzheimer’s Disease: A Review of Applications in Early Diagnosis and Longitudinal Monitoring” in International Journal of Molecular Sciences. 2021; 22(4):2110.")
    st.write("""<p style="font-size:85%; color:#0c113899; text-align:left; font-weight:400">[AD leads to decreased hippocampal volume, shrinkage of cerebral cortex and ventricle enlargement. MTA: medial temporal lobe atrophy; MTA = 0: no atrophy in medial temporal lobe; MTA = 4: severe volume loss of hippocampus]</p>""", unsafe_allow_html=True)
    st.write("""<hr><p style="font-size:105%; color:#3b4a46; font-weight:450; text-align:left">We propose a deep convolutional neural network for early-stage Alzheimer's Disease diagnosis using brain MRI data analysis. Based on VGG16 (OxfordNet) architecture, the model achieves accuracy higher than 98%.<br><br>
                The data used to design the algorithm consists of 6400 preprocessed MRI (axial slices T1 weighted) categorized as non-demented, very mildly demented, mildly demented and moderately demented. Labels are based on the level of neurological degeneration as defined by the Global Deterioration Scale (or “Reisberg Scale”).<br><br>
                Please upload a brain MRI or choose a random image to determine the stage of Alzheimer's disease the patient is experiencing.<br><br>
                <u>Note</u> : you can also estimate the cognitive decline and eventually confirm the diagnosis of Alzheimer’s by administering the patient the Mini-mental State Examination.<br><br></p>""", unsafe_allow_html=True)
    st.write("")

    uploaded_file = st.file_uploader('')
    st.write("")
    random_check = st.checkbox("OR Select random image")
    st.write("")
    st.write("")
    generate_pred = st.button("Prediction")
    st.write("")
    st.write("")
    st.write("")

    if uploaded_file is not None and generate_pred:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes, cv2.IMREAD_ANYCOLOR)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) 
        
        with st.spinner('ANALYSIS IN PROGRESS'):
            st.write("")
            st.write("")
            col1, col2 = st.beta_columns(2)
          
            with col1:
                test_img = cv2.resize(test_img, (350,350))
                st.image(test_img)
                            
            with col2:
                    test_mri = cv2.resize(test_img, (224,224))
                    test_mri = np.expand_dims(test_mri, axis=0) 
                    verif_is_mri(test_mri)
                    if verif_is_mri(test_mri) == 1:
                        components.html("""<div><br><p style="background-color:#F63366; text-align:center; font-size:120%; color:white"><br>
                        Are you sure it is<br> a brain scan image ?<br><br>Please upload another file.<br><br></p></div>""", height=200) 
                    if verif_is_mri(test_mri) == 0:
                        analyse_alzheimer(test_img)
                               
    if random_check and generate_pred:
        
        file_test_alzheimer = "alzheimer_random"
        rand_image = random_scan_alzheimer(file_test_alzheimer)
        
        with st.spinner('ANALYSIS IN PROGRESS'):
            st.write("")
            st.write("")
            col1, col2 = st.beta_columns(2)
            with col1:
                rand_image = cv2.resize(rand_image, (350,350))
                st.image(rand_image)
            
            with col2:
                resize_image = cv2.resize(rand_image, (224, 224))
                resize_image = np.expand_dims(resize_image, axis=0) 
                analyse_alzheimer(rand_image)


# PAGE BRAIN TUMORS

def page_tumors():
        
    st.write("""<style>
        .st-bq, .st-br {color:#10515C} .st-ag {font-weight: bold} .st-af {font-size: 1rem} .st-ek, .st-el {padding-left: 5px; padding-top:5px}                                        
        .st-e4 .st-e5 .st-e6, .st-e7, .st-d7, .st-d8, .st_d9, st-da {border-color:yellow}
        .st-e0 {width: 30px} .st-e1 {height: 30px} .st-fo, .st-en {background-color: #12A67F} .st-eg {margin-left: 19px; margin-right: 14px;} 
        #.st-el, .st-ep, .st-ey, .st-ew {background-color:#ffe4e1}
        .css-9ycgxx.exg6vvm3 {color:white}
        .css-113fe4q.euu6i2w0 {color:gray}
        .css-1op0mqd.exg6vvm2 {color:yellow} 
        .css-1qpos38 {background-color:#12A67F; color:white; font-size:18px} 
        .row-widget.stButton{text-align:center}
        .uploadedFileData.css-1l4firl.exg6vvm8{color:#10515C} 
         p {font_size:100%; text-align:center; font-weight:bold; color:#3b4a46}
         label {display: inline-flex}
        .css-rncmk8 > * {margin:0px}
        .css-rncmk8 {display: flex; flex-flow : row wrap; justify-content: space-around; width:698px;} 
        .css-1w0ubgf.e1tzin5v2 {background-color: lightyellow; height: 350px; margin:0px}
        @media screen and (max-width: 450px){.css-rncmk8{flex-flow: column wrap;}} 
        </style>""", unsafe_allow_html=True)

    st.write("")
    title = """<h1 style = "text_align:center; color:#12A67F; font-family:Gadugi;"><b>BRAIN TUMORS</b><br><br></h1>"""
    st.markdown(title, unsafe_allow_html=True)
    st.write("")
    
    st.write("")
    st.write("""<h2 style = "text_align:left; color:#12A67F; font-family:Gadugi;"><b>1. Tumors Classifier</b><br><br></h2>""", unsafe_allow_html=True)
    st.write("")
    st.image("images/tumors_pres.jpg", use_column_width='auto')
           
    tumors_text = """<p style="font-size:105%; color:#3b4a46; font-weight:450;"><br>
      We have designed a deep convolutional neural network aimed to detect and classify the most common primary brain tumors : glioma, meningioma and pituitary tumors.<br><br>
      The model has been trained on 2870 brain MRI (T1, T2 and FLAIR images) and tested on 395 MRI, manually labeled and verified by medical doctors.<br><br>
      The model's accuracy is up to 95%.<br><br>
      Please test our diagnosis tool : upload a brain scan or choose a random image and obtain the result in less than a minute !</p>"""

    st.markdown(tumors_text, unsafe_allow_html=True)

    st.write("")
    uploaded_file = st.file_uploader('')
    st.write("")
    random_check = st.checkbox("OR Select random image")
    st.write("")
    st.write("")
    generate_pred = st.button("Prediction")
    st.write("")
    st.write("")


    if uploaded_file is not None:
        
        with st.spinner('ANALYSIS IN PROGRESS'):
            st.write("")
            st.write("")
        
            col1, col2 = st.beta_columns(2)
            
            with col1:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                up_image = cv2.imdecode(file_bytes, cv2.IMREAD_ANYCOLOR)
                up_image = cv2.cvtColor(up_image, cv2.COLOR_BGR2RGB)
                resize_image = cv2.resize(up_image, (350, 350))
                st.image(resize_image)
                test_img = cv2.resize(up_image, (224, 224))
                cropped_image = crop_image(up_image)
                up_image = np.expand_dims(test_img, axis=0)
                
            if generate_pred:
                with col2:
                    verif_is_mri(up_image)
                    if verif_is_mri(up_image) == 1:
                        components.html("""<div><br><p style="background-color:#F63366; text-align:center; font-size:120%; color:white"><br>
                        Are you sure it is<br> a brain scan image ?<br><br>Please upload another file.<br><br></p></div>""", height=200) 
                    if verif_is_mri(up_image) == 0:
                        analyse_mri_tumors(up_image)

    if random_check and generate_pred:
        with st.spinner('ANALYSIS IN PROGRESS'):
            st.write("")
            st.write("")
            col1, col2 = st.beta_columns(2)
            with col1:
                file_test_tumors = "tumors_random"
                rand_image = random_scan_tumors(file_test_tumors)

            with col2:
                analyse_mri_tumors(rand_image)
                     
# PAGE MMSE

def MMSE_page():
    
    st.write("")
    st.write("")
    title = """<h1 style = "text_align:center; color:#12A67F; font-family:Gadugi;"><b>Mini-Mental State Examination</b></h1><br>"""
    st.markdown(title, unsafe_allow_html=True)
    
    st.markdown("""<style>
                p {color:#126F90; font-weight:bold; text-align:justify;}
                li {color:#3b4a46; text-align:justify; font-weight:450}
                .st-bq{color:#3A4044}
                .time{color:#bb0f6e}
                div.row-widget.stRadio > div{flex-direction:row;} .st-bg.st-bj.st-bk.st-bl.st-bm.st-bn.st-az.st-b4.st-bo.st-bp.st-bq.st-br.st-bs.st-bt.st-bu.st-bv.st-bw.st-bx.st-b2.st-by{background-color:#E2005B}
                .css-1qpos38 {text-align:center; background-color:#12A67F; color:white; font-size:150%; height:60px; margin-top:20px;} 
                code, kbd {font-size: 120%; background-color:white; color: #bb0f6e;}
                .css-1w0ubgf {background-color: lightyellow; height:250px}
                .css-rncmk8.e1tzin5v0 p {text-align: center}
                </style>""", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    st.write("""<p style="color:#3b4a46; text-align:justify; font-weight:450">
             Originally introduced by Folstein et <i>al.</i> in 1975, The Mini–Mental State Examination (MMSE) is a 30-point questionnaire 
             used extensively as a screening device for cognitive impairment and as a diagnostic adjunct for assessing 
             Alzheimer’s disease or other types of dementias (Parkinson's disease, Lewy bodies, vascular dementia, etc.)<br><br>
             Administration of the test takes between 5 and 10 minutes and measures the following cognitive functions:</p>
             <ul>
             <li> Orientation to time and place</li>
             <li> Short-term memory</li>
             <li> Attention and ability to solve problems</li>
             <li> Language use and comprehension</li>
             <li> Basic motor skills</li>
             </ul>
             <p style="color:#3b4a46; text-align:justify; font-weight:450">Any score of 24 or more (out of 30) indicates a normal cognition. Below this, scores can indicate mild (19–23 points), moderate (10–18 points) or severe (≤9 points) cognitive impairment.
             </p>""", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    not_applicable = 30
    
    st.write("""<h2 style="color:#051D31;"><b>1. Orientation to Time</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 10 seconds for each reply.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("A) What year is this ?")
    year = st.radio("",('Correct', 'Incorrect', 'Not applicable'), key='year')
    st.write("B) What season is this ?") 
    season = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='season')
    st.write("C) What month is this ?")
    month = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='month')
    st.write("D) What is today’s date ?") 
    date = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='date')
    st.write("E) What day of the week is this ?") 
    day = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='day')
    orientation_time_score = 0
    for answer in [year, season, month, date, day]:
        if answer == 'Correct':
            orientation_time_score += 1
        elif answer == 'Not applicable':
            not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>2. Orientation to Place</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 10 seconds for each reply.</p>""", unsafe_allow_html=True)
    st.write("")        
    st.write("A) What country are we in ?")
    country = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='country')
    st.write("B) What province are we in ?")
    province = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='province')
    st.write("C) What city/town are we in ?")
    city = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='city')
    st.write("D) What is the address/name of this place ?")
    address = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='address')
    st.write("E) What room/floor are we in/on ?")
    building = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='building')
    orientation_place_score = 0
    for answer in [country, province, city, address, building]:
        if answer == 'Correct':
            orientation_place_score += 1
        elif answer == 'Not applicable':
            not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>3. Registration</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 20 seconds.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><u>Say</u> : I am going to name three objects. When I am finished, I want you to repeat them. Remember what they
    are because I am going to ask you to name them again in a few minutes.<br> 
    <span style="color:#3A4044";>[<i>Say the following words slowly at approximately one-second intervals</i> :]</span> Ball / Car / Man<br>
    Please repeat the three items for me.<br>
    <span style="color:#3A4044";>[<i>Score one point for each correct reply on the first attempt. If the patient did not repeat all three, repeat until they are learned or up to a maximum of five times.
    But only score first attempt.</i>]</span></p>""", unsafe_allow_html=True)
    st.write("""<p style="color:#12A67F;">Number of correct replies :</p>""", unsafe_allow_html=True)
    three_words = st.radio('', [0, 1, 2, 3, 'Not applicable'])
    if three_words in range(4):
        learning_score = three_words
    if three_words == 'Not applicable':
        not_applicable -= 1
        learning_score = 0
    
    st.write("""<h2 style="color:#051D31;"><b><br>4. Attention</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 30 seconds.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><u>Say</u> : I would like you to count backward from 100 by sevens.<br>
    <span style="color:#3A4044";>[93, 86, 79, 72, 65, …]</span></p>""", unsafe_allow_html=True) 
    st.write("""<p style="color:#12A67F;">Number of correct answers :</p>""", unsafe_allow_html=True)      
    count = st.radio('', [0, 1, 2, 3, 4, 5, 'Not applicable'], key="count")
    if count in range(6):
        count_score = count
    if count == 'Not applicable':
        count_score = 0
        not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>5. Recall</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 10 seconds.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><u>Say</u> : Earlier I told you the names of three things. Can you tell me what those were?<br>
    <span style="color:#3A4044";>[<i>Items may be repeated in any order.</i>]</span></p>""", unsafe_allow_html=True)
    st.write("""<p style="color:#12A67F;">Number of correct answers :</p>""", unsafe_allow_html=True)  
    recall = st.radio('', [0, 1, 2, 3, 'Not applicable'], key="recall")
    if recall in range(4):
        recall_score = recall
    if recall == 'Not applicable':
        recall_score = 0
        not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>6. Naming</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 10 seconds for each reply.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p>A) <u>Say</u> : What is this called?<br><span style="color:#3A4044";>[<i>Show wristwatch.</i>]</span><br>
    <span style="color:#3A4044";>[<i>Accept “wristwatch” or “watch” ; do not accept “clock” or “time”, etc.</i>]</p>""", unsafe_allow_html=True)
    wristwatch = st.radio("",('Correct', 'Incorrect', 'Not applicable'), key='wristwatch')
    st.write("""<p>B) <u>Say</u> : What is this called?<br><span style="color:#3A4044";>[<i>Show pencil.</i>]</span><br>
    <span style="color:#3A4044";>[<i>Accept “pencil” only.</i>]</p>""", unsafe_allow_html=True)
    pencil = st.radio("",('Correct', 'Incorrect', 'Not applicable'), key='pencil')
    naming_score = 0
    for answer in [wristwatch, pencil]:
        if answer == 'Correct':
            naming_score += 1
        elif answer == 'Not applicable':
            not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>7. Repetition</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 10 seconds.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><u>Say</u> I would like you to repeat a phrase after me : "No ifs, ands or buts"<br>
    <span style="color:#3A4044";>[<i>Repetition must be exact.</i>]</span></p>""", unsafe_allow_html=True)
    repetition = st.radio("",('Correct', 'Incorrect', 'Not applicable'), key='repetition')
    repetition_score = 0
    if repetition == 'Correct':
        repetition_score = 1
    if repetition == 'Not applicable': 
        not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>8. Comprehension</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 30 seconds.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><span style="color:#3A4044";>[<i>Ask the patient if he is right or left handed. Take a piece of paper and hold it up in front of the patient.</i>]</span><br>
    <u>Say</u> : Take this paper in your right/left hand (whichever is non-dominant), fold the paper in half once with both hands and put the paper down on the floor.</p>""", unsafe_allow_html=True)
    st.write("""<p style="color:#12A67F;">Number of instructions executed correctly :</p>""", unsafe_allow_html=True)  
    action = st.radio("", [0, 1, 2, 3, 'Not applicable'], key='action')
    if action in range(4):
        action_score = action 
    if action == 'Not applicable':
        action_score = 0
        not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>9. Reading</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 20 seconds.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><span style="color:#3A4044";>[<i>Hand patient the card with ’CLOSE YOUR EYES’ on it.</i>]</span><br>
    <u>Say</u> : Please read the words on this card and then do what it says.<br>
    <span style="color:#3A4044";>[<i>Repeat instructions up to three times if necessary.</i>]</span></p>""", unsafe_allow_html=True)
    eyes = st.radio("", ("The patient closes his/her eyes", "The patient doesn't close his/her eyes", "Not applicable"), key='eyes')
    eyes_score = 0
    if eyes == "The patient closes his/her eyes":
        eyes_score = 1
    if eyes == 'Not applicable':
        not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>10. Writing</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 30 seconds.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><span style="color:#3A4044";>[<i>Hand patient a pencil and paper.</i>]</span><br>
    <u>Say</u> : Write any complete sentence on that piece of paper.<br>       
    <span style="color:#3A4044";>[<i>The sentence must contain a subject, verb and object, and make sense. Ignore spelling errors.</i>]</span></p>""", unsafe_allow_html=True)      
    sentence = st.radio("", ('Correct', 'Incorrect', 'Not applicable'), key='sentence')
    sentence_score = 0
    if sentence == "Correct":
        sentence_score = 1
    if sentence == 'Not applicable':
        not_applicable -= 1
    
    st.write("""<h2 style="color:#051D31;"><b><br>11. Drawing</b></h2><hr>""", unsafe_allow_html=True)
    st.write("""<p class="time"><u>Time</u> : 1 minute maximum.</p>""", unsafe_allow_html=True)
    st.write("")
    st.write("""<p><span style="color:#3A4044";>[<i>Place design, eraser and pencil in front of the patient.</i>]</span><br>
    <u>Say</u> : Copy this design please.<br><span style="color:#3A4044";>[<i>Allow multiple tries. Wait until the person is finished and hands it back. The patient must have drawn a four-sided figure between two five-sided figures.</i>]</span></p>""", unsafe_allow_html=True)
    figure_photo = Image.open("images/figure_mmse.jpg")
    st.image(figure_photo)
    figure = st.radio("",('Correct', 'Incorrect', 'Not applicable'), key='figure')
    figure_score = 0
    if figure == 'Correct':
        figure_score = 1
    if figure == 'Not applicable':
        not_applicable -= 1
       
    total_score = orientation_time_score + orientation_place_score + learning_score + count_score + recall_score + naming_score + repetition_score + eyes_score + sentence_score + action_score + figure_score
    
    if not_applicable > 0:
        adjusted_score = round(total_score * 30 / not_applicable)
    
    if adjusted_score >= 24:
        result = """<p style="color:#051D31; font-size:110%; font-weight:bold; text-align:center;">Normal cognition : <br>no dementia<br><br><br></p>"""
    elif adjusted_score < 24 and adjusted_score >= 19:
        result = """<p style="color:#051D31; font-size:110%; font-weight:bold; text-align:center;">Mild<br><br><br></p>"""
    elif adjusted_score < 19 and adjusted_score >= 10:
        result = """<p style="color:#051D31; font-size:110%; font-weight:bold; text-align:center;">Moderate<br><br><br></p>"""
    elif adjusted_score < 10:
        result = """<p style="color:#051D31; font-size:110%; font-weight:bold; text-align:center;">Severe<br><br><br></p>"""
    else: 
        result = "Error. Please retry."
    
    st.write("")
    st.write("")
    st.write("")
    
    adjusted = st.button("Obtain score")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    if adjusted:
        
        col1, col2 = st.beta_columns(2)
        
        with col1:
            st.write("")
            st.write("""<h2 style="color:#12A67F; font-size:120%; font-weight:bold; text-align:center;">Total Score<hr></h2>""", unsafe_allow_html=True) 
            st.write(adjusted_score)
            st.write("")
            
        with col2:
            st.write("")
            st.write("""<h2 style="color:#12A67F; font-size:120%; font-weight:bold; text-align:center;">Level of dementia<hr></h2>""", unsafe_allow_html=True) 
            st.write(result, unsafe_allow_html=True)
            st.write("")
        

# PAGE ONGOING PROJECTS

def ongoing_page():
    st.write("")
    st.write("")
    st.write("""<h1 style = "text_align:center; color:#12A67F; font-family:Gadugi;"><b>Ongoing projects</b></h1><br>""", unsafe_allow_html=True)
    st.write("")
    st.write("")
    image_projects = "images/projects.jpg"
    st.image(image_projects, use_column_width=True)

    
# CALL MAIN FUNCTION (MENU)

if __name__ == "__main__":
    main()
