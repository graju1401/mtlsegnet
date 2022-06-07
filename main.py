import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import transforms
import io
import re
import pandas as pd
import time
import uuid
import csv
from PIL import Image
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature

import requests
from pathlib import Path

model_path = "weights.pth"
path = Path(model_path)

if path.is_file():
    pass
else:
    url = "https://www.dropbox.com/s/37rtedwwdslz9w6/all_datasets.pth?dl=1"
    response = requests.get(url)
    open("weights.pth", "wb").write(response.content) 


    

def write_st_end():

    st.markdown("---")
    st.markdown(
        "Developed and Maintained by Gudhe Raju"
    )
    st.markdown(
        "[Arto Mannermaa Lab](https://uefconnect.uef.fi/henkilo/arto.mannermaa/) - [Institute of Clinical Medicine](https://uefconnect.uef.fi/tutkimusryhma/molecular-pathology-and-genetics-of-cancer/)"
    )
   
    st.markdown("Copyright (c) 2022 Gudhe Raju")

def show_st_table(df, st_col=None, hide_index=True):

    if hide_index:
        hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

    if st_col is None:
        st.table(df)
    else:
        st_col.table(df)


def create_st_button(link_text, link_url, hover_color="#e78ac3", st_col=None):

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    button_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: {hover_color};
                color: {hover_color};
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: {hover_color};
                color: white;
                }}
        </style> """

    html_str = f'<a href="{link_url}" target="_blank" id="{button_id}";>{link_text}</a><br></br>'

    if st_col is None:
        st.markdown(button_css + html_str, unsafe_allow_html=True)
    else:
        st_col.markdown(button_css + html_str, unsafe_allow_html=True)
        
def header():
    st.write("<h2 style='text-align: center;'>MTLSegNet: Area-based mammogram breast density estimation tool</h2>", unsafe_allow_html=True)


  
    st.markdown("**Created by Gudhe Raju**")
    st.markdown("**University of Eastern Finland**")

    st.markdown("---")

    st.markdown(
        """
        ### Summary
        **MTLSegNet** is a tool for predicting area-based mammogram breast density. **MTLSegNet** 
        is based on adaptive mutli-task learning approach. 
        Details of our work are 
        provided in the [*Scientific Reports*](https://www.nature.com/srep/)
        paper, **Area-based breast density estimation using adaptive multi-task learning**.
        We hope that researchers will use 
        *MTLSegNet* to gain novel insights into Mammorgram breast density and 
        breast cancer risk. 
        """
    )
    
    
def footer():

   
    st.markdown("---")

    left_info_col, right_info_col = st.columns(2)

    left_info_col.markdown(
        f"""
        ### Authors
        Please feel free to contact us with any issues, comments, or questions.
        ##### Gudhe Raju [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg?style=social&label=Follow%20%40graju1401)](https://twitter.com/graju1401)
        - Email:  <raju.gudhe@uef.fi> 
        - GitHub: https://github.com/graju1401
        ##### Hamid Behravan [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg?style=social&label=Follow%20%40Hamid_Behravan)](https://twitter.com/Hamid_Behravan)
        - Email: <hamid.behravan@uef.fi>
        - GitHub: https://github.com/Hamid_Behravan
        ##### Mazen Sudah
        ##### Hidemi Okuma
        ##### Ritva Vanninen
        ##### Arto Mannermaa
        ##### Veli-matti Kosma
        """,
        
        unsafe_allow_html=True,
    )

    right_info_col.markdown(
        """
        ### Funding
        - North Savo Cultutal fund
        - Funds to UEF
         """
    )

    right_info_col.markdown(
        """
        ### License
        Apache License 2.0
        """
    )

    write_st_end()

header()

MASK_COLORS = [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
]


def mask_to_rgba(mask, color="red"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: [description]
    """    
    assert(color in MASK_COLORS)
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)
    
    
#st.set_page_config(page_title="MTLSegNet", initial_sidebar_state="expanded")



st.sidebar.title("Menu")

activities = ["Breast Density (2D)"]
activity_choice = st.sidebar.selectbox("Select activity", activities)

Data_modality =["Mammogram"]
data_choice = st.sidebar.selectbox("Data modality", Data_modality)

files = ["Single file", "Multiple files"]
files_choice = st.sidebar.radio("Upload file(s)", files) 

html_tile = """
    <div style='background-color:tomato; padding:10px'>
    <h1 style="color:white; text-align:center;"> Breast percentage density Web application </h1>
    </div>
    """
@st.cache      
def image_tensor(img):
    torch_tensor = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.ToTensor()])

    image = torch_tensor(img)
    image = image.unsqueeze(0)
    return image


def breast_density_2d(image, model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model = nn.DataParallel(model.module)
    
    pred1, pred2 = model.module.predict(image)

    image = image[0].cpu().numpy().transpose(1, 2, 0)
    image = image[:, :, 0]

    pred1 = pred1[0].cpu().numpy().transpose(1, 2, 0)
    pred1 = pred1[:, :, 0]

    pred2 = pred2[0].cpu().numpy().transpose(1, 2, 0)
    pred2 = pred2[:, :, 0]

    breast_area = np.sum(np.array(pred1) == 1)
    dense_area = np.sum(np.array(pred2) == 1)
    density = (dense_area / breast_area) * 100
    
    
    return image, pred1, pred2, density

def canny_edges(image_array):
    edges = feature.canny(image_array,  sigma=3)
    return edges

with st.spinner(f"Loading selction:"):
    if files_choice == "Single file":
        if data_choice =="Mammogram":
            if activity_choice =="Breast Density (2D)":
                file = st.file_uploader("Please load the image", type=['png', 'jpg', 'jpeg'])
                if file is not None:
                    img = Image.open(file).convert('RGB')
                    #st.write('# Source Image:')
                    st.image(img, width=200)
                    image = image_tensor(img)
                    #model_path = 'all_datasets.pth'
                    
                    clicked = st.button('Estimate Density', key='single')
                    
                    
                    if clicked:
                        image,pred1, pred2, density = breast_density_2d(image, model_path)
                        
                        edges = canny_edges(pred1)
                        breast = mask_to_rgba(edges, color='red')
                        dense = mask_to_rgba(pred2, color='green')
                       
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write('##### Original mammogram')                            
                            st.image(image, width=220)
                        with col2:                            
                            st.write('##### Breast Segmentation')   
                            st.image(breast, width=220)
                        with col3:                            
                            st.write('##### Dense Segmentation')                          
                            st.image(dense, width=220)
                            
                        
                        st.success('#### Breast density (2D): ' + str(np.round(density, 2)))
                        
    if files_choice == "Multiple files":
        if data_choice =="Mammogram":
            if activity_choice =="Breast Density (2D)":
                
                files = st.file_uploader("Upload Mammograms", accept_multiple_files=True)
                st.markdown('''
                            <style>
                            .uploadedFile {display: none}
                                <style>''',
                                unsafe_allow_html=True)
                
                st.write("##### Successfully uploaded "+ str(len(files)) + " files")
                
                try:
                    os.remove("results.txt")
                except OSError:
                    pass
                
                col1, col2, col3 = st.columns([1,5,1])
                with col2:
                    clicked2 = st.button("Predict densitiy", key='run')
                    if clicked2:
                        my_bar = st.progress(0)
                        for percent_complete in range(100):
                            time.sleep(0.1)
                            my_bar.progress(percent_complete + 1)
                        if files is not None:
                            with open('results.txt', 'a') as result:
                                for file in files:
                                    img = Image.open(file).convert('RGB')
                                    image = image_tensor(img)
                                    #model_path = 'all_datasets.pth'
                                    image,pred1, pred2, density = breast_density_2d(image, model_path)
                                    print(file.name, str(np.round(density,2)), file=result)
                        st.info("Results saved in results.txt file")
                                    
                        df = pd.read_csv('results.txt', delimiter=' ', header=None)
                        df.columns = ['File', 'Density']
                        fig, ax = plt.subplots()
                        
                        sns.histplot(df['Density'], kde=True, bins=10, binwidth=2)
                        st.pyplot(fig)
                        
  

col1, col2 = st.columns(2)
with col1:
    st.sidebar.image("https://www.itewiki.fi/write/post_images/15455.png", width=250)
with col2:
    st.sidebar.image("https://www.psshp.fi/documents/7796362/0/kys_logo_en_footer_pieni.png/d07eb7e4-be53-4637-ab42-8c8073b8f8cf?t=1591363443438", width=250)

footer()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                    content:'(C) University of Eastern Finland' ; 
                    visibility: visible;
                    display: block;
                    position: relative;
                    #background-color: red;
                    padding: 5px;
                    top: 2px;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 















