import streamlit as st
from PIL import Image
from joblib import load as jload
import numpy as np
import cv2
from pandas import DataFrame as pdf
import io
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


# loading segmentation model
PATH = '/Users/gouravchirkhare/PycharmProjects/Unet_segment/R-Forest/'
filename = PATH+"Saved_Model/Model2_int16.sav"
loaded_model = jload(filename)


st.set_page_config(layout="wide")
font = st.markdown("<link href='https://fonts.googleapis.com/css2?family=Josefin+Sans:wght@500&\
                    family=Kumbh+Sans&family=Ranchers&display=swap' rel='stylesheet'>", unsafe_allow_html=True)


# Header
st.markdown("<h1 style='text-align: center; color: white;'>REGIONAL VEGETATION COVER PREDICTION USING ML</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'> for \
            <br>Maharashtra Remote Sensing Applications Centre, Nagpur</p>", unsafe_allow_html=True)

# Desription
st.markdown("<h3 style='text-align: left; color: gray;'>Description</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; color: gray;'>Vegetation is an essential part of our ecosystem,\
            it also determines health of our planet. According to “The State of the World’s Forests 2020”\
            only 31% of the global land area is covered with vegetation.\
            This Web App helps to segment vegetation based of region.</p>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: blue;'>INPUT</h3>", unsafe_allow_html=True)


input_image = st.file_uploader("", type=['tif', 'jpg', 'png'])

if input_image is not None:
    img = Image.open(input_image)
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.radio('Select Model:', ("Randon Forest", ' '))
    with col2:
        submit_button = st.button("Start Segmentation")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<h3 style="text-align: center;">Original</h3>', unsafe_allow_html=True)
        st.image(img, width=600)

    if submit_button:
        with col4:
            def feature_extraction(img1):
                df = pdf()
                img2 = img1.reshape(-1)
                df['Original Image'] = img2

                # Generate Gabor features
                num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
                kernels = []
                # print("[3/4] Processing: [", end=" ")
                for theta in range(2):  # Define number of thetas
                    theta = theta / 4. * np.pi
                    for sigma in (1, 3):  # Sigma with 1 and 3
                        for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                            for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
                                gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                                #                print(gabor_label)
                                ksize = 9
                                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                                kernels.append(kernel)
                                # Now filter the image and add values to a new column
                                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                                filtered_img = fimg.reshape(-1)
                                df[gabor_label] = filtered_img.astype('int16')
                                num += 1  # Increment for gabor column label

                # CANNY EDGE
                edges = cv2.Canny(img, 100, 200)  # Image, min and max values
                edges1 = edges.reshape(-1)
                df['Canny Edge'] = edges1.astype('int16')  # Add column to original dataframe

                # ROBERTS EDGE
                edge_roberts = roberts(img)
                edge_roberts1 = edge_roberts.reshape(-1)
                df['Roberts'] = edge_roberts1.astype('float16')

                # SOBEL
                edge_sobel = sobel(img)
                edge_sobel1 = edge_sobel.reshape(-1)
                df['Sobel'] = edge_sobel1.astype('float16')

                # SCHARR
                edge_scharr = scharr(img)
                edge_scharr1 = edge_scharr.reshape(-1)
                df['Scharr'] = edge_scharr1.astype('float16')

                # PREWITT
                edge_prewitt = prewitt(img)
                edge_prewitt1 = edge_prewitt.reshape(-1)
                df['Prewitt'] = edge_prewitt1.astype('float16')

                # GAUSSIAN with sigma=3

                gaussian_img = nd.gaussian_filter(img, sigma=3)
                gaussian_img1 = gaussian_img.reshape(-1)
                df['Gaussian s3'] = gaussian_img1.astype('int16')

                # GAUSSIAN with sigma=7
                gaussian_img2 = nd.gaussian_filter(img, sigma=7)
                gaussian_img3 = gaussian_img2.reshape(-1)
                df['Gaussian s7'] = gaussian_img3.astype('int16')

                # MEDIAN with sigma=3
                median_img = nd.median_filter(img, size=3)
                median_img1 = median_img.reshape(-1)
                df['Median s3'] = median_img1.astype('int16')
                return df

            img = np.array(img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X = feature_extraction(img)
            result = loaded_model.predict(X)
            st.markdown('<h3 style="text-align: center;">Segmented</h3>', unsafe_allow_html=True)
            segmented = result.reshape(img.shape)
            plt.imshow(segmented, cmap='viridis')
            im_ratio = segmented.shape[0] / segmented.shape[1]
            plt.colorbar(label="Vegetation", orientation="vertical",fraction=0.047*im_ratio)
            segmented = io.BytesIO()
            plt.savefig(segmented, format='png')
            st.pyplot()

            btn = st.download_button(label="Download image", data=segmented, file_name='Segmented.png', mime="image/png")