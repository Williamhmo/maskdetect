import streamlit as st
from PIL import Image,ImageOps
import keras
import tensorflow as tf
import numpy as np
from keras_preprocessing.image import img_to_array
import streamlit.components.v1 as components

def uploadPhoto():
    image = st.file_uploader("Upload the file")
    if image is not None:
        st.success("Photo was successfully uploaded!")
        img = Image.open(image)
        st.image(img)
        return img

def takePhoto():
    image = st.camera_input("Please take a photo")
    if image is not None:
        st.success("Photo was successfully taken!")
        img = Image.open(image)
        st.image(img)
        return img
    
def detect(img):
    model=keras.models.load_model("model3_maskdataall.h5")
    img=ImageOps.fit(img,(250,250),Image.ANTIALIAS)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x /= 255.0
    output=model.predict(x)
    result=output.argmax(axis=-1)
    st.write(result)
    if result[0]==0:
        st.write('He/She is wearing mask.')
    elif result[0]==1:
        st.write('He/She is not wearing mask.')
        
def contact():
    contact_form="""<form action="https://formsubmit.co/hlaingminoo29917@gmail.com" method="POST">
    <input type="text" name="name" placeholder="Name "required>
    <input type="email" name="email" placeholder="Enter email address">
    <textarea id="subject" name="subject" placeholder="Your message here..." style="height:200px"></textarea>
    <input type="hidden" name="_captcha" value="false">
    <button type="submit">Send</button>
    </form>
    <style>
input[type=text],input[type=email], select, textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
  margin-top: 6px;
  margin-bottom: 16px;
  resize: vertical;
}
button[type=submit] 
{
  background-color: #D1E5F3;
  color: black;
  padding: 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
button[type=submit]:hover
{
  background-color: #2E34DA;
  color = white;
}
</style>    
    
    """
    st.markdown(contact_form,unsafe_allow_html=True)
        

def main():
    menu=['Home','Detect mask😷',' Contact developer📧']
    st.sidebar.image("images/homeimg.jpg")
    choice=st.sidebar.selectbox('Menu',menu)
    
    if choice == 'Home':
        st.header('Welcome to the app!')
        st.image("images/face-mask-emoji-concept-illustration_114360-6244.jpg")
        st.write('The issue of wearing face coverings in public\
            comes up frequently these days. A common sentiment \
            is, “If I am not personally at high risk for COVID-19, why should I wear a mask?” \
            I suspect this is why I see so many people in \
            public places who are not covering their nose and \
            mouth.So,I made this app to detect whether people are\
            wearing mask or not.')
        
    elif choice == 'Detect mask😷':
        st.title("Mask Detection")
        upload_option = st.sidebar.selectbox("Photo Options ",('Upload 📁','Shoot Photo📷'))
        if upload_option == 'Upload 📁':
            photo = uploadPhoto()
        else:
            photo = takePhoto()   
        if st.button("Detect"):
            if photo is None:
                st.warning("Please Upload or Shoot photo before classifying")
            else:
                st.success("Detecting photo in the model...\nPlease wait a second!")
                detect(photo)
    elif choice=='Contact':
        contact()           
    
    
    
    
if __name__ =='__main__':
    main()
