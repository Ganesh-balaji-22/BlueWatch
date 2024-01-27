import streamlit as st
from PIL import Image
import os
import imageio
import shutil 
from ultralytics import YOLO
import cv2

def convert_avi_to_mp4(avi_file, mp4_file):
    reader = imageio.get_reader(avi_file)
    fps = reader.get_meta_data()['fps']
    
    writer = imageio.get_writer(mp4_file, fps=fps, codec='libx264', quality=8)

    for frame in reader:
        writer.append_data(frame)

    writer.close()

def yolov8(img_path):
    import cv2
    model=YOLO("Model/best.pt")
    result=model.predict(img_path,save=True)
    return result

def main():
    st.title("Media Uploader")

    media_type = st.radio("Select media type", ["Image", "Video"])

    uploaded_file = st.file_uploader(f"Choose a {media_type.lower()} file", type=["jpg", "jpeg", "png", "mp4"])
    
    
    if uploaded_file is not None:
        file_name=uploaded_file.name.split('.')[0]
        file_extension = uploaded_file.name.split('.')[-1]

        # Save the uploaded file to the input folder
        input_folder = "input"
        os.makedirs(input_folder, exist_ok=True)
        file_path = os.path.join(input_folder, uploaded_file.name)

        with open(file_path, "wb") as file:
            file.write(uploaded_file.read())

        st.write(f"{media_type} saved to {file_path}")
        x=yolov8(file_path)
        output_path="runs/detect/predict/"+file_name+".avi"

        if media_type == "Image" and file_extension in ["jpg", "jpeg", "png"]:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Picture", use_column_width=True)
            st.write("")
            st.write("Classifying image... (add your classification logic here)")

        elif media_type == "Video" and file_extension == "mp4":
            # Display the uploaded video
            st.video(uploaded_file)
            st.write("")
            st.write("Analyzing video... (add your analysis logic here)")
        mp4_path="output/1.mp4"
        convert_avi_to_mp4(output_path,mp4_path)
        st.video(mp4_path)
        shutil.rmtree('runs/detect')
main()
