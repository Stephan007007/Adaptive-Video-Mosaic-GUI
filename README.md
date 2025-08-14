# Adaptive-Video-Mosaic-GUI
Make mosaic videos and GIFs

![video_mosaic_1754764805](https://github.com/user-attachments/assets/d0c69af4-04f2-441e-ba0e-895e788101e4)


# First time
Download source (click "Code" and "Download zip"), extract the zip, install requirements, open adaptive_video_mosaic_gui.py

OR

Use the prebuilt version built with PyInstaller - open the .exe file.

# Requirements
If you haven't already, install Python. On Windows it's relatively simple - open cmd as administrator and type in: 

  winget install python

(Hopefully it should automatically add Python to PATH. Try the command below - if it fails, look up "Add Python to PATH on Windows")


Before opening adaptive_photo_mosaic_gui.py, you have to open cmd as administrator and type in:

  pip install pillow numpy moviepy imageio imageio-ffmpeg tqdm

# Usage
1. Download a variety of images or GIFs (the more colours the better) and put them inside a folder.

You can use e.g. https://data.caltech.edu/records/mzrjq-6wc02 (download and extract the caltech-101.zip file) inside which you can find 101_ObjectCategories.tar.gz - extract that and you have a variety of different sample folders.

2. Select the target image inside the app. Be sure its dimensions are large enough.

3. Select the sample folder ("Source folder") with the images/GIFs you downloaded.

4. Click "Scan source folder ..."

5. Select the folder where you want the finished mosaic image ("Output folder")

6. Click "Generate video/GIF"

# Other options
You should try all the options you find (like the "Half mosaic" with the slider) and see for yourself what they do :P
