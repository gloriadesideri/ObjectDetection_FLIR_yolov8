# Object detection using YOLOv8
This repository has the notebook for the training of YOLOv8 for object detection. To train this network we used the FLIR dataset. The data has been imported and annotated with roboflow. To run the notebook and train change the API key in the notebook and train the model. You can train with both the Infrared and RGB images. Then copy the weights in model_weights_streamlit and run 
streamlit run app.py --server.enableXsrfProtection=false. 
Note, once you download the data for training reorganize them as dataset/projectName/projectName/train (or val)