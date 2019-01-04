# Tensorflow Tutorial

Within the project heavily the concept of [Transfer Learning](https://medium.com/the-official-integrate-ai-blog/transfer-learning-explained-7d275c1e34e2) within Deep Learning is used. 

The main API documentation is at: https://github.com/tensorflow/models/tree/master/research/object_detection.

## Set Up
Install [Miniconda](https://conda.io/miniconda.html) and create a new environment
```
conda create -n _yourenvname python=3.5
conda activate _yourenvname
pip install -r requirements.txt
```

If you want to train your own Object Detection model, you need to follow these steps:
1. Download the [tensorflow repository](https://github.com/tensorflow/models)
2. Most of the [required libraries](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) are included in the *requirements.txt* file, but some need to be done manually: 
   1. If you wish to install **tensorflow-gpu**, you are required to satisfy software and hardware [prerequisites](https://www.tensorflow.org/install/gpu) If so run:
   ```
   pip install tensorflow-gpu
   ```
   2.  Watch out for tricky installation errors on Windows. For example for *protoc* package:
         1. Download [Google Protobuf](https://github.com/google/protobuf) Windows v3.4.0 release *protoc-3.4.0-win32.zip*
         2. Extract the Protobuf download to Program Files, specifically: *C:\Program Files\protoc-3.4.0-win32*. Run in your conda env console:
         ```         
         cd path\to\models\research
         ```
         3. Execute the protobuf compile:
         ```
         for %f in (object_detection/protos/*.proto) do protoc.exe object_detection/protos/%f — python_out=.
         ```
         4. Now navigate to *models\research\object_detection\protos* and and verify the *.py* files were created successfully as a result of the compilation. (only the *.proto* files were there to begin with)
         5. cd to *\models\research\object_detection*. Open the jupyter notebook *object_detection_tutorial.ipynb*. Here you can play with the API.
   3. More popular errors can be found on [medium blog](https://medium.com/@rohitrpatil/how-to-use-tensorflow-object-detection-api-on-windows-102ec8097699). 



## Object Detection (OD)
Object Detection on a new dataset.

In the project, Faster RCNN with Inception v2 on Coco pretrained model was used. Good explanation of the [Faster RCNN network](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8).

**Train your own OD Network**:
Follow [tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) about OD Tensorflow API to get the basic understanding of the pipeline. Next, follow the steps:
1. Generate Training Data
    1. Annotate images as described in the [chapter 3](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#3-gather-and-label-pictures). I recommend using [labelImg software](https://github.com/tzutalin/labelImg) for manual annotation. 
    LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.
    2. Generate TFrecords by running 
        ```
        python generate_tfrecord.py --xml_input_path=relative_path_to_xmls --image_input_path=relative_path_to_images_folder --output_path=relative_path_to_output_folder --set_root 
        ```
        You will be provided with following output files:
        - *sketch_label_map.pbtxt* - label map used as an input in OD training,
        - *train.record* and *test.record* - training and validation dataset used in OD training,
        - *.csv files* - converted from annotation xmls, for your own insight.
        
        The output-path is optional, as a default path is provided.
        
        Examples:

        > **python generate_tfrecord.py** --xml_input_path=*C:/long_path/files/xmls/* --image_input_path=*C:/long_path/files/imgs/* 

        > **python generate_tfrecord.py** --set_root --xml_input_path=*files/xmls/* --image_input_path=*files/imgs/* --output_path=*files/output/*
    
2. Next step -- [config preperation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md) -- is to prepare *config* file. 

        
     [Download model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  matching to the config. If you wish to follow the same model, get the:

     > faster_rcnn_inception_v2_coco

     Finally, follow [running locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md) Tensorflow information, to run the training. Remember to follow the structure within *object_detection* folder from Tensorflow and check paths in *pipeline.config*.
            
2. **Deploy your OD model** 
    1. For analysis of model performance you may use tensorboard (see [running locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md))
    2. Export your model for inference. Tensorflow again provides [inctructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)
    3. After exporting the *frozen_inference_graph.pb*, you can run object detection on new images.
