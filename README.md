# Tensorflow Tutorial

Within the project heavily the concept of [Transfer Learning](https://medium.com/the-official-integrate-ai-blog/transfer-learning-explained-7d275c1e34e2) within Deep Learning is used. 

## Set Up
Install Miniconda and create a new environment
```
conda create -n _yourenvname python=3.5
conda activate _yourenvname
pip install -r requirements.txt
```

If you want to train your own Object Detection model, you need to follow these steps:
1. Download the [tensorflow repository](https://github.com/tensorflow/models) (version 1.10 used) 
2. Set up Tensorflow [prerequisites](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
   1. Watch out for tricky installation errors on Windows. For example for *protoc* package. More popular errors can be found on [medium blog](https://medium.com/@rohitrpatil/how-to-use-tensorflow-object-detection-api-on-windows-102ec8097699)


## Object Detection (OD)
Object Detection on new dataset, defined by rectangular localization and classification of objects.

In the project, Faster RCNN with Inception v2 on Coco pretrained model was used. Good explanation of the [Faster RCNN network](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8).

The project is equipped in following functionalities:
1. **Train your own OD Network**:
Follow [tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) about OD Tensorflow API to get the basic understanding of the pipeline. Next, follow the steps:
    1. Annotate images as described in the [chapter 3](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#3-gather-and-label-pictures). I used the following classes: *good* and *medium*, as previously described.
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
    
    3. Next step -- [config generation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md) -- is to prepare *config* file. Config file that was used in the project is provided under following project folder:
    
        > model_files/sketch_detection/configs

         You may use it, but make sure that the paths within the config file are correctly pointing to the right files. 
        
        [Download model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  matching to the config. If you wish to follow the same model, get the:
        
        > faster_rcnn_inception_v2_coco
        
        Finally, follow [running locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md) Tensorflow information, to run the training. Remember to follow the structure within *object_detection* folder from Tensorflow and check paths in *pipeline.config*.
            
2. **Deploy your OD model** 
    1. For analysis of model performance you may use tensorboard (see [running locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md))
    2. Export your model for inference. Tensorflow again provides [inctructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)
    3. After exporting the *frozen_inference_graph.pb*, you can run object detection on new images.
