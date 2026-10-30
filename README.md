# Your first steps in Computer Vision on AWS

## [Part 1] ASL dataset + preprocessing 

⚠️ **For the detailed step-by-step explanation, read the [blog post on my website](https://daryapetrashka.com/blog/cv-asl-part-1/), or on [Dev.to](https://dev.to/dashapetr/part-1-your-first-steps-in-computer-vision-on-aws-asl-dataset-preprocessing-4h5p), or on [AWS Builder Center](https://builder.aws.com/content/3KdiKzuB4nl91kM0DfQOvjgc81k/part-1-your-first-steps-in-computer-vision-on-aws-asl-dataset-preprocessing).**

This is the first part of the series "Your first steps in Computer Vision on AWS". If you are just starting, this tutorial will guide you step by step from the initial dataset to a deployed app.

### 1. Dataset

The [ASL Alphabet dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) comprises 87,000 training images and 29 test images. Images are 200x200 pixels and in 3-channel RGB format. The dataset has 29 classes: 26 English letters (A-Z) and 3 control signs (SPACE, DELETE, NOTHING). As you can see, it is perfectly balanced: 3,000 images per class in the training set.

![Some examples of training images](images/training_images_examples.png)

### 2. Uploading to S3

For the speed, use AWS CLI. ([How to install/update AWS CLI?](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))

Then run the commands to configure the appropriate concurrency (considering you have the default profile):

```
aws configure set default.s3.max_concurrent_requests 50
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB
```  

Unzip the downloaded archive. To copy the files: 

```
cd archive/asl_alphabet_train
aws s3 sync ./asl_alphabet_train s3://your-bucket/raw/asl_alphabet_train/
```

⚠️ Note: if any file failed to be uploaded, use this command to find which one:

```aws s3 sync ./asl_alphabet_train s3://your-bucket/raw/asl_alphabet_train/ --dryrun```

### 3. Create SageMaker notebook instance

Since we are not training any model yet (only processing 200x200 images one at a time), a bare CPU SageMaker notebook instance with additional disk space would be enough.
I created an `ml.t3.medium` instance with a 50 GB EBS volume:

⚠️ **Don't forget to stop and terminate your instance once the data is processed and uploaded to S3.** 

### 4. Let's preprocess the data: run Jupyter notebook

Once the notebook instance is ready, you can **Open Jupyter / Open JupyterLab**.

Upload the ```data_preprocessing.ipynb``` Jupyter notebook [from the current GitHub repo](https://github.com/dashapetr/Computer_Vision_ASL_AWS/blob/main/sagemaker_notebooks/data_preprocessing.ipynb).

Run cell by cell. You might need to restart the Kernel after running pip install of libraries. (first cell)

⚠️ Don't forget to replace the placeholder and put your S3 bucket name:
```BUCKET = "YOUR-BUCKET-NAME"```

#### What it does (high-level):

1. Downloads the data from S3
2. Creates the train/validation/test split
3. Separately applies Gaussian filtering and Canny edge detection to each image
4. Uploads the preprocessed images to S3 

After uploading data to S3, your bucket should have a new ```processed/``` directory:

```
s3://YOUR-BUCKET-NAME/

processed/
├── filtered/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── edges/
│   ├── train/
│   ├── validation/
│   └── test/
│
└── manifest.csv
```

## [Part 2] Training CNN from scratch

⚠️ **For the detailed step-by-step explanation, read the [blog post on my website](https://daryapetrashka.com/blog/cv-asl-part-2/), or on [Dev.to](https://dev.to/dashapetr/part-2-your-first-steps-in-computer-vision-on-aws-training-a-custom-cnn-model-1bjm), or on [AWS Builder Center](https://builder.aws.com/content/3LQMywk8ABbdx1JJcD84SbTux9j/part-2-your-first-steps-in-computer-vision-on-aws-training-a-custom-cnn-model).**

This is the second part of the series "Your first steps in Computer Vision on AWS". If you are just starting, this tutorial will guide you step by step from the initial dataset to a deployed app.

### 1. Create a GPU-powered SageMaker notebook instance

Since training on a CPU would require a lot of time (one epoch can take 1.5 hours), let's use GPU power!

The AWS `ml.g5.xlarge` SageMaker instance (and the corresponding EC2 `g5.xlarge` instance) has 1 NVIDIA A10G Tensor Core GPU with 24 GiB of VRAM (video memory).

Pricing: `ml.g5.xlarge` is not as cheap as `ml.t3.medium`. The exact price can vary depending on the region (e.g., for Frankfurt it is $1.258 per hour); see [this page for on-demand pricing reference](https://aws.amazon.com/ec2/pricing/on-demand/).

⚠️ **Don't forget to stop and terminate your instance once the data is processed and uploaded to S3.** 

### 2. Copy the image data to the instance

Once the SageMaker notebook instance is ready, you can **Open Jupyter / Open JupyterLab**.

Upload the `Raw_images_training.ipynb` Jupyter notebook [from the current GitHub repo](https://github.com/dashapetr/Computer_Vision_ASL_AWS/blob/main/sagemaker_notebooks/Raw_images_training.ipynb).

Run cell by cell.

⚠️ Don't forget to replace the placeholder and put your S3 bucket name:
```BUCKET = "YOUR_BUCKET"```

This cell copies the preprocessed data from S3 to the instance. You only need to run it once and then check that the data has been copied successfully.
```
!aws s3 sync s3://YOUR-BUCKET/raw/asl_alphabet_train/ /home/ec2-user/SageMaker/asl_alphabet_train/ --quiet
!aws s3 sync s3://YOUR-BUCKET/processed/ /home/ec2-user/SageMaker/asl_preprocessed/  --quiet
```

### 3. Train three CNN models

⚠️ Remember we had 3 versions of data: raw, Gaussian-filtered and Canny edge? Well, you can use [this `Gaussian_filtered_training.ipynb` notebook](https://github.com/dashapetr/Computer_Vision_ASL_AWS/blob/main/sagemaker_notebooks/Gaussian_filtered_training.ipynb) to train the CNN with Gaussian-filtered images; and [this `Canny_edges_training.ipynb` notebook](https://github.com/dashapetr/Computer_Vision_ASL_AWS/blob/main/sagemaker_notebooks/Canny_edges_training.ipynb) to train the CNN with Canny edge images.
They don't differ much. I created three separate notebooks mainly to keep the cell output histories separate. 

If for you it is more comfortable to use a single notebook, feel free to do it. The only thing you would need to change is the `REPRESENTATION` variable. Set it to `raw`, `filtered`, or `edges` depending on the image version. 

#### What each notebook does (high-level):

1. Creates train/validation/test dataframes
2. Designs and compiles the CNN
3. Trains the CNN from scratch
4. Evaluates on the held-out test set
5. Saves the model and uploads it to S3

Let's show the model architecture:

![The model architecture](images/cnn-architecture.png)

We are saving the **final** model, **best** model, and **training history** to S3.
After running all the cells in all three notebooks (one for each dataset version), your bucket should contain a new ```models/``` directory with the following structure:

```
s3://YOUR-BUCKET-NAME/

models/
├── asl_cnn_raw_best.keras
├── asl_cnn_raw_final.keras
├── training_history_raw.csv
├── asl_cnn_filtered_best.keras
├── asl_cnn_filtered_final.keras
├── training_history_filtered.csv
├── asl_cnn_edges_best.keras
├── asl_cnn_edges_final.keras
└── training_history_edges.csv
```

### 4. Which model is the best?

After running the notebook for 3 dataset versions (raw, Gaussian-filtered and Canny edge), it is interesting to find out which data resulted in the best model performance.

| Dataset Representation | Loss | Accuracy |
| ------ | ----------- |----------|
|Raw RGB | 0.0324 | 99.34%   |
|Canny Edges | 0.1291 | 95.51%   | 
|Gaussian Filtered | 0.2972 | 90.11%   |

It turns out that the optimal performance (highest accuracy, lowest loss) was achieved using raw RGB images.

## [Part 3] Deploy models to use in an ASL prediction app (to be continued...)