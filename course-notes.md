# Course Notes

## Table of Contents
1. [AWS Sage Maker](#1)
  1. [Initialization](#1-1)
  2. [Storing data to S3](#1-2)
  3. [Training and building a model](#1-3)
  4. [Testing the model](#1-4)
  5. [Deploying a model](#1-5)
  6. [How to use a deployed model](#1-6)
  7. [Hyper parameter tuning](#1-7)
  8. [Model A/B testing](#1-8)
  9. [Updating an endpoint](#1-9)

## AWS Sage Maker<a name="1" />

- [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/latest/index.html#)
- [Udacity notebooks repository](https://github.com/udacity/sagemaker-deployment)

### Initialization<a name="1-1" />

Note that the following code uses SageMaker 1.x and it is run inside a SageMaker notebook instance.

```bash
pip install sagemaker==1.72.0
```

This is an object that represents the **SageMaker session** that we are currently operating in. 
This object contains some useful information that we will need to access later such as our region.

```python
import sagemaker
session = sagemaker.Session()
```

This is an object that represents the **IAM role** that we are currently assigned. 
When we construct and launch the training job later we will need to tell it what IAM role it should have. 
Since our use case is relatively simple we will simply assign the training job the role we currently have.

```python
from sagemaker import get_execution_role
role = get_execution_role()
```

### Storing data to S3<a name="1-2" />

Since we are currently running inside of a SageMaker session, we can use the object which represents this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with data uploaded from some other notebook or project.

```python
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

**NOTE**: The documentation for the XGBoost algorithm in SageMaker requires that the saved datasets should contain **no headers or index** and that for the training and validation data, the **label should occur first** for each sample.

### Training and building a model<a name="1-3" />

Sage Maker trains models with a training job and creates the model artifacts.

#### High level API

Now that we have the training and validation data uploaded to S3, we can construct our **XGBoost model** and train it. We will be making use of the **high level SageMaker API** to do this which will make the resulting code a little easier to read at the cost of some flexibility.

To construct an **estimator**, the object which we wish to train, we need to provide the location of a **container** which contains the training code. Since we are using a **built in algorithm** this container is provided by Amazon. However, the full name of the container is a bit lengthy and depends on the region that we are operating in. Fortunately, SageMaker provides a useful utility method called **get_image_uri** that constructs the image name for us.

To use the get_image_uri method we need to provide it with our current region, which can be obtained from the session object, and the name of the algorithm we wish to use. In this notebook we will be using XGBoost however you could try another algorithm if you wish. The list of built in algorithms can be found in the list of [Common Parameters](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).

```python
container = get_image_uri(session.boto_region_name, 'xgboost')

xgb = sagemaker.estimator.Estimator(container, 
                                    role,
                                    train_instance_count=1, 
                                    rain_instance_type='ml.m4.xlarge',                                   
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                    sagemaker_session=session)
```

Before asking SageMaker to begin the **training job**, we should probably set any model specific **hyperparameters**. There are quite a few that can be set when using the XGBoost algorithm, below are just a few of them. If you would like to change the hyperparameters below or modify additional ones you can find additional information on the [XGBoost hyperparameter page](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html).

```python
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)
```

Now that we have our estimator object completely set up, it is time to **train** it. To do this we make sure that SageMaker knows our **input data** is in csv format and then execute the fit method.

```python
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_val = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_val})
```

**Note**: the fit method does not create the Model per se. It creates the **model artifacts**. With the high level API, the model is created when needed. For exsmple with a transformer is created or the model is deployed.

### Testing the model<a name="1-4" />
### Deploying a model<a name="1-5" />
### How to use a deployed model<a name="1-6" />
### Hyper parameter tuning<a name="1-7" />
### Model A/B testing<a name="1-8" />
### Updating an endpoint<a name="1-9" />
