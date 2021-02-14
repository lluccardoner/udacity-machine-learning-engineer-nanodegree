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

Note that the following code uses SageMaker 1.x

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
### Training and building a model<a name="1-3" />
### Testing the model<a name="1-4" />
### Deploying a model<a name="1-5" />
### How to use a deployed model<a name="1-6" />
### Hyper parameter tuning<a name="1-7" />
### Model A/B testing<a name="1-8" />
### Updating an endpoint<a name="1-9" />
