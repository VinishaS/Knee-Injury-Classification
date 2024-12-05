# **KNEE INJURY CLASSIFICATION**

We have developed a project which uses MRI scans from the Stanford MRNet dataset, preprocessed into three planes: axial, coronal, and sagittal. Three different ResNet-18 models extract features from each plane individually, which are then combined into one feature vector. This feature vector is used to train a logistic regression classifier that makes the final predictions. A Streamlit dashboard allows users to upload MRI images and receive real-time injury predictions.

# **DATASET**

We have used the dataset *'MRNet'* prepared by the Stanford ML group. The MRNet dataset consists of 1,370 knee MRI exams performed at Stanford University Medical Center. The dataset includes 1,104 (80.6%) abnormal exams, including 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; we manually extracted the labels from clinical reports. Due to the research agreement and usage limitations, we are not able to provide a preview of the dataset or link to the download page of the dataset.

***Link to the dataset description:*** https://stanfordmlgroup.github.io/competitions/mrnet/

# **TECHNIQUES, APPROACH AND MODELS USED**

To identify knee injuries using MRI scans from three planes of an image—axial, coronal, and sagittal—the project uses a multistep pipeline. Each gives a distinct perspective of the knee's structure, so different ResNet-18 models are fine-tuned on each plane to get features from MRI slices that have already been processed. They are trained to recognize trends that point to problems, like tears in the ACL and the meniscus. After training, these models make feature embeddings for the validation set that bring together important data from each plane. The features are then put together to make a single feature vector that includes information from all three planes. Those features are used by a logistic regression model in learning how to predict the final type of injury. The result is a strong diagnostic system that is also easy to understand. The pipeline makes sure that knee accidents are looked at as a whole by combining data from several planes, which makes the predictions very accurate.

# **RESULTS AND OBSERVATIONS**

After running the model on the validation data, the model yielded good results in terms of performance metrics. It has an accuracy of 87.50%, precision of 86.83%, recall of 87.50%, and an F1-score of 86.70%. Also, the AUC score is 0.9023, further showing the strength of the model in distinguishing different injury types effectively. Hence the system can be a very useful tool in medical diagnosis automation. Nevertheless, further testing on various datasets and clinical settings might help validate its generalizability and utility.

***Here are a few screenshots of the final dashboard that we were able to create***
<img width="1512" alt="Screenshot 2024-12-05 at 12 20 59 PM" src="https://github.com/user-attachments/assets/c62e557f-fa21-44e4-a8ac-ef62c838c1a1">

<img width="1512" alt="Screenshot 2024-12-05 at 12 21 33 PM" src="https://github.com/user-attachments/assets/33b65486-74f4-4848-944a-15fe2d3e0bbd">

