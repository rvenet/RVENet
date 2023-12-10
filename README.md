## <span style="color:red">NOTE: This is a beta version of the source code, indicating that it is currently under development, and bugs may occur. If you have any questions, please feel free to open an issue or contact us directly (refer to the contact information below).</span>

# Source code related to the research paper entitled RVENet: A Large Echocardiographic Dataset for the Deep Learning-Based Assessment of Right Ventricular Function


The purpose of this repository is to enable the application of our deep learning pipeline that was developed for estimating right ventricular ejection fraction (RVEF) from 2D apical 4-chamber view echocardiographic videos. For detailed information about our model and the entire data analysis pipeline, please refer to the following papers:


> [**RVENet: A Large Echocardiographic Dataset for the Deep Learning-Based Assessment of Right Ventricular Function**](https://doi.org/10.1007/978-3-031-25066-8_33)<br/>
  Bálint Magyar*, Márton Tokodi*, András Soós, Máté Tolvaj, Bálint K. Lakatos, Alexandra Fábián, Elena Surkova, Béla Merkely, Attila Kovács and András Horváth<br/>
  <b>ECCV</b> (2022)

> [**Deep Learning-Based Prediction of Right Ventricular Ejection Fraction Using 2D Echocardiograms**](https://doi.org/10.1016/j.jcmg.2023.02.017)<br/>
Márton Tokodi*, Bálint Magyar*, András Soós, Masaaki Takeuchi, Máté Tolvaj, Bálint K. Lakatos, Tetsuji Kitano, Yosuke Nabeshima, Alexandra Fábián, Mark B. Szigeti, András Horváth, Béla Merkely, and Attila Kovács<br/>
<b>JACC: Cardiovascular Imaging</b> (2023)

## Clinical Significance


Two-dimensional echocardiography is the most frequently performed imaging test to assess RV function. However, conventional 2D parameters are unable to reliably capture RV dysfunction across the entire spectrum of cardiac diseases. Three-dimensional echocardiography-derived RVEF – a sensitive and reproducible parameter that has been validated against cardiac magnetic resonance imaging – can bypass most of their limitations. Nonetheless, 3D echocardiography has limited availability, is more time-consuming, and requires significant human expertise. Therefore, novel automated tools that utilize readily available and routinely acquired 2D echocardiographic recordings to predict RVEF and detect RV dysfunction reliably would be highly desirable.

## Brief Description of the Deep Learning Pipeline


Our deep learning pipeline was designed to analyze a DICOM file containing a 2D apical 4-chamber view echocardiographic video to predict 3D echocardiography-derived RVEF as either a continuous variable or a class. Following a multi-step preprocessing, the preprocessed frames of the video and a binary mask are passed to a deep learning model that consists of three key components:

  1) a **feature extractor** (different off-the-shelf backbones such as ResNext or ShuffleNet) that derives unique features from each frame,<br/>
  2) a **temporal convolutional layer** that combines the extracted features to evaluate temporal changes,
  3) and a **regression head** or a **classification head** (comprising two fully connected layers), which returns a single predicted RVEF value or a class (e.g. normal or reduced RVEF).

Mean absolute error or binary cross-entropy was used as the loss function (depending on the task). As the model was trained to analyze one cardiac cycle at a time, the average of the per-cardiac cycle predictions can be calculated to get a single predicted RVEF value for a given video.

## Datasets Used for Model Development and Evaluation


A large single-center dataset – comprising 3,583 echocardiographic videos of 831 subjects (in DICOM format) and the corresponding labels (e.g., 3D echocardiography-derived RVEF) – was used for the training and internal validation of the models provided in this repository. To encourage collaborations and promote further innovation, we made this dataset publicly available (<ins>**RVENet dataset**</ins> – https://rvenet.github.io/dataset/).

![Example videos](imgs/dicom_collage.gif)
<div align="center"><i><b>Figure 1</b> Apical 4-chamber view echocardiographic videos sampled from the RVENet dataset</i></p></div>


## Contents of the Repository

  - `RVENet` - the RVENet codebase containing training and evaluation scripts
  - `RVENet_dataset_preprocessing` - scripts required for preprocessing the RVENet dataset
  - `run_training.py` - the single script that is required to start a training
  - `requirements.txt` - the list of the required Python packages
  - `setup.py` - installation script
  - `LICENSE.md` - details of the license (MIT License)
  - `README.md` - text file that introduces the project and explains the content of this repository

## Usage

### Install requirements and the RVENet python package

Run the follwoing script to install all requirements and the RVENet python package:

```
pip install .
```

### Preprocessing the RVENet dataset

The RVENet dataset can be preprocessed by following the steps below

  1) Download the dataset (further information at https://rvenet.github.io/dataset/)
  2) Decompress the downloaded zip files (both training and validation) into a single folder (use the 7z package under Ubuntu)
  3) Run the `RVENet_dataset_preprocessing/process_dicoms.py` script with the corresponding parameters to generate images from the dicom files
  4) Run the `RVENet_dataset_preprocessing/create_annotation.py` script with the corresponding parameters to generate the training and validation json files

### Train models using the RVENet dataset

Models can be trained by following the steps below

  1) Edit the parameters of the RVEnet\parameters_classification_baseline.json or the RVEnet\parameters_regression_baseline.json files depending on you target task (regression or classificaiton). The single parameter that must be changed is the ***DICOM_database_path*** (path to the folder containing preprocessed frames and annotation json files),
  2) Run the the ***run_training.py*** script

```
python run_training.py parameters_regression_baseline.json
```

## Contact


For inquiries related to the content of this repository, contact Bálint Magyar (magy<!--
-->ar.ba<!--
-->lint[at]itk.pp<!--
-->ke.h<!--
-->u)