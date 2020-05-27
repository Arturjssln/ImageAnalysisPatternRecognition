# Image Analysis and Pattern Recognition project
# Project – Solving a math problem

## Requirements
numpy
opencv-python
sympy
scikit-image
scikit-learn
matplotlib
Augmentor
Keras


To install the previous requirements, please run the following command:
```
pip install -r requirements.txt
or
python -m pip install -r requirements.txt 
```


## Usage
Run the following command from the src directory:
```bash
python main.py

Options:
    --input PATH            Path of input video
                                (defaut: '../data/robot_parcours.avi')
    --output PATH           Path of output video
                                (default: '../results/robot_parcours.avi')
    --digit_model PATH      Path of digit model
                                (defaut: 'model.h5')
    --operator_model PATH   Path of operator model
                                (default: 'operators.h5')
    --train_operators       Do the training of the operators model 
    --train_digits          Do the training of the digits model
    --augment_images        Augment images to have more training data
    --debug                 Display debug information

```