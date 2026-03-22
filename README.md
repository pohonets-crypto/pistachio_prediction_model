## App for prediction of pistachio type.

This app can recognize which type of pistachio in the picture(Kirmizi/Siirt).

## Get started

```shell
python -m venv venv
venv\Scripts\activate (on Windows)
source venv/bin/activate (on macOS)
pip install -r requirements.txt
```

```shell
python app.py
```
Send POST request to http://127.0.0.1:5000/predict with image file

Dataset:
https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset

Google Colab:
https://colab.research.google.com/drive/1FCScutdpxx7x-xGqGL-Z6NG8YhJJeHrH?usp=sharing

## Model Architecture
Input: 64×64 RGB images, normalized to [0, 1]

| Layer         | Details               |
|---------------|-----------------------|
| Conv2D        | 32 filters, 3×3, ReLU |
| MaxPooling2D  | 2×2                   |
| Conv2D        | 64 filters, 3×3, ReLU |
| MaxPooling2D  | 2×2                   |
| Flatten       | -                     |
| Dense         | 128 units, ReLU       |
| Dense(output) | 2 units, Softmax      |

## Results

|Test accuracy | 88 % |
|--------------|------|
|Test loss     | 0.27 |

