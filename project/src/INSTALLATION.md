Please install Tesseract

## Windows

1. install librairy from https://github.com/UB-Mannheim/tesseract/wiki

2. Note the tesseract path from the installation.Default installation path at the time the time of this edit was: C:\Users\USER\AppData\Local\Tesseract-OCR. It may change so please check the installation path.

3. python -m pip install tesseract

4. change variable in python script pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

## Mac

brew install tesseract

## Linux 

sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev