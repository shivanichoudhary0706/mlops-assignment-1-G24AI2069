# Boston Housing Price Prediction - mlops-assignment-1

This project builds a simple machine learning pipeline to predict house prices using the Boston Housing Dataset. I've implemented two classic regression models:

1.Decision Tree Regressor
2.Kernel Ridge Regressor

Each model lives in its own branch, and all shared logic are kept reusable.

File Structure:
mlops-assignment-1-G24AI2069
| -main
|   |- readme
| -dtree
|   |- requirements.txt
|   |- misc.py
|   |- train.py
| -kernelridge
    |- train2.py
    |- .github/workflows/ci.yml
