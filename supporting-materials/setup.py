from setuptools import setup, find_packages

setup(
    name="barry",  
    version="1.0.1",  
    packages=find_packages(include=["barry", "barry.*"]),  # This includes all sub-packages inside the 'barry' folder
    install_requires=[
            "IPython",              # For working with IPython in Jupyter
            "Pillow",               # For image processing (replaces PIL)
            "h5py",                 # For HDF5 file handling
            "livelossplot",         # For live loss plotting during training
            "matplotlib",           # For data visualization
            "numpy",                # For numerical computations
            "pandas",               # For data manipulation
            "pycm",                 # For confusion matrix visualization
            "scikit-learn",         # For machine learning tasks
            "seaborn",              # For statistical data visualization
            "scipy",                # For scientific computing
            "tqdm",                 # For progress bars
            "torch",                # Core PyTorch library
            "torchinfo",            # For model summary
            "torchmetrics",         # For PyTorch metrics
            "torchvision",          # For computer vision tasks
    ],
    python_requires=">=3.11.11",
    authors="Team Barry: Daniel Kaupa, Xingyu Liu, Vadim Malz, Joe Najem, Aditi Srivastava, Haitong Wang, Zewei, Zhang",
    author_email="daniel.kaupa24@imperial.ac.uk",
    description="This project was developed as part of an assessment at **Imperial College** within the **Ada Lovelace Academy**. The goal of this project is to provide a solution to the problem of forecasting the evolution of lightning storms in real-time.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ese-ada-lovelace-2024/acds-storm-prediction-barry",
    license="CC BY 4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International License",
        "Operating System :: OS Independent",
    ],
)
# This file was created with the assistance of the ChatGPT.