# New Application Setup

## Prerequisites
Ensure you have `conda` installed. If not, download and install Anaconda or Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

## Setting Up the Environment

### 1. Create a Conda Environment
```sh
conda create --name new_app_env python=3.9 -y
conda activate new_app_env
```

### 2. Install Dependencies

Using Conda:
```sh
conda install flask flask-cors numpy=1.26.4 werkzeug opencv -y
```

Using Pip:
```sh
pip install flask-socketio ultralytics python-engineio python-socketio flask-jwt-extended pymongo waitress
```

## Setting Up MongoDB
Ensure MongoDB is installed and running. If you don’t have it installed, follow the official installation guide [here](https://www.mongodb.com/try/download/community).

To start MongoDB on MacOS:
```sh
brew services start mongodb-community
```

To start MongoDB on Linux:
```sh
sudo systemctl start mongod
```

To start MongoDB on Windows (if installed as a service):
```sh
net start MongoDB
```

## Signing Up a User
Before running the application, register an admin user using `curl`:
```sh
curl -X POST http://localhost:5000/api/auth/signup \
     -H "Content-Type: application/json" \
     -d '{
         "email": "admin@example.com",
         "password": "securPass1@3.3r2AA",
         "first_name": "John",
         "last_name": "Doe"
     }'
```

## Running the Application
To start the application, run:
```sh
python run.py
```

## Troubleshooting

### OpenMP Error (MacOS)
If you encounter the following error when running the application:

```
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
```

Try setting the following environment variable before running the script:
```sh
export KMP_DUPLICATE_LIB_OK=TRUE
```
Alternatively, uninstall one of the OpenMP versions:
```sh
conda remove intel-openmp llvm-openmp
```
Then reinstall only one:
```sh
conda install llvm-openmp -y
```

## Deleting the Conda Environment
If you need to delete the environment:
```sh
conda deactivate
conda env remove --name new_app_env
```