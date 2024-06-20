# Predict API

## Trigger the predict function from the model and get the API prediction endpoint :

This is a python code, so make sure you have python installed on your system.

### Installation and Usage

[Docker](https://docs.docker.com/desktop/install/mac-install/) must be installed on your system.

### Building the Docker Image

1. Clone the repository:

```bash
    git clone https://github.com/nutriomatic/predict-api.git
    cd predict-api
```

2. Build the Docker image:

```bash
   docker build -t predict-api .
```

## Running the Application

1. Run the Docker container:

```bash
   docker run -d -p 8080:8080 --name predict-api predict-api
```

This will start the application and map port 8080 of the Docker container to port 8080 on your local machine.

2. Access the application:
   Open your web browser and navigate to [http://localhost:8080](http://localhost:8080)

3. If it shows 'Hello, world!' then you have successfully run the predict api.

4. The next step is to configure the backend service, you can find it in the [backend](https://github.com/nutriomatic/backend.git) repository.

## Stopping the Application

1. Stop the Docker container:

```bash
   docker stop predict-api
```

2. Remove the Docker container:

```bash
   docker rm predict-api
```

## Directory Structure

```bash
.
├── core
│   ├── main.py
│   └── utils.py
├── models
│   ├── model_grade_predict_dum.h5
│   ├── model_new.h5
│   └── scaler.joblib
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or new features.
