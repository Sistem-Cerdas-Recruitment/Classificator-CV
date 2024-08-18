# Kafka Job Vacancy Matcher

This project implements a Kafka-based service to match job vacancies with user CVs. The service uses a Support Vector Machine (SVM) model to perform the classification, with the model being stored in a `svc.pkl` file. The Kafka consumer listens to messages on the `matcher` topic, processes the data, and outputs the matching results.

## Features

- **Kafka Integration**: Consumes job and CV data from the `matcher` topic.
- **SVM Model**: Utilizes a pre-trained SVM model saved in `svc.pkl` to classify and match job vacancies with CVs.
- **Docker Support**: Easily deploy the service using Docker.
- **Huggingface Hub Integration**: Allows easy model management and updates.

## Project Structure

- `classificator.py`: Contains the logic for loading the SVM model and classifying job vacancy and CV data.
- `kafkaWorker.py`: Manages the Kafka consumer that listens to the `matcher` topic and triggers the classification process.
- `main.py`: Entry point of the application, sets up the Kafka consumer and handles the data flow.
- `svc.pkl`: The pre-trained SVM model used for classification.
- `Dockerfile`: Defines the Docker container setup for the service.
- `requirements.txt`: Lists the Python dependencies needed to run the service.

## Installation

1. Clone the repository:

   ``` git clone https://github.com/mrfirdauss-20/kafka-job-matcher.git && cd kafka-job-matcher```

2. Build the docker:
   ```  docker build -t kafka-job-matcher```
4. Run the dockerfile:
   ``` docker run -d kafka-job-matcher ```

## Contributor
- 13520043 Muhammad Risqi Firdaus
