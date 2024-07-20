import os
import requests
import	json
from kafka import KafkaConsumer
from datetime import datetime
import classificator

def classify(data):
    mininmal_start = 0
    maximal_end = 0
    positions = []
    userMajors = []
    yoe = 0
    if len(data['cv']['experiences']) > 0:
        mininmal_start = datetime.strptime(data['cv']['experiences'][0]['start'], "%Y-%m-%d").date() if data['cv']['experiences'][0].get('start') != None else datetime.today().date()
        maximal_end = datetime.strptime(data['cv']['experiences'][0]['end'], "%Y-%m-%d").date() if data['cv']['experiences'][0].get('end') != None else datetime.today().date()
        for exp in data['cv']['experiences']:
            positions.append(exp['position'])
            if exp.get('end') == None:
                exp['end'] = datetime.today().strftime("%Y-%m-%d")
            if datetime.strptime(exp['start'], "%Y-%m-%d").date() < mininmal_start:
                mininmal_start = datetime.strptime(exp['start'], "%Y-%m-%d").date()
            if datetime.strptime(exp['end'], "%Y-%m-%d").date() > maximal_end:
                maximal_end = datetime.strptime(exp['end'], "%Y-%m-%d").date()
        yoe = (maximal_end - mininmal_start).days//365  
    
    for edu in data['cv']['educations']:
        userMajors.append(edu['major'])
    
    cv = {
        "experiences": str(data['cv']['experiences']), 
        "positions": str(positions), 
        "userMajors": str(userMajors), 
        "skills": str(data['cv'].skills), 
        "yoe": yoe
    }
    job = {
        "jobDesc": data['job']['jobDesc'], 
        "role": data['job']['role'], 
        "majors": str(data['job']['majors']), 
        "skills": str(data['job']['skills']), 
        "minYoE": data['job']['minYoE']
    }
    results = classificator.predict(cv, job)
    return results

def send_results_back(full_results: dict[str, any], job_application_id: str):
    print(f"Sending results back with job_app_id {job_application_id}")
    url = os.env.get("KAFKA_URL")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": os.environ.get("KAFKA_API_KEY")
    }

    body = {
        "job_application_id": job_application_id,
        "evaluations": full_results
    }

    response = requests.patch(url, json=body, headers=headers)
    print(f"Data sent with status code {response.status_code}")

def consume_message():
    consumer = KafkaConsumer(
        "matching",
        boostrap_servers=[os.environ.get("KAFKA_IP")],
        auto_offset_reset='earliest',
        client_id="matcher-1",
        group_id="matcher",
        api_version=(0, 10, 2)
    )

    for message in consumer:
        try:
            incoming_message = json.loads(json.loads(message.value.decode('utf-8')))
            data = incoming_message['data']
        except json.JSONDecodeError as e:
            print(f"Error in decoding message: {e}")
            continue
        print(f"Message: {data}")
        result = classify(data)
        send_results_back(result, incoming_message['job_application_id'])