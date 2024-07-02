from fastapi import FastAPI, HTTPException
from models import CVExtracted, InsertedText, JobAndCV, ClassificationResult, InsertedLink
import os
from io import BytesIO
import extractor
from datetime import datetime
from PyPDF2 import PdfReader
import requests
import classificator

os.environ['TRANSFORMERS_CACHE'] = '/transformers_cache'
os.environ['HF_HOME'] = '/transformers_cache'



app =  FastAPI()
@app.get("/", response_model=dict[str, str])
def getall():
    return {"hello":"world"}


@app.post("/ext", response_model=CVExtracted)
async def extract(text: InsertedText):
    dictresult = extractor.predict(text.text)
    return CVExtracted(**dictresult)


@app.post("/classify", response_model=ClassificationResult)
async def classify(body:JobAndCV):
    mininmal_start = 0
    maximal_end = 0
    positions = []
    userMajors = []
    yoe = 0
    if len(body.cv.experiences) > 0:
        mininmal_start = datetime.strptime(body.cv.experiences[0]['start'], "%Y-%m-%d").date() if body.cv.experiences[0].get('start') != None else datetime.today().date()
        maximal_end = datetime.strptime(body.cv.experiences[0]['end'], "%Y-%m-%d").date() if body.cv.experiences[0].get('end') != None else datetime.today().date()
        for exp in body.cv.experiences:
            positions.append(exp['position'])
            if exp.get('end') == None:
                exp['end'] = datetime.today().strftime("%Y-%m-%d")
            if datetime.strptime(exp['start'], "%Y-%m-%d").date() < mininmal_start:
                mininmal_start = datetime.strptime(exp['start'], "%Y-%m-%d").date()
            if datetime.strptime(exp['end'], "%Y-%m-%d").date() > maximal_end:
                maximal_end = datetime.strptime(exp['end'], "%Y-%m-%d").date()
        yoe = (maximal_end - mininmal_start).days//365  
    
    for edu in body.cv.educations:
        userMajors.append(edu['major'])
    
    cv = {
        "experiences": str(body.cv.experiences), 
        "positions": str(positions), 
        "userMajors": str(userMajors), 
        "skills": str(body.cv.skills), 
        "yoe": yoe
    }
    job = {
        "jobDesc": body.job.jobDesc, 
        "role": body.job.role, 
        "majors": str(body.job.majors), 
        "skills": str(body.job.skills), 
        "minYoE": body.job.minYoE
    }
    results = classificator.predict(cv, job)
    return ClassificationResult(**results)

@app.post("/cv", response_model=CVExtracted)
async def extract(link: InsertedLink):
    response = requests.get(link.link)
    if response.status_code == 200:
        # Open the PDF from bytes in memory
        pdf_reader = PdfReader(BytesIO(response.content))
        number_of_pages = len(pdf_reader.pages)
        # Optionally, read text from the first page
        page = pdf_reader.pages[0]
        text = page.extract_text()
        for i in range(1, number_of_pages):
            text+= '\n' + pdf_reader.pages[i].extract_text()
    else:
        #return error, make 500 because file server error
        raise HTTPException(status_code=response.status_code, detail="File server error")

    dictresult = extractor.predict(text)
    return CVExtracted(**dictresult)