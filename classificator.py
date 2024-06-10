from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle 
st = SentenceTransformer('all-mpnet-base-v2')
filename = 'svc.pkl'

with open(filename, 'rb') as file:
  model = pickle.load(file)

# role_req-exp        0.341522
# role_pos            0.350747
# major_similarity    0.846268
# skill_similarity    0.774542
# score               0.986356
# cv = {
#     "experiences": str(body.cv.experiences), 
#     "positions": str(positions), 
#     "userMajors": str(userMajors), 
#     "skills": str(body.cv.skills), 
#     "yoe": yoe
# }
# job = {
#     "jobDesc": body.job.jobDesc, 
#     "role": body.job.role, 
#     "majors": str(body.job.majors), 
#     "skills": str(body.job.skills), 
#     "minYoE": body.job.minYoE
# }

def predict(cv, job):
  diffYoe = cv['yoe'] - job['minYoE']
  results = {}
  role_req_exp = cosine_similarity(st.encode(cv['experiences']).reshape(1,-1), st.encode(job['role']+'\n'+job['jobDesc']).reshape(1,-1))[0][0]
  role_pos = cosine_similarity(st.encode(cv['positions']).reshape(1,-1), st.encode(job['role']).reshape(1,-1))[0][0]
  major_similarity = cosine_similarity(st.encode(cv['userMajors']).reshape(1,-1), st.encode(job['majors']).reshape(1,-1))[0][0]
  skill_similarity = cosine_similarity(st.encode(cv['skills']).reshape(1,-1), st.encode(job['skills']).reshape(1,-1))[0][0]
  score_yoe = 0.5 if diffYoe == -1 else (0 if diffYoe < 0 else 1)
  score = 0.35 * role_req_exp + 0.1 * role_pos  + 0.15 * major_similarity + 0.3* score_yoe + 0.1 * skill_similarity 
  X = np.array([role_req_exp, role_pos, major_similarity, skill_similarity, score]).reshape(1, -1)
  res = model.predict(X)
  results['score'] = model.predict_proba(X)[:, 1]
  results['is_accepted'] = np.argmax(res) 
  return results