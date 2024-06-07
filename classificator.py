from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st = SentenceTransformer('all-mpnet-base-v2')

def predict(cv, job):
  diffYoe = cv.yoe - job.minimumYoe
  results = {}
  results['score'] = 0.6
  results['is_accepted'] = True
  return results