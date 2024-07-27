from kafkaWorker import consume_message
from dotenv import load_dotenv
import os 

os.environ['TRANSFORMERS_CACHE'] = '/transformers_cache'
os.environ['HF_HOME'] = '/transformers_cache'

if __name__ == "__main__":
    load_dotenv()
    consume_message()