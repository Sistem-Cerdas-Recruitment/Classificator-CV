from transformers import RobertaTokenizerFast, AutoModelForTokenClassification
import re
import torch

tokenizer = RobertaTokenizerFast.from_pretrained("mrfirdauss/robert-base-finetuned-cv")
model = AutoModelForTokenClassification.from_pretrained("mrfirdauss/robert-base-finetuned-cv")

id2label = {0: 'O',
     1: 'B-NAME',
     3: 'B-NATION',
     5: 'B-EMAIL',
     7: 'B-URL',
     9: 'B-CAMPUS',
     11: 'B-MAJOR',
     13: 'B-COMPANY',
     15: 'B-DESIGNATION',
     17: 'B-GPA',
     19: 'B-PHONE NUMBER',
     21: 'B-ACHIEVEMENT',
     23: 'B-EXPERIENCES DESC',
     25: 'B-SKILLS',
     27: 'B-PROJECTS',
     2: 'I-NAME',
     4: 'I-NATION',
     6: 'I-EMAIL',
     8: 'I-URL',
     10: 'I-CAMPUS',
     12: 'I-MAJOR',
     14: 'I-COMPANY',
     16: 'I-DESIGNATION',
     18: 'I-GPA',
     20: 'I-PHONE NUMBER',
     22: 'I-ACHIEVEMENT',
     24: 'I-EXPERIENCES DESC',
     26: 'I-SKILLS',
     28: 'I-PROJECTS'}

def merge_subwords(tokens, labels):
    merged_tokens = []
    merged_labels = []

    current_token = ""
    current_label = ""

    for token, label in zip(tokens, labels):
        if token.startswith("Ġ"):
            if current_token:
                # Append the accumulated subwords as a new token and label
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            # Start a new token and label
            current_token = token[1:]  # Remove the 'Ġ'
            current_label = label
        else:
            # Continue accumulating subwords into the current token
            current_token += token

    # Append the last token and label
    if current_token:
        merged_tokens.append(current_token)
        merged_labels.append(current_label)

    return merged_tokens, merged_labels

def chunked_inference(text, tokenizer, model, max_length=512):
    # Tokenize the text with truncation=False to get the full list of tokens
    tok = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    tokens = tokenizer.tokenize(tok, is_split_into_words=True)
    # Initialize containers for tokenized inputs
    input_ids_chunks = []
    # Create chunks of tokens that fit within the model's maximum input size
    for i in range(0, len(tokens), max_length - 2):  # -2 accounts for special tokens [CLS] and [SEP]
        chunk = tokens[i:i + max_length - 2]
        # Encode the chunks. Add special tokens via the tokenizer
        chunk_ids = tokenizer.convert_tokens_to_ids(chunk)
        chunk_ids = tokenizer.build_inputs_with_special_tokens(chunk_ids)
        input_ids_chunks.append(chunk_ids)

    # Convert list of token ids into a tensor
    input_ids_chunks = [torch.tensor(chunk_ids).unsqueeze(0) for chunk_ids in input_ids_chunks]

    # Predictions container
    predictions = []

    # Process each chunk
    for input_ids in input_ids_chunks:
        attention_mask = torch.ones_like(input_ids)  # Create an attention mask for the inputs
        output = model(input_ids, attention_mask=attention_mask)
        logits = output[0] if isinstance(output, tuple) else output.logits
        predictions_chunk = torch.argmax(logits, dim=-1).squeeze(0)
        predictions.append(predictions_chunk[1:-1])

    # Optionally, you can convert predictions to labels here
    # Flatten the list of tensors into one long tensor for label mapping
    predictions = torch.cat(predictions, dim=0)
    predicted_labels = [id2label[pred.item()] for pred in predictions]
    return merge_subwords(tokens,predicted_labels)

def process_tokens(tokens, tag_prefix):
    # Process tokens to extract entities based on the tag prefix
    entities = []
    current_entity = {}
    for token, tag in tokens:
        if tag.startswith('B-') and tag.endswith(tag_prefix):
            # Start a new entity
            if current_entity:
                # Append the current entity before starting a new one
                entities.append(current_entity)
                current_entity = {}
            current_entity['text'] = token
            current_entity['type'] = tag
        elif tag.startswith('I-') and (('GPA') == tag_prefix or tag_prefix == ('URL')) and tag.endswith(tag_prefix) and current_entity:
            current_entity['text'] += '' + token
        elif tag.startswith('I-') and tag.endswith(tag_prefix) and current_entity:
            # Continue the current entity
            current_entity['text'] += ' ' + token
    # Append the last entity if there is one
    if current_entity:
        entities.append(current_entity)
    return entities

def predict(text):
    tokens, predictions = chunked_inference(text, tokenizer, model)
    data = list(zip(tokens, predictions))
    profile = {
        "name": "",
        "links": [],
        "skills": [],
        "experiences": [],
        "educations": []
    }
    profile['name'] = ' '.join([t for t, p in data if p.endswith('NAME')])

    for skills in process_tokens(data, 'SKILLS'):
      profile['skills'].append(skills['text'])
    #Links
    for links in process_tokens(data, 'URL'):
      profile['links'].append(links['text'])
    # Process experiences and education
    for designation, company, experience_desc in zip(process_tokens(data, 'DESIGNATION'),process_tokens(data, 'COMPANY'),process_tokens(data, 'EXPERIENCES DESC') ):
        profile['experiences'].append({
            "start": None,
            "end": None,
            "designation": designation['text'],
            "company": company['text'],  # To be filled in similarly
            "experience_description": experience_desc['text']  # To be filled in similarly
        })
    for major, gpa, campus in zip(process_tokens(data, 'MAJOR'), process_tokens(data, 'GPA'), process_tokens(data, 'CAMPUS')):
        profile['educations'].append({
            "start": None,
            "end": None,
            "major": major['text'],
            "campus": campus['text'],  # To be filled in similarly
            "GPA": gpa['text'] # To be filled in similarly
        })

    return profile