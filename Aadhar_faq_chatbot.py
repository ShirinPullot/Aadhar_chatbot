import json
from transformers import DPRQuestionEncoderTokenizer,DPRQuestionEncoder
from sentence_transformers.util import cos_sim
question_model= DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer= DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')




def aadhar_chatbot(query):
    index,score=get_score(query)
    print(index)
    if score<=0.86:
        return 'sorry, no match found'
    else:
        faq_questions = get_faq_questions()
        match_question = faq_questions[index]
        faq_dict= get_faq_dict()
        answer = faq_dict[match_question]
        return answer
    

def get_score(query):
    query_token= question_tokenizer(query,max_length=256, padding='max_length',
    return_tensors='pt')
    query_out= question_model(**query_token)
    query_embedding= query_out.pooler_output
    faq_out= faq_questions_embeddings()
    scores=cos_sim(query_embedding, faq_out.pooler_output)
    scores=scores.tolist()[0]
    print('score is', scores)
    scores=[(m, score) for m,score in enumerate(scores)]
    scores.sort(key=lambda x:x[1],reverse=True)
    print('score after sorting', scores)    
    high_score= scores[0]
    return high_score
    

def get_faq_dict():
        dataset_file_path = "/Users/shirinwadood/Desktop/projects/Q&A Sentence similarity/archive/Aadhar_Faq.json"
        question_answers_list = read_question_answers_list(dataset_file_path)
        faq_dict = {}
        for qa_dict in question_answers_list:
            question = qa_dict['question']
            faq_dict[question] = qa_dict['answer']
        return faq_dict
    

def faq_questions_embeddings():
    questions= get_faq_questions()
    faq_tokens=question_tokenizer(questions,max_length=256, padding='max_length',
    return_tensors='pt')
    faq_out= question_model(**faq_tokens)
    return faq_out

def get_faq_questions():
        dataset_file_path = "/Users/shirinwadood/Desktop/projects/Q&A Sentence similarity/archive/Aadhar_Faq.json"
        question_answers_list = read_question_answers_list(dataset_file_path)
        questions = []
        for qa_dict in question_answers_list:
            questions.append(qa_dict['question'])
        return questions


def read_question_answers_list(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data['faq']




    