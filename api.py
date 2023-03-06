import uvicorn
from fastapi import FastAPI
from Aadhar_faq_chatbot import aadhar_chatbot

app = FastAPI()


@app.get('/')
def get_root():
    return {'message': 'Aadhar Chatbot app'}


@app.get('/Aadhar chatbot/')
def aadhar_related_query(text: str):
    return aadhar_chatbot(text) 