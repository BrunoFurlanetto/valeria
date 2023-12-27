import os

import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()
token = 'AIzaSyBDOxnt2u7BPGHE9pM8x9nZeOLhm3RGvYg'
genai.configure(api_key=token)

model = genai.GenerativeModel('gemini-pro')

chat = model.start_chat()

while True:
    chat.send_message('Você é Valéria uma assistente virtual e responde perguntas de forma mais sucinta possível')
    text = input('...')

    if text == 'sair':
        break

    response = chat.send_message(text)
    console.print("Valeria: ", response.text, '\n')
