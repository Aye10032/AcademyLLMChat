GENERATE_QUESTION = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector database.  
By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
The given question may be in Chinese or English, and the question you return must in English.
Provide these alternative questions separated by newlines.
Original question: {question}
"""


TRANSLATE_TO_EN = """Assuming you are an AI assistant, your task is to accurately translate the user's 
input questions from Chinese to English. 
If the input question is already in English, there is no need to translate 
and you can return the original sentence as it is.
{format_instructions}
{question}
"""


TRANSLATE_TO_ZH = """Assuming you are an AI assistant, your task is to accurately translate 
the user's input questions from Engilsh to Chinese. 
{format_instructions}
{question}
"""


ASK_SYSTEM = """Assuming you are a professor.
Your task is to answer the user question both in English and Chinese based only on the following given essay fragment, and cite the fragment used.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Here are the articles: {context}
"""

ASK_USER = """Answer the question within 100 words, translate your answer into Chinese, and use line breaks to separate Chinese and English responses!
{question}"""
