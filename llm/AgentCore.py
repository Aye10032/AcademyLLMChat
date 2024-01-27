from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st

from llm.ModelCore import load_gpt


class Response(BaseModel):
    answer: List[str] = Field(description='List of generated questions.')


@st.cache_data
def get_similar_question(question: str):
    parser = PydanticOutputParser(pydantic_object=Response)
    prompt = PromptTemplate(
        template='Generate three similar questions based on my query.\n{format_instructions}\n{query}\n',
        input_variables=['query'],
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )
    llm = load_gpt()

    prompt_and_model = prompt | llm
    output = prompt_and_model.invoke({'query': question})
    result = parser.invoke(output)

    return result
