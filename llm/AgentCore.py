from langchain_core.prompts import PromptTemplate
import streamlit as st
from pydantic import BaseModel, Field

from llm.ModelCore import load_gpt4o_mini
from langchain_core.output_parsers import PydanticOutputParser


class Response(BaseModel):
    origin: str = Field(description='the origin input sentence.')
    trans: str = Field(description='the translated sentence')


@st.cache_data(show_spinner='Translate sentence...')
def translate_sentence(question: str, template: str):
    llm = load_gpt4o_mini()
    parser = PydanticOutputParser(pydantic_object=Response)
    prompt = PromptTemplate(
        template=template,
        input_variables=['question'],
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    prompt_and_model = prompt | llm | parser
    result = prompt_and_model.invoke({'question': question})

    return result
