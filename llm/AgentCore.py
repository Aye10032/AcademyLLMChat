from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st

from llm.ModelCore import load_gpt


class Response(BaseModel):
    origin: str = Field(description='the origin input sentence.')
    trans: str = Field(description='the translated sentence')


@st.cache_data
def translate_sentence(question: str, template: str):
    llm = load_gpt()
    parser = PydanticOutputParser(pydantic_object=Response)
    prompt = PromptTemplate(
        template=template,
        input_variables=['question'],
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    prompt_and_model = prompt | llm | parser
    result = prompt_and_model.invoke({'question': question})

    return result
