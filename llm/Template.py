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
For each question, a list of fragments is given. Your task is to answer the user question both in English and Chinese based only on the given essay fragment, and cite the fragment used.
If you don't know the answer, just say that you don't know, don't try to make up an answer!
{format_instructions}

Here is an example of a Q&A.
Input: {example_q}

Output: {example_a}
"""

EXAMPLE_Q = """Answer the question within 100 words.
Here are the fragments: 

Fragment ID: 0
Essay Title: Attention Is All You Need
Essay Author: A, Vaswani.
Publish year: 2017
Essay DOI: 10.48550/arXiv.1706.03762
Fragment Snippet: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. 

Fragment ID: 1
Essay Title: Attention Is All You Need
Essay Author: A, Vaswani.
Publish year: 2017
Essay DOI: 10.48550/arXiv.1706.03762
Fragment Snippet: Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences.

Fragment ID: 2
Essay Title: Attention Is All You Need
Essay Author: A, Vaswani.
Publish year: 2017
Essay DOI: 10.48550/arXiv.1706.03762
Fragment Snippet: The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

Question: What structural model does this article address?
"""

EXAMPLE_A = """```
{
"answer_en": "The article addresses the Transformer model as a structural model. The Transformer uses stacked self-attention and fully connected layers for both the encoder and the decoder. This model allows the modeling of dependencies without regard to their distance in the input or output sequences, making attention mechanisms an integral part.",
"answer_zh": "文章论述了作为结构模型的 transformer 模型。transformer 在编码器和解码器中都使用了堆叠自注意和全连接层。该模型允许对依赖关系进行建模，而不考虑它们在输入或输出序列中的距离，使注意机制成为不可分割的一部分。",
"citations": [0,2]
}
```
"""

ASK_USER = """Answer the question within 300 words.
Here are the fragments: {context}

Question: {question}
"""
