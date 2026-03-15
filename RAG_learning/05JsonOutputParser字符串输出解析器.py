#当上个模型的输出没有被处理就输入下一个模型，这种方法是错误的
#正确的做法是使用一个输出解析器来处理上个模型的输出，然后再输入下一个模型
# 初始输入 ->提示词模板 -> 模型 -> 数据处理 -> 提示词模板 -> 模型 -> 解析器 -> 结果

"""所以，我么必须要完成：
将模型输出的AIMessage -> 转为字典 -> 注入第二个提示词模板中，形成新的提示词（PromptValue对象）
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

str_parser = StrOutputParser()
json_parser = JsonOutputParser()
LLM = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)

first_prompt = PromptTemplate.from_template(
    "我邻居姓{last_name},刚生了一个{gender}孩,请起名，并封装到JSON格式返回给我"
    "要求JSON格式如下：{{'name': '名字'}}"
)

second_prompt = PromptTemplate.from_template(
    "姓名{name}，请你解析这个名字"
)

chain = first_prompt | LLM | json_parser | second_prompt | LLM | str_parser

# res = chain.invoke(input={"last_name":"张","gender":"男"})
# print(res)
# print(type(res))

for chunk in chain.stream(input={"last_name":"张","gender":"男"}):
    print(chunk,end="",flush=True)