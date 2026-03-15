"""
promptTemplte:通用提示词模板，支持动态注入信息
Few-shot 提示词示例：用「模板 + 示例数据」拼出一段带例子的 prompt。
ChatPromptTemplate:支持注入任意数量的历史会话信息
知识点总结：
1. 示例模板里的占位符（如 {word}、{antonym}）与 example_data 的关联：
   - 模板里写什么名字，每条示例的字典里就要有同名的键。
   - 库会用每条 dict 的值去填模板，生成一条条例子。键名 = 占位符名，才能对上。

2. 两套占位符：
   - 示例模板中的 {word}、{antonym}：由 examples 里每条数据填，用于生成每个 few-shot 例子。
   - prefix/suffix 中的 {input_word}：属于「整段 prompt 的输入变量」，由 invoke(input={...}) 传入。
   - 在 input_variables 里声明的，是「调用时必传」的变量（如 input_word）。

3. 若 suffix 里没有 {input_word}：
   - 不必在 input_variables 里写 input_word，可设为 input_variables=[]。
   - 调用时用 invoke(input={})，无需传 input_word。

   format.(k=v,k=v)      #解析{}占位符
   invoke({"k":v,"k":v}) #解析{}占位符和MessagesPlaceHolder结构占位符
"""

from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_groq import ChatGroq  # 若要用模型跑完整 prompt 可再用
from dotenv import load_dotenv
load_dotenv()
import os 

groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()

LLM = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)
#示例模板
example = PromptTemplate(template='我的孩子要生了，我姓{last_name}，孩子是一个{gender}孩，请你帮我取一个合适的名字',
   input_variables=["last_name","gender"]
)
prompt_text = example.format(last_name="刘",gender="女")
res = LLM.invoke(input=prompt_text)
print(res.content)


# 示例fewshot模板：占位符 {word}、{antonym} 的名字必须和 example_data 里每条 dict 的键一致
example_temples = PromptTemplate.from_template("单词:{word}，反义词:{antonym}")

# 每条 dict 的键 "word"、"antonym" 对应模板中的 {word}、{antonym}，库会逐条填进去生成例子
example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]

few_show_prompt = FewShotPromptTemplate(
    example_prompt=example_temples,
    examples=example_data,
    prefix="给出制定此的反义词，有如下实例",
    # suffix 里的 {input_word} 是「整段 prompt」的输入变量，由下面 invoke(input={...}) 传入
    suffix="基于例子告诉我，{input_word}的反义词是？",
    input_variables={"input_word"}  # 声明调用时必须传入的变量名，与 suffix 中的占位符对应
)

# 获取最终提示词：传入 input_word="左"，suffix 中的 {input_word} 会被替换成「左」
prompt_text = few_show_prompt.invoke(input={"input_word": "左"}).to_string()

res = LLM.invoke(input=prompt_text)
print(res.content)



chat_propt_template = ChatPromptTemplate.from_messages(
   [
      ("system","你是一个边塞诗人，擅长做七言律诗"),
      ("ai","你好"),
      MessagesPlaceholder("history"),
      ("human","请再来一首唐诗")
   ]
)

history_data = [
   ("human","写一首唐诗"),
   ("ai","床前明月光，疑是地上霜。举头望明月，低头思故乡."),
   ("human","好诗好诗，再来一首"),
   ("ai","千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪.")
]

prompt_text = chat_propt_template.invoke({"history":history_data}).to_string()
print(prompt_text)
res = LLM.invoke(prompt_text)
print(f'AI:{res.content}')
