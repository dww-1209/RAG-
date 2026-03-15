#将组件串联，上一个组件的输出作为下一个组件的输入，实现数据的自动化流转和协同工作.

from langchain_groq import  ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()


LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

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

#组成chian,要求每一个组件都是Runnalbe的实例，chain会自动将上一个组件的输出作为下一个组件的输入

chain = chat_propt_template | LLM
res = chain.invoke(input={"history": history_data})

print(res.content)