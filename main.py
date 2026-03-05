"""
LangChain 1.2 简单聊天 - 支持多种模型（同一 .env 里可写多个 API Key，不冲突）

优先使用：有 GROQ_API_KEY 用 Groq（免费），否则用 OPENAI_API_KEY。
"""

import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage

from SYSTEM_PROMPT import SYSTEM_PROMPT

# 根据 .env 里已有的 Key 选一个模型（多个 Key 并存时只选一个用，不会冲突）
def get_llm():
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

    if groq_key:
        from langchain_groq import ChatGroq
        return ("Groq", ChatGroq(
            model="llama-3.3-70b-versatile",  # 或 llama-3.1-8b-instant（更快）
            temperature=0.7,
        ))
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ("OpenAI", ChatOpenAI(model="gpt-4o-mini", temperature=0.7))
    return (None, None)

_llm_which, llm = get_llm()


def main():
    groq_set = bool((os.getenv("GROQ_API_KEY") or "").strip())
    openai_set = bool((os.getenv("OPENAI_API_KEY") or "").strip())

    if llm is None:
        print("请在 .env 中至少配置一个 API Key：")
        print("  - GROQ_API_KEY（免费）：https://console.groq.com/keys")
        print("  - 或 OPENAI_API_KEY：https://platform.openai.com/api-keys")
        return

    # 明确显示当前用的是哪个 API，避免和「以为在用 Groq 却实际走 OpenAI」搞混
    print(f"[当前使用] {_llm_which}")
    if _llm_which == "OpenAI":
        print("提示：当前走的是 OpenAI。若想用 Groq，请在 .env 里正确填写 GROQ_API_KEY=你的key \n")
    print("LangChain 简单聊天（输入 quit 或 空行 退出）\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input or user_input.lower() == "quit":
            print("再见！")
            break

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]

        #流式输出
        # for chunk in llm.stream(messages):
        #     print(chunk.content, end="", flush=True)
        # print()

        #正常输出
        response = llm.invoke(messages)
        print(response.content)

if __name__ == "__main__":
    main()
