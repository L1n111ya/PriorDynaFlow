from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

base_url = ''
api_key = ''


def get_llm(model_name: str, temperature: float = 0.1, max_tokens: int = 2048):
    """
    获得LLM
    :param max_tokens:
    :param temperature:
    :param model_name:
    :return llm:
    """
    try:
        llm = ChatOpenAI(model=model_name,
                         base_url=base_url,
                         api_key=api_key,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         # extra_body={"top_k": 1}
                         # extra_body={"enable_search": True}
                         )
        return llm
    except Exception as e:
        raise e


# llm = get_llm("qwen-plus")
# print(llm.invoke("帮我查询最近热门沿海旅游城市的天气，找到2025年5月19-23号天气晴朗的城市，请查中国天气网的天气").content)
# print(llm.invoke([
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="帮我查询最近热门沿海旅游城市的天气，找到2025年5月19-23号天气晴朗的城市，请查中国天气网的天气")
# ]))
