#jq是一个跨平台的命令行JSON处理工具，可以用来解析、过滤、转换和格式化JSON数据。它提供了强大的功能，使得处理JSON数据变得非常方便和高效。
#Langchain底层对JSON数据的处理也非常依赖jq，特别是在处理复杂的JSON结构时，jq可以帮助我们轻松地提取和转换数据，使得Langchain能够更好地理解和利用这些数据。
#将JSON数据信息抽取出来，转换成Document对象，供Langchain使用。

from langchain_community.document_loaders import JSONLoader


loader = JSONLoader(
    file_path="RAG_learning/data/stu.json",   #JSON文件路径
    jq_schema=".",           #jq表达式，用于指定如何从JSON数据中提取信息并转换成Document对象
    text_content=False,         #默认为False，保留原本数据结构和类型，如果设置为True，则会将提取的的值转换为字符串形式
)

documents = loader.load()
print(type(documents))  
print(documents[0].page_content) 

loader = JSONLoader(
    file_path="RAG_learning/data/stus.json",   #JSON文件路径
    jq_schema=".[]",           #jq表达式，用于指定如何从JSON数据中提取信息并转换成Document对象
    text_content=False,         #默认为False，保留原本数据结构和类型，如果设置为True，则会将提取的的值转换为字符串形式
)

documents = loader.load()
print(type(documents))  
print(documents[0].page_content)

loader = JSONLoader(
    file_path="RAG_learning/data/stus_json_lines.json",   #JSON文件路径
    jq_schema=".",           #jq表达式，用于指定如何从JSON数据中提取信息并转换成Document对象
    text_content=False,         #默认为False，保留原本数据结构和类型，如果设置为True，则会将提取的的值转换为字符串形式
    json_lines=True         #是否将JSON文件视为JSON Lines格式，默认为False，如果设置为True，则每行都被视为一个独立的JSON对象，并分别转换成Document对象
)

documents = loader.load()
print(type(documents))  
print(documents[0].page_content)        