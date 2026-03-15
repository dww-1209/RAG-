# 文档加载器提供了一套标准接口，将不同来源（CSV、PDF、JSON等）的文档统一加载为
# 可Langchain处理的格式，确保了数据的一致性和可用性。
# 通过使用文档加载器，开发者可以轻松地将各种类型的文档集成到Langchain应用中.
#文档加载器需要实现BaseLoader接口，主要方法是load()，它负责从指定来源加载文档并返回一个文档列表。

#clss Document 是Langchain内文档的统一载体，所有文档加载器最终返回此类的实例
# load()方法一次性加载所有文档，适用于小型数据集；lazy_load()方法则是一个生成器，适用于大型数据集，可以逐步加载文档以节省内存。

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="RAG_learning/data/stu.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": ",",   #指定分隔符
        "quotechar": '"',   #指定引用字符
        "fieldnames": ["a","b","c","d"]  #指定表头名称，一般在数据没有表头的时候使用，从而自定义表头名称
    }
)
# #批量加载，.loade（）-> list[Document]
# documents = loader.load()
# for doc in documents:
#     print(doc)

#逐步加载，.lazy_load() -> Generator[Document]
for doc in loader.lazy_load():
    print(doc)