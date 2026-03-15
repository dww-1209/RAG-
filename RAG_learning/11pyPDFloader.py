# pyPDFLoader 用于加载PDF文件，提取文本内容并将其转换为Document对象。
# 它支持处理PDF文件中的文本、图像和表格等内容，并提供了多种选项来控制加载过程，例如是否保留原始格式、是否提取元数据等。

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="RAG_learning/data/sample.pdf",   #PDF文件路径
    mode="page",           #读取模式，默认为"page","page"(按页面划分不同document)，"single"(将整个PDF作为一个document)."element"(按元素划分不同document)，
                           #"image"提取图像内容，"table"提取表格内容等
    password="password",       #如果PDF文件受密码保护，可以提供密码来解密文件
)