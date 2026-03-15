"""
知识库
"""
import os 
import config_data as config 
import hashlib
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理了
        return False(md5未处理过)，True(已经处理过，有记录)
    """

    if not os.path.exists(config.md5_path):
        open(config.md5_path,"w",encoding="utf-8").close
        return False
    else:
        for line in open(config.md5_path,'r',encoding="utf-8").readlines():
            line = line.strip()  #处理字符串前后的空格和回车
            if line == md5_str:
                return True #已处理过
        return False

def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path,"a",encoding='utf-8') as f:
        f.write(md5_str +"\n")



def get_string_md5(input_str: str,encoding = "utf-8"):
    """将传入的字符串转换为md5字符串"""

    #将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    #创建md5对象
    md5_obj = hashlib.md5()  #得到md5对象
    md5_obj.update(str_bytes) #更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest()  #得到md5的十六进制字符串

    return md5_hex

class KnowledgeBaseService(object):

    def __init__(self):
        #如果文件夹不存在，创建文件夹
        os.makedirs(config.persist_directory,exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name, #数据库表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory  # 数据库本地储存文件
        )   #向量存储的实例，Chroma向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,  #分割后的文本段最大长度
            chunk_overlap = config.chunk_overlap, #连续文本段之间的字符重叠数量
            separators = config.separators,   #自然段划分的符号
            length_function = len,      #使用python自带的len函数，进行长度统计.
        )  #文本分割器的对象


    def upload_by_str(self,data:str,filename):
        """将传入的字符串，进行向量化，存入本地向量数据库"""
        print(f"开始上传文件:{filename},内容长度：{len(data)}")
        #先得到传入字符串的md5的值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]，内容已经存在在知识库中"
        
        #进行文本分割，分割前先判断这段文本是否很长，若很短就不分割了
        #这样做是为了避免不必要的函数调用开销，虽然单词调用分割器的开销很小
        #但是在高并发或批量处理大量短文本时，累计调用开销会变得显著
        #通过提前判断，跳过分割器，提升整体吞吐量
        if len(data) > config.max_split_char_number:
            knowledage_chunks:list[str] = self.spliter.split_text(data)
        else:
            knowledage_chunks = [data]

        metadata = {
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
            "operator" : "小邓"
        }
        self.chroma.add_texts(
            texts= knowledage_chunks,
            metadatas=[metadata for _ in knowledage_chunks]
        )

        #
        save_md5(md5_hex)

        return "[成功] 内容已经成功载入向量库"

if __name__ =="__main__":
    pass
