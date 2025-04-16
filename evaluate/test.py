import os

def list_directory_contents(path):
    """
    列出指定路径下的所有文件和子目录
    :param path: 要查看的路径
    """
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return
    
    # 检查是否是目录
    if not os.path.isdir(path):
        print(f"这不是一个目录: {path}")
        return
    
    print(f"\n目录内容列表: {path}")
    print("=" * 50)
    
    # 获取目录下所有内容并排序
    contents = sorted(os.listdir(path))
    
    # 打印每个项目并标注是文件还是目录
    for item in contents:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            print(f"[目录] {item}")
        else:
            print(f"[文件] {item}")
    
    print(f"\n总计: {len(contents)} 个项目")

# 使用示例
if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname("../dataset")
    
    # 查看当前目录
    list_directory_contents("../dataset/CASIA")
    
    # 查看上级目录