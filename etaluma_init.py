import os, sys

def init_etaluma_environment():
    stock_path = r"C:\Program Files (x86)\LumaViewPro-stock"
    if os.path.exists(stock_path) and stock_path not in sys.path:
        sys.path.insert(0, stock_path)
    os.environ['JAVA_HOME'] = r"C:\Program Files\Zulu\zulu-8"

if __name__ == '__main__':
    init_etaluma_environment()
