## 项目正确运行方式
1. 使用虚拟环境及装包：
```bash
# 1. 创建虚拟环境
python3 -m venv .venv
# 2.1 激活（macOS / bash / zsh）
source .venv/bin/activate
# 2.2 激活（cmd）
.\.venv\Scripts\activate.bat
# 3. 可选：升级 pip
python -m pip install --upgrade pip
# 4. 安装 requirements.txt里所有的包
python -m pip install -r requirements.txt
# 5. 运行接口
python mushroom_api.py
```

2. 在安装新包时：
```bash
pip install 包名
# 安装时将包记录在requirements.txt
pip freeze > requirements.txt
```

3. 查看本地接口文档
`http://localhost:8001/docs`
