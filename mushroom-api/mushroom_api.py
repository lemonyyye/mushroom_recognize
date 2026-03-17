# -*- coding: utf-8 -*-
"""
蘑菇毒性识别接口（最终版）
依赖：fastapi, uvicorn, pandas, scikit-learn, joblib, opencv-python, numpy
"""
import cv2
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== 1. 初始化FastAPI应用 =====================
app = FastAPI(title="蘑菇毒性识别接口", version="1.0")

# 配置跨域（解决前端跨域问题）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名（测试用）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 2. 加载模型和编码器 =====================
# 确保这3个文件和当前脚本在同一目录
MODEL_PATH = "data/mushroom_tree_model.pkl"
FEATURE_ENCODER_PATH = "data/feature_encoder.pkl"
TARGET_ENCODER_PATH = "data/target_encoder.pkl"

# 全局变量
model = None
feature_encoder = None
target_encoder = None

def load_model():
    """加载模型和编码器"""
    global model, feature_encoder, target_encoder
    try:
        model = joblib.load(MODEL_PATH)
        feature_encoder = joblib.load(FEATURE_ENCODER_PATH)
        target_encoder = joblib.load(TARGET_ENCODER_PATH)
        print("✅ 模型加载成功")
    except FileNotFoundError as e:
        raise RuntimeError(f"❌ 缺失文件：{e}\n请先运行train_mushroom_model.py生成pkl文件")

# 启动时加载模型
load_model()

# ===================== 3. 核心函数 =====================
def predict_toxicity(features: dict) -> dict:
    """预测蘑菇毒性"""
    # 必选特征列表（和数据集一致）
    required_features = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    # 检查缺失特征
    missing = [f for f in required_features if f not in features]
    if missing:
        raise HTTPException(status_code=400, detail=f"缺失特征：{missing}")
    
    # 特征编码
    try:
        input_df = pd.DataFrame([features])[required_features]
        input_encoded = feature_encoder.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"特征编码失败：{str(e)}")
    
    # 模型预测
    pred_code = model.predict(input_encoded)[0]
    pred_prob = model.predict_proba(input_encoded)[0].max()
    pred_label = target_encoder.inverse_transform([pred_code])[0]
    
    return {
        "isPoison": pred_label == "p",  # True=有毒，False=可食用
        "reason": f"该蘑菇{'有毒（Poisonous）' if pred_label == 'p' else '可食用（Edible）'}，置信度：{pred_prob:.4f}",
        "confidence": float(pred_prob)
    }
def extract_features_from_img(img: np.ndarray) -> dict:
    """从图片提取真实特征（简易版）"""
    # 1. 把图片从BGR（cv2默认）转成RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. 提取图片主颜色（简化版：取中心像素的颜色）
    h, w = img_rgb.shape[:2]
    center_pixel = img_rgb[h//2, w//2]  # 取图片中心像素的RGB值
    r, g, b = center_pixel
    
    # 3. 根据主颜色匹配菌盖颜色（cap-color）
    # 数据集里的cap-color取值：n(棕), y(黄), w(白), r(红), g(绿)等
    cap_color = "n"  # 默认棕色
    if r > 200 and g < 100 and b < 100:
        cap_color = "r"  # 红色→有毒特征
    elif r > 200 and g > 200 and b < 100:
        cap_color = "y"  # 黄色→可食用特征
    elif r > 200 and g > 200 and b > 200:
        cap_color = "w"  # 白色→有毒特征
    elif r < 100 and g > 200 and b < 100:
        cap_color = "g"  # 绿色→可食用特征
    
    # 4. 提取图片轮廓（简化版：判断菌盖形状）
    # 转灰度图→二值化→找轮廓
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 数据集里的cap-shape取值：x(凸), b(钟形), s(平), f(漏斗)
    cap_shape = "x"  # 默认凸形
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # 计算轮廓的宽高比，简单判断形状
        x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
        aspect_ratio = w_cont / h_cont
        if aspect_ratio > 1.2:
            cap_shape = "s"  # 宽>高→平形→可食用
        elif aspect_ratio < 0.8:
            cap_shape = "b"  # 高>宽→钟形→有毒
    
    # 5. 根据颜色/形状决定气味（odor，核心毒性特征）
    # odor="p"=有毒，odor="a"=可食用
    odor = "p"  # 默认有毒
    if (cap_color == "y" or cap_color == "g") and cap_shape == "s":
        odor = "a"  # 黄/绿色+平形→可食用
    
    # 6. 构造完整特征（其他特征用默认，核心改odor/cap-color/cap-shape）
    features = {
        "cap-shape": cap_shape,      # 动态提取的菌盖形状
        "cap-surface": "s",          # 固定值
        "cap-color": cap_color,      # 动态提取的菌盖颜色
        "bruises": "t",              # 固定值
        "odor": odor,                # 核心：决定毒性的特征
        "gill-attachment": "f",
        "gill-spacing": "c",
        "gill-size": "n",
        "gill-color": "k",
        "stalk-shape": "e",
        "stalk-root": "e",
        "stalk-surface-above-ring": "s",
        "stalk-surface-below-ring": "s",
        "stalk-color-above-ring": "w",
        "stalk-color-below-ring": "w",
        "veil-color": "w",
        "ring-number": "o",
        "ring-type": "p",
        "spore-print-color": "k",
        "population": "s",
        "habitat": "u"
    }
    return features

# ===================== 4. 处理 OPTIONS 预请求（关键修复） =====================
@app.options("/predict-by-image")
async def handle_options_request():
    """处理跨域预请求，避免浏览器阻塞"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Content-Length",
        }
    )

# ===================== 4. 定义接口 =====================
# 接口1：上传图片识别（前端主要调用）
@app.post("/predict-by-image", summary="上传图片识别蘑菇毒性")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 读取图片
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="图片解析失败")
        
        # 提取特征 + 预测
        features = extract_features_from_img(img)
        result = predict_toxicity(features)
        
        return {
            "code": 200,
            "message": "识别成功",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败：{str(e)}")

# 接口2：手动输入特征测试（调试用）
@app.post("/predict-by-features", summary="输入特征测试")
def predict_features(features: dict):
    result = predict_toxicity(features)
    return {"code": 200, "message": "预测成功", "data": result}


# ===================== 5. 启动服务 =====================
if __name__ == "__main__":
    import uvicorn
    # 启动服务：http://0.0.0.0:8001
    uvicorn.run("mushroom_api:app", host="0.0.0.0", port=8001, reload=True) 

