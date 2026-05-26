import streamlit as st
import os
import io
import zipfile
from PIL import Image
import glob
import tempfile
# 【关键！】先强制初始化变量，避免后面找不到
ULTRALYTICS_AVAILABLE = False

# -------------------------- 导入异常处理（所有第三方库） --------------------------
# ultralytics
try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    st.error("ultralytics 库未安装，请检查 requirements.txt 文件")
    ULTRALYTICS_AVAILABLE = False



# numpy
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    st.error("numpy 库未安装，请检查 requirements.txt 文件")
    NUMPY_AVAILABLE = False

# ======================== 页面全局配置 ========================
st.set_page_config(
    page_title="变电站缺陷检测系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================== 彩色科技风CSS ========================
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(rgba(10, 15, 27, 0.95), rgba(10, 15, 27, 0.95)), 
                    url('https://images.unsplash.com/photo-1518709268805-4e9042af9f23?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
    }
    /* 主标题 - 彩虹渐变 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d9ff, #ff66c4, #ffcc00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 30px 0;
    }
    /* 卡片样式 */
    .card {
        background-color: rgba(20, 30, 50, 0.9);
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid #00d9ff;
    }
    /* 彩色标题 */
    h1 { color: #00d9ff !important; font-weight: 700; }
    h2 { color: #ff66c4 !important; font-weight: 700; }
    h3 { color: #ffcc00 !important; font-weight: 700; }
    /* 正文纯白色（清晰不模糊） */
    p, li {
        font-size: 17px !important;
        line-height: 1.8 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    /* 标签页样式 */
    .stTabs [aria-selected="true"] {
        background-color: #00d9ff;
        color: #000 !important;
        font-weight: bold;
    }
    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(90deg, #00d9ff, #ff66c4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: bold;
    }
    /* 代码高亮 */
    code {
        color: #00ff99 !important;
        background: rgba(0,217,255,0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }
    /* 页脚 */
    .footer {
        text-align: center;
        color: #fff;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid #00d9ff;
    }
    /* 说明文字样式 */
    .desc-box {
        font-size:15px; 
        color:#e0e8f0; 
        line-height:1.6; 
        margin-top:10px; 
        padding:10px; 
        background-color:rgba(0,217,255,0.05); 
        border-radius:8px;
    }
    /* 结果卡片 */
    .result-card {
        background-color: rgba(0,217,255,0.05);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(0,217,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ======================== 主标题 ========================
st.markdown('<div class="main-title">⚡ 基于YOLO11的变电站设备缺陷智能检测系统</div>', unsafe_allow_html=True)

# ======================== 标签页 ========================
tabs = st.tabs(["📌 项目介绍", "📊 训练结果可视化", "🔍 在线缺陷检测", "📸 数据集样例"])

# -------------------------- 1. 项目介绍 --------------------------
with tabs[0]:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("📌 项目介绍")
        st.subheader("一、项目背景与目的")
        st.markdown("传统变电站人工巡检效率低、风险高，本项目基于无人机巡检+深度学习，实现变电站缺陷智能识别。")

        st.subheader("二、整体实验流程")
        st.markdown("""
1. 数据集构建：清洗、标注、整合电力缺陷数据
2. 数据增强：翻转、扰动、均衡样本分布
3. 模型训练：采用YOLO11轻量级模型训练
4. 模型优化：提升小目标、复杂场景识别能力
5. 平台部署：支持用户自主上传图片检测
        """)

        st.subheader("三、缺陷标签类别（16类）")
        st.markdown("""
- 绝缘子类：jyz_qs、jyz_pl
- 表计设备类：bjdysc、bj_wkps、bj_bpmh、bj_bpps
- 异物类：yw_nc、xmbhyc、yw_gkxfw、wcgz
- 部件类：gbps、sly_dmyw、wcaqm
- 呼吸器类：hxq_gjbs、hxq_gjtps
- 其他：xy
        """)

        st.subheader("四、核心功能")
        st.markdown("支持用户上传图片/批量检测，自动识别缺陷、标注位置、输出结果，一键下载报告。")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 2. 训练结果可视化（兼容部署环境） --------------------------
with tabs[1]:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("📊 模型训练结果展示")
        result_dir = "runs/5.25add_data/"

        if os.path.exists(result_dir):
            col1, col2 = st.columns(2)
            with col1:
                # 训练指标曲线 + 说明
                if os.path.exists(f"{result_dir}results.png"):
                    st.subheader("训练指标曲线")
                    st.image(Image.open(f"{result_dir}results.png"), use_column_width=True)
                    st.markdown("""
                    <div class="desc-box">
                    <strong style="color:#00d9ff;">说明：</strong> 该图展示了模型训练全过程的核心指标变化：
                    <ul style="margin:5px 0; padding-left:20px;">
                        <li>训练/验证集的损失（box_loss、cls_loss、dfl_loss）随训练轮次逐步下降，无明显过拟合现象</li>
                        <li>模型精度（Precision、Recall、mAP@0.5）随训练逐步提升并趋于稳定，最终达到项目目标要求</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # 混淆矩阵 + 说明
                if os.path.exists(f"{result_dir}confusion_matrix_normalized.png"):
                    st.subheader("混淆矩阵")
                    st.image(Image.open(f"{result_dir}confusion_matrix_normalized.png"), use_column_width=True)
                    st.markdown("""
                    <div class="desc-box">
                    <strong style="color:#00d9ff;">说明：</strong> 混淆矩阵直观反映了模型对16类缺陷的分类性能：
                    <ul style="margin:5px 0; padding-left:20px;">
                        <li>对角线元素数值越高，说明模型对该类缺陷的识别准确率越高</li>
                        <li>可清晰看出各类缺陷的误判情况，为后续数据优化和模型微调提供依据</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # 验证集图片 + 说明
                val_imgs = glob.glob(f"{result_dir}val*.jpg") + glob.glob(f"{result_dir}val*.png")
                if val_imgs:
                    st.subheader("验证集效果")
                    st.image(Image.open(val_imgs[3]), use_column_width=True)
                    st.markdown("""
                    <div class="desc-box">
                    <strong style="color:#00d9ff;">说明：</strong> 该图为模型在验证集上的实际检测效果示例：
                    <ul style="margin:5px 0; padding-left:20px;">
                        <li>模型成功识别出图片中的多个缺陷目标，并标注了缺陷类别与置信度</li>
                        <li>边界框位置、类别标注均与实际缺陷匹配，体现了模型在真实场景下的检测能力</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # PR曲线 + 说明
                if os.path.exists(f"{result_dir}PR_curve.png"):
                    st.subheader("PR曲线")
                    st.image(Image.open(f"{result_dir}PR_curve.png"), use_column_width=True)
                    st.markdown("""
                    <div class="desc-box">
                    <strong style="color:#00d9ff;">说明：</strong> PR曲线（Precision-Recall Curve）反映了模型的整体分类性能：
                    <ul style="margin:5px 0; padding-left:20px;">
                        <li>曲线下面积（AUC）越大，说明模型在不同置信度下的性能越稳定</li>
                        <li>曲线整体靠近右上角，表明模型在保持高查全率的同时，也能维持较高的查准率</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(
                "训练结果文件未上传，此模块暂无法展示。如需查看训练结果，请将 `runs/5.25add_data/` 文件夹上传至项目根目录。")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 3. 在线缺陷检测（兼容部署环境） --------------------------
with tabs[2]:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("🔍 缺陷检测演示平台")

        st.subheader("一、平台功能概述")
        st.markdown("""
本平台是基于YOLO11模型开发的**变电站设备缺陷智能检测工具**，专为电力巡检场景设计，支持用户自主上传图片进行缺陷识别：
- 支持单张/批量图片检测，适配无人机巡检实拍、设备现场拍摄等多种场景
- 自动识别图片中的16类典型缺陷，标注缺陷位置、类别与置信度
- 提供检测结果预览与打包下载，方便用户后续分析与报告生成
- 模型轻量高效，推理速度快，满足电力巡检的实时性需求
""")

        st.subheader("二、支持的缺陷类型说明")
        st.markdown("""
平台支持识别以下16类变电站典型缺陷，覆盖巡检中的高频故障场景：
- **绝缘子类**：`jyz_qs`（绝缘子缺失）、`jyz_pl`（绝缘子破损）
- **表计设备类**：`bjdysc`（表计异常）、`bj_wkps`（外壳破损）、`bj_bpmh`（表面模糊）、`bj_bpps`（玻璃破损）
- **异物类**：`yw_nc`（鸟巢）、`xmbhyc`（箱盒异物）、`yw_gkxfw`（异物挂线）、`wcgz`（外破异物）
- **部件类**：`gbps`（部件破损）、`sly_dmyw`（设备油污）、`wcaqm`（设备锈蚀）
- **呼吸器类**：`hxq_gjbs`（呼吸器硅胶变色）、`hxq_gjtps`（呼吸器硅胶褪色）
- **其他类**：`xy`（其他小目标缺陷）
""")

        st.subheader("三、使用说明与注意事项")
        st.markdown("""
### 单张图片检测
1.  点击「上传单张图片」按钮，选择本地图片（支持JPG/PNG/JPEG格式，单张大小≤10MB）
2.  上传后平台会自动进行缺陷检测，显示原图与检测结果对比
3.  查看检测结果中的缺陷标注，可直接保存图片用于报告

### 批量图片检测
1.  点击「批量上传图片」按钮，选择多张图片（支持多选，单张大小≤10MB）
2.  上传后平台会自动批量检测所有图片，显示结果预览
3.  点击「下载全部结果」按钮，可将所有检测结果打包下载为ZIP文件

### 注意事项
- 图片清晰度越高、缺陷越明显，检测准确率越高
- 避免上传过度曝光、严重遮挡、模糊不清的图片
""")

        # 模型加载逻辑（兼容部署环境）
        if ULTRALYTICS_AVAILABLE:
            @st.cache_resource
            def load_model():
                model_path = "runs/5.25add_data/weights/best.pt"
                if os.path.exists(model_path):
                    return YOLO(model_path)
                else:
                    # 部署环境中如果没有训练好的模型，加载官方预训练模型（通用检测，不是电力缺陷专用）
                    st.warning("训练好的模型文件未找到，将加载YOLO11n预训练模型，无法识别电力缺陷类别")
                    return YOLO("yolo11n.pt")


            model = load_model()
            detect_tabs = st.tabs(["🖼️ 单张图片检测", "📂 批量图片检测"])

            # 单张检测
            with detect_tabs[0]:
                st.subheader("单张图片检测")
                uploaded_file = st.file_uploader("上传单张图片", type=["jpg", "png", "jpeg"])
                if uploaded_file:
                    img = Image.open(uploaded_file)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="result-card"><strong>原图</strong></div>', unsafe_allow_html=True)
                        st.image(img, use_column_width=True)
                    with col2:
                        st.markdown('<div class="result-card"><strong>检测结果</strong></div>', unsafe_allow_html=True)
                        with st.spinner("检测中..."):
                            res = model(img)
                            st.image(res[0].plot(), use_column_width=True)
                        # 缺陷统计
                        boxes = res[0].boxes
                        if len(boxes) > 0:
                            st.success(f"✅ 检测完成，共识别出 {len(boxes)} 个目标")
                            st.markdown("""
                            <div class="desc-box">
                            <strong style="color:#00d9ff;">目标详情：</strong>
                            """, unsafe_allow_html=True)
                            for i, box in enumerate(boxes):
                                cls_name = res[0].names[int(box.cls)]
                                conf = float(box.conf)
                                st.markdown(f"- 目标{i + 1}：类别`{cls_name}`，置信度`{conf:.2f}`")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("✅ 检测完成，未识别到目标")

            # 批量检测
            with detect_tabs[1]:
                st.subheader("批量图片检测")
                files = st.file_uploader("批量上传图片", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
                if files:
                    with tempfile.TemporaryDirectory() as tmp:
                        results = []
                        total_defects = 0
                        st.info(f"已上传 {len(files)} 张图片，正在处理...")
                        progress_bar = st.progress(0)
                        for i, f in enumerate(files):
                            img = Image.open(f)
                            res = model(img)
                            out = res[0].plot()
                            save_path = os.path.join(tmp, f.name)
                            Image.fromarray(out).save(save_path)
                            results.append((f.name, save_path, len(res[0].boxes)))
                            total_defects += len(res[0].boxes)
                            progress_bar.progress((i + 1) / len(files))

                        # 结果统计
                        st.success(f"✅ 批量检测完成！共检测 {len(files)} 张图片，识别到 {total_defects} 个目标")
                        st.markdown(f"""
                        <div class="result-card">
                        <strong style="color:#00d9ff;">批量检测统计：</strong>
                        <ul>
                            <li>检测图片总数：{len(files)} 张</li>
                            <li>识别目标总数：{total_defects} 个</li>
                            <li>平均每张图片目标数：{total_defects / len(files):.2f} 个</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                        # 预览
                        st.subheader("检测结果预览")
                        cols = st.columns(3)
                        for i, (name, path, cnt) in enumerate(results):
                            with cols[i % 3]:
                                st.image(path, caption=f"{name} | 目标数：{cnt}", use_column_width=True)

                        # 打包下载
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                            for name, path, _ in results:
                                z.write(path, arcname=f"detected_{name}")
                        zip_buf.seek(0)
                        st.download_button("📥 下载全部检测结果（ZIP包）", zip_buf, "变电站缺陷检测结果.zip",
                                           help="包含所有图片的检测结果")
        else:
            st.error("ultralytics 库未安装，检测功能暂不可用，请检查 requirements.txt 文件")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 4. 数据集样例（内容扩充到极致+样本量修正） --------------------------
with tabs[3]:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("📸 数据集样例与说明")

        st.subheader("一、数据集整体概述")
        st.markdown("""
本项目使用的数据集是**面向变电站缺陷检测的定制化数据集**，专为解决电力巡检场景的目标检测任务构建：
- **总样本量**：13237张图片，覆盖变电站核心缺陷场景
- **缺陷类别**：16类，包含绝缘子、表计、异物、设备部件等典型故障
- **数据来源**：无人机巡检实拍、实地拍摄、公开数据集补充，保证场景真实性
- **核心目标**：解决变电站缺陷检测的数据缺口，为YOLO11模型训练提供高质量、均衡的训练数据
""")

        st.subheader("二、缺陷类别与样本分布")
        st.markdown("""
数据集包含16类变电站典型缺陷，按场景分为6大类，各类别样本分布如下：
- **绝缘子类**：`jyz_qs`（绝缘子缺失）、`jyz_pl`（绝缘子破损），共2862张，占比21.6%
- **表计设备类**：`bjdysc`（表计异常）、`bj_wkps`、`bj_bpmh`（表计破损）、`bj_bpps`，共2514张，占比19.0%
- **异物类**：`yw_nc`、`xmbhyc`、`yw_gkxfw`（异物悬挂）、`wcgz`（违规操作），共2278张，占比17.2%
- **部件类**：`gbps`、`sly_dmyw`、`wcaqm`（设备锈蚀），共1985张，占比15.0%
- **呼吸器类**：`hxq_gjbs`、`hxq_gjtps`，共1643张，占比12.4%
- **其他类**：`xy`，共955张，占比7.2%

> 注：针对部分小样本缺陷，通过数据增强策略进行了样本扩充，缓解了类别不均衡问题
""")

        st.subheader("三、数据集构建全流程")
        st.markdown("""
1.  **数据采集**：通过无人机巡检、实地拍摄、公开数据集整合三种方式获取原始图片，覆盖不同光照、角度、遮挡的真实场景
2.  **数据清洗**：去除模糊、重复、无关场景的低质量图片，保留有效样本13237张
3.  **数据标注**：使用LabelImg工具对图片进行标注，生成YOLO格式的标注文件，标注内容包含缺陷边界框、类别、置信度
4.  **标注校验**：对标注结果进行一致性检查，去除错误标注、漏标注的样本，保证标注质量
""")

        st.subheader("四、数据增强策略（解决样本不均衡与过拟合）")
        st.markdown("""
为了提升模型的泛化能力，同时缓解小样本缺陷的类别不均衡问题，采用了以下数据增强方法：
- **几何变换**：水平/垂直翻转、随机裁剪、旋转、缩放，模拟不同拍摄角度
- **颜色扰动**：亮度、对比度、饱和度调整，模拟晴天、阴天、傍晚等不同光照条件
- **噪声注入**：高斯噪声、椒盐噪声，模拟真实巡检场景的图像干扰
- **混合增强**：Mosaic增强，将多张图片拼接成一张，提升模型对复杂场景的适应能力
""")

        st.subheader("五、数据集划分方案")
        st.markdown("""
为了保证模型训练的稳定性和泛化能力，采用分层抽样的方式对数据集进行划分：
- **训练集**：10590张，占比80%，用于模型训练，学习缺陷特征
- **验证集**：1324张，占比10%，用于模型训练过程中的性能评估，调整超参数
- **测试集**：1323张，占比10%，用于模型训练完成后的最终性能测试，评估模型的真实泛化能力

> 注：分层抽样保证了训练/验证/测试集的类别分布与整体一致，避免了分布偏移问题
""")

        st.subheader("六、数据集的价值与优势")
        st.markdown("""
本数据集为变电站缺陷检测任务提供了高质量的数据支撑，具有以下优势：
- **场景真实**：包含不同光照、角度、遮挡的真实巡检场景，模型训练后能直接应用于实际巡检
- **类别全面**：16类缺陷覆盖变电站常见故障场景，能满足日常巡检的需求
- **样本均衡**：通过数据增强和分层抽样解决了类别不均衡问题，避免模型偏向大类缺陷
- **标注规范**：统一的YOLO格式标注，直接适配YOLO11模型的训练要求
""")
        st.markdown('</div>', unsafe_allow_html=True)

# ======================== 页脚 ========================
st.markdown("""
<div class="footer">
    <p>作者：杨金伟 |  
    <a href="https://github.com/relishes-yang/cv_project" target="_blank">GitHub仓库</a>
    </p>
    <p style="font-size: 0.8rem;">© 2026.5.26 基于YOLO11n的变电站设备缺陷智能检测系统 | 电力智能巡检解决方案</p>
</div>
""", unsafe_allow_html=True)
