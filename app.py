import streamlit as st
import cv2
import numpy as np
from PIL import Image
from analyzer import FaceAnalyzer

@st.cache_resource
def get_analyzer():
    return FaceAnalyzer()

st.set_page_config(page_title="Face Analyzer", page_icon="👤", layout="wide")

# 透過 CSS 將版面寬度限制在大約 2/3
st.markdown(
    """
    <style>
    .block-container {
        max-width: 66vw !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("👤 Face Analyzer")
st.write("上傳一張正面人像照片，我們將為您分析臉型、眼型、眉型、鼻型與嘴唇。您可以多次上傳並保留多筆分析紀錄。")

uploaded_file = st.file_uploader("選擇一張圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # 轉換成 BGR (OpenCV 預設格式)
    if len(image_np.shape) == 2: # Grayscale
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4: # RGBA
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else: # RGB
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="目前上傳的圖片", width=350)
    
    if st.button("開始分析並加入紀錄"):
        with st.spinner("分析中..."):
            analyzer = get_analyzer()
            result = analyzer.analyze(image_cv2)
            
        if result[0] is None:
            st.error("❌ 找不到臉部！請確認照片中包含清晰的正面人臉，光線充足且沒有被遮擋。")
        else:
            st.success("✅ 分析完成！已加入下方紀錄中。")
            analysis, annotated_image = result
            
            # 將 OpenCV 的 BGR 轉回 RGB 以供 Streamlit 顯示
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # 儲存到歷史紀錄
            st.session_state.history.append({
                "file_name": uploaded_file.name,
                "annotated_image": annotated_image_rgb,
                "analysis": analysis
            })

# 顯示所有歷史紀錄
if st.session_state.history:
    st.markdown("---")
    st.header("📚 分析紀錄")
    
    # 清除紀錄按鈕
    if st.button("清除所有紀錄"):
        st.session_state.history = []
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        st.subheader("📈 長度統計趨勢")
        st.write("橫軸為樣本編號（依上傳順序），縱軸為長度（單位：眼距）。")
        
        import pandas as pd
        chart_data = {
            "眼睛寬度": [r["analysis"]["eye_width_eye_dist_ratio"] for r in st.session_state.history],
            "鼻翼寬度": [r["analysis"]["nose_eye_ratio"] for r in st.session_state.history],
            "嘴唇寬度": [r["analysis"]["mouth_eye_ratio"] for r in st.session_state.history]
        }
        df = pd.DataFrame(chart_data)
        df.index = range(1, len(df) + 1)  # 樣本編號從 1 開始
        
        chart_col, stat_col = st.columns([2, 1])
        with chart_col:
            import altair as alt
            df_long = df.reset_index().melt('index', var_name='測量項目', value_name='比例 (倍眼距)')
            chart = alt.Chart(df_long).mark_line(point=True).encode(
                x=alt.X('index:O', title='樣本編號', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('比例 (倍眼距):Q', scale=alt.Scale(zero=False)),
                color='測量項目:N',
                tooltip=['index', '測量項目', '比例 (倍眼距)']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
        with stat_col:
            st.markdown("**📊 統計數據**")
            for key in chart_data.keys():
                mean_val = df[key].mean()
                std_val = df[key].std() if len(df) > 1 else 0.0
                st.write(f"**{key}**")
                st.write(f"- 平均: {mean_val:.2f}")
                st.write(f"- 標準差: {std_val:.2f}")
        st.markdown("---")
        
        if st.toggle("顯示五官類型分布圖表 (長條圖)"):
            st.subheader("📊 五官類型分布")
            st.write("以下顯示目前所有樣本中各種類型的數量統計。")
            
            features = ["face_shape", "eye_shape", "eyebrow_shape", "nose_shape", "lips"]
            titles = ["臉型分布", "眼型分布", "眉型分布", "鼻型分布", "嘴唇分布"]
            
            tabs = st.tabs(titles)
            
            possible_categories = {
                "face_shape": ["長臉 (Long)", "心形臉 (Heart)", "方臉 (Square)", "圓臉 (Round)", "橢圓臉 (Oval)"],
                "eye_shape": ["上揚眼 (Upturned)", "下垂眼 (Downturned)", "圓眼 (Round)", "細長眼 (Slender)", "杏仁眼 (Almond)"],
                "eyebrow_shape": ["拱眉 (Arched)", "上揚眉 (Upturned)", "下垂眉 (Downturned)", "平眉 (Straight)"],
                "nose_shape": ["寬鼻 (Wide)", "窄鼻 (Narrow)", "高鼻樑 (High bridge)", "低鼻樑 (Low bridge)", "中等鼻 (Medium)"],
                "lips": ["上唇較厚 (Thicker upper lip)", "下唇較厚 (Thicker lower lip)", "厚唇 (Thick)", "薄唇 (Thin)", "中等唇 (Medium)"]
            }
            
            for tab, feature, title in zip(tabs, features, titles):
                with tab:
                    counts = {cat: 0 for cat in possible_categories[feature]}
                    for r in st.session_state.history:
                        val = r["analysis"].get(feature, "未知")
                        counts[val] = counts.get(val, 0) + 1
                    
                    if counts:
                        df_counts = pd.DataFrame(list(counts.items()), columns=["類型", "數量"])
                        bar_chart = alt.Chart(df_counts).mark_bar().encode(
                            x=alt.X("類型:N", title="類型", axis=alt.Axis(labelAngle=0, labelLimit=0)),
                            y=alt.Y("數量:Q", title="數量", axis=alt.Axis(tickMinStep=1)),
                            color=alt.Color("類型:N", legend=None),
                            tooltip=["類型", "數量"]
                        ).properties(height=300)
                        st.altair_chart(bar_chart, use_container_width=True)
                    else:
                        st.write("尚無資料")
                        
            st.markdown("---")
    
    # 反向迴圈以將最新的紀錄顯示在最上面
    for i, record in enumerate(reversed(st.session_state.history)):
        real_idx = len(st.session_state.history) - 1 - i
        record_idx = real_idx + 1
        
        header_col, btn_col = st.columns([5, 1])
        with header_col:
            st.subheader(f"紀錄 #{record_idx}: {record['file_name']}")
        with btn_col:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ 刪除", key=f"del_{real_idx}"):
                st.session_state.history.pop(real_idx)
                st.rerun()
        
        col_img, col_data = st.columns([1, 1.2])
        
        with col_img:
            st.image(record["annotated_image"], caption="標註後的圖片", use_container_width=True)
            
        with col_data:
            analysis = record["analysis"]
            
            st.markdown("**🔹 五官類型**")
            types_data = {
                "特徵": ["臉型", "眼型", "眉型", "鼻型", "嘴唇"],
                "分類結果": [
                    analysis.get('face_shape', '未知'),
                    analysis.get('eye_shape', '未知'),
                    analysis.get('eyebrow_shape', '未知'),
                    analysis.get('nose_shape', '未知'),
                    analysis.get('lips', '未知')
                ]
            }
            st.table(types_data)
            
            st.markdown("**🔹 比例量測 (以眼距為單位)**")
            measurements_data = {
                "量測項目": ["眼睛寬度", "鼻翼寬度", "嘴唇寬度"],
                "比例 (倍眼距)": [
                    analysis.get('eye_width_eye_dist_ratio', '未知'),
                    analysis.get('nose_eye_ratio', '未知'),
                    analysis.get('mouth_eye_ratio', '未知')
                ]
            }
            st.table(measurements_data)
            
        st.markdown("---")
