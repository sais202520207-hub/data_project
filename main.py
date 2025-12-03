import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def load_data(file_path):
    """
    CSV 파일에서 데이터를 로드합니다.
    """
    try:
        # 파일 경로가 'fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv'임을 가정
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return None

def preprocess_data(df):
    """
    데이터를 전처리하고 분석에 사용할 숫자형 데이터프레임을 반환합니다.
    """
    st.subheader("📊 데이터 전처리")

    # 1. 분석에 필요한 열 선택 및 숫자형으로 변환
    # '신장', '체중', '체지방율', '허리둘레', '악력_좌', '악력_우', '윗몸말아올리기', '제자리 멀리뛰기', 'BMI' 등을 포함
    # 분석에 부적합하거나 결측치가 많은 열은 제외합니다.
    numerical_cols = ['신장', '체중', '체지방율', '허리둘레', '이완기혈압_최저', '수축기혈압_최고', 
                      '악력_좌', '악력_우', '윗몸말아올리기', '제자리 멀리뛰기', 'BMI', 
                      '상대악력', '허리둘레-신장비', '반복옆뛰기']
    
    # 데이터프레임에서 위 열들만 선택
    df_numeric = df[numerical_cols].copy()

    # 2. 숫자형으로 변환 (오류가 있으면 NaN으로 처리)
    for col in numerical_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # 3. 결측값 처리 (간단하게 평균값으로 대체)
    df_numeric.fillna(df_numeric.mean(), inplace=True)
    
    st.write(f"**전처리 후 사용 가능한 숫자형 데이터 수:** {len(df_numeric)}")
    st.dataframe(df_numeric.head())
    
    return df_numeric

def analyze_and_visualize(df_numeric):
    """
    상관관계 분석 및 시각화를 수행합니다.
    """
    
    # '체지방율' 열이 있는지 확인
    if '체지방율' not in df_numeric.columns:
        st.error("데이터에 '체지방율' 열이 없습니다. 컬럼 이름을 확인해주세요.")
        return

    # --- 1. 상관관계 분석 ---
    st.header("🔍 체지방율 상관관계 분석")
    correlation_matrix = df_numeric.corr()
    
    # '체지방율'과의 상관관계만 추출하고 절대값 기준으로 내림차순 정렬
    target_corr = correlation_matrix['체지방율'].sort_values(ascending=False).drop('체지방율')
    
    st.markdown("### 🥇 체지방율과 상관관계가 가장 높은 속성")
    st.dataframe(target_corr.abs().sort_values(ascending=False).head(10))
    
    highest_corr_features = target_corr.abs().sort_values(ascending=False).head(3).index.tolist()
    st.success(f"**체지방율**과 **절대값 기준**으로 상관관계가 가장 높은 상위 3개 속성: **{', '.join(highest_corr_features)}**")

    # --- 2. 히트맵 시각화 ---
    st.header("🔥 전체 데이터 상관관계 히트맵")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    # 상관관계가 0.7 이상이면 진한 색으로 표시
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, ax=ax, 
                cbar_kws={'label': '상관관계 계수'})
    ax.set_title('전체 속성 간 상관관계 히트맵', fontsize=16)
    st.pyplot(fig)
    
    st.markdown("---")

    # --- 3. 산점도 시각화 ---
    st.header(f"📉 체지방율 vs. 상위 상관관계 속성 산점도")

    for feature in highest_corr_features:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 상관관계 값 가져오기 (양의 상관관계 vs 음의 상관관계)
        corr_value = target_corr[feature]
        
        # 색상 및 추세선 설정
        color = 'red' if corr_value < 0 else 'blue'
        
        # 산점도 그리기
        sns.scatterplot(x=df_numeric[feature], y=df_numeric['체지방율'], ax=ax, color=color, alpha=0.6)
        
        # 추세선 추가
        sns.regplot(x=df_numeric[feature], y=df_numeric['체지방율'], scatter=False, color='gray', ax=ax)
        
        ax.set_title(f"체지방율 vs. {feature} (상관관계: {corr_value:.2f})", fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('체지방율', fontsize=12)
        st.pyplot(fig)


def main():
    st.set_page_config(layout="wide", page_title="운동 데이터 분석 웹사이트")
    st.title("🏃‍♀️ 운동 데이터 분석 웹사이트")
    st.markdown("파일 **`fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv`**을 분석합니다.")

    # 파일 경로 (사용자가 업로드한 파일 이름을 사용)
    file_path = "fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv"
    
    # 1. 데이터 로드
    df = load_data(file_path)

    if df is not None:
        st.subheader("📄 원본 데이터 미리보기")
        st.dataframe(df.head())
        st.write(f"**전체 데이터 수:** {len(df)}")
        st.markdown("---")

        # 2. 데이터 전처리
        df_numeric = preprocess_data(df)

        if df_numeric is not None and not df_numeric.empty:
            st.markdown("---")
            # 3. 분석 및 시각화
            analyze_and_visualize(df_numeric)

if __name__ == "__main__":
    main()
