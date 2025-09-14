import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from scipy.stats import chi2_contingency
st.set_page_config(layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")
df = load_data()

def apply_filter(df, col, selected_value):
    if selected_value != 'All':
        df = df[df[col]==selected_value]
    return df

def number_format(number):
    if number > 1000000:
        number = f"{round(number/1000000)}M"
    if number > 1000:
        number = f"{round(number/1000)}K"
    return number

def line_chart(x,y,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, text=y, mode='lines+markers+text', textposition='top right',hoverinfo='skip'))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',yaxis=dict(showticklabels=False,showgrid=False))
    st.plotly_chart(fig,use_container_width=True)

def stack_bar_chart(data,chart_title=''):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Bar(x=data.index,y=data[col],text=data[col],hoverinfo='skip',name=col))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',yaxis=dict(showticklabels=False,showgrid=False))
    st.plotly_chart(fig,use_container_width=True)

def bar_chart(x,y,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,y=y,text=y,hoverinfo='skip'))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',yaxis=dict(showticklabels=False,showgrid=False))
    st.plotly_chart(fig,use_container_width=True)

def horizontal_bar_chart(x,y,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,y=y,text=x,hoverinfo='skip',name=col,orientation='h'))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',height=600,yaxis=dict(showgrid=False,showticklabels=True))
    st.plotly_chart(fig,use_container_width=True)

def pie_chart(x,y,total_customers,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=x,values=y,textinfo='label+percent',hole=0.5))
    fig.update_layout(title_text=chart_title,annotations=[dict(text=total_customers,x=0.5,y=0.5,showarrow=False)])
    st.plotly_chart(fig,use_container_width=True)

header_col,nav_bar = st.columns([3,10])
with header_col: 
    st.write('### Bank Churn Analysis')
with nav_bar:
    pages = ['Customer Overview','Churn Analysis','Churn Prediction', 'Strong Engagement Customer']
    nav_col = st.columns(len(pages))

    for col,pagename in zip(nav_col,pages):
        with col:
            if st.button(pagename):
                st.query_params.update(page=pagename)

query_params = st.query_params
page = query_params.get('page','Customer Overview')

if page == 'Customer Overview':
    filter_pane, visual_pane, recommendation_pane = st.columns([2,7,2])
    with filter_pane:
        st.write("#### Filter")
        select_tenure = st.selectbox('Select Tenure',['All'] + df['tenure'].unique().tolist())
        select_geography = st.selectbox('Select Geography',['All'] + df['geography'].unique().tolist())
        select_gender = st.selectbox('Select Gender',['All'] + df['gender'].unique().tolist())
        active_status_display = {'Active':1,'Inactive':0}
        select_active_status = st.selectbox('Select Active Status',['All'] + list(active_status_display.keys()))
        active_churn_display = {'Churner':1,'Non Churner':0}
        select_churn_status = st.selectbox('Select Churn Status',['All'] + list(active_churn_display.keys()))
        select_product_holdings = st.selectbox('Select Product Holdings',['All'] + df['numofproducts'].unique().tolist())
        select_age_group = st.selectbox('Select Age Group',['All'] + df['age_group'].unique().tolist())
        select_creditscore_group = st.selectbox('Select Credit Score Group',['All'] + df['creditscore_group'].unique().tolist())
        select_saving_group = st.selectbox('Select Saving Group',['All'] + df['saving_group'].unique().tolist())
        select_income_group = st.selectbox('Select Income Group',['All'] + df['estimatedsalary_group'].dropna().unique().tolist())
        filter_column = [
            ('tenure',select_tenure),
            ('geography',select_geography),
            ('gender',select_gender),
            ('isactivemember',select_active_status),
            ('exited',select_churn_status),
            ('numofproducts',select_product_holdings),
            ('age_group',select_age_group),
            ('creditscore_group',select_creditscore_group),
            ('saving_group',select_saving_group),
            ('estimatedsalary_group',select_income_group)]
        filtered_df = df.copy()
        for col,selected_value in filter_column:
            if col == 'isactivemember':
                selected_value = active_status_display.get(selected_value,selected_value)
            if col == 'exited':
                selected_value = active_churn_display.get(selected_value,selected_value)
            filtered_df = apply_filter(filtered_df,col,selected_value)
    with visual_pane:
        kpi1,kpi2,kpi3,kpi4,kpi5 = st.columns(5)
        kpi1.metric("Total Customers",len(filtered_df))
        kpi2.metric("Active Rate",f"{round(filtered_df['isactivemember'].mean()*100)}%")
        kpi3.metric("Average Credit Score",f"{round(filtered_df['creditscore'].mean(),2)}")
        kpi4.metric("Average Saving ($)",f"{number_format(filtered_df[filtered_df['balance']>0]['balance'].mean())}")
        kpi5.metric("Average Income ($)",f"{number_format(filtered_df['estimatedsalary'].median())}")
        
        tenure_customer_data = filtered_df['tenure'].value_counts().reset_index()
        tenure_customer_data.columns = ['Tenure','Customer Count']
        tenure_customer_data = tenure_customer_data.sort_values(by='Tenure')
        tenure_customer_data['Tenure'] = tenure_customer_data['Tenure'].apply(lambda x: f"{x} year" if x <=1 else f"{x} years")

        credit_active_data = pd.crosstab(filtered_df['hascrcard'],filtered_df['isactivemember'],rownames=['Credit Card Holder'],colnames=['Active Status']).reset_index()
        credit_active_data = credit_active_data.rename(index={0:'no credit card holder',1:'credit card holder'})
        credit_active_data = credit_active_data.rename(columns={0:'inactive',1:'active'})
        credit_active_data = credit_active_data.drop(columns='Credit Card Holder')
        
        age_group_data = filtered_df['age_group'].value_counts().reset_index()
        age_group_data.columns = ['Age Group','Customer Count']
        age_group_data = age_group_data.sort_values(by='Age Group')

        geo_saving_data = pd.crosstab(filtered_df['geography'],filtered_df['saving'],rownames=['Geography'],colnames=['Saving Status'])
        geo_saving_data = geo_saving_data.rename(columns={1:'saving',0:'no saving'})
        
        products_gender_data = pd.crosstab(filtered_df['numofproducts'],filtered_df['gender'],rownames=['Product Holder'],colnames=['Gender']).reset_index()
        products_gender_data.index = products_gender_data['Product Holder'] = products_gender_data['Product Holder'].apply(lambda x: f"{x} Product Holder" if x==1 else f"{x} Product Holders")
        products_gender_data = products_gender_data.drop(columns="Product Holder")
        
        creditscore_group_data = filtered_df['creditscore_group'].value_counts().reset_index()
        creditscore_group_data.columns = ['Credit Score Group','Customer Count']
        creditscore_group_data['Sorting'] = creditscore_group_data['Credit Score Group'].map({'below 650':1,'650-700':2,'above 700':3})
        creditscore_group_data = creditscore_group_data.sort_values(by='Sorting')
        creditscore_group_data = creditscore_group_data.drop(columns='Sorting')

        saving_group_data = filtered_df['saving_group'].value_counts().reset_index()
        saving_group_data.columns = ['Saving Group','Customer Count']
        saving_group_data['Sorting'] = saving_group_data['Saving Group'].map({'above 200k':6,'below 50k':2,'150k-200k':5,'50k-100k':3,'no saving':1,'100k-150k':4})
        saving_group_data = saving_group_data.sort_values(by='Sorting')
        saving_group_data = saving_group_data.drop(columns='Sorting')

        income_group_data = filtered_df['estimatedsalary_group'].value_counts().reset_index()
        income_group_data.columns = ['Income Group','Customer Count']
        income_group_data['Sorting'] = income_group_data['Income Group'].map({'below 1k':1,'1k-20k':2,'20k-50k':3,'150k-200k':6,'50k-100k':4,'100k-150k':5})
        income_group_data = income_group_data.sort_values(by='Sorting')
        income_group_data = income_group_data.drop(columns='Sorting')
        
        row1_left_chart_title, row1_right_chart_title, detail_button=st.columns([2,1,1])
        row1_left_chart, row1_right_chart = st.columns([1,1])
        with detail_button:
            if st.button('View Potential Active Customer'):
                st.query_params.update(page='Potential Active Customer')
        with row1_left_chart:
            line_chart(tenure_customer_data['Tenure'],tenure_customer_data['Customer Count'],chart_title='Customer by Tenure')
        with row1_right_chart:
            stack_bar_chart(credit_active_data,chart_title='Credit Card Holder by Active Status')
        row2_left_chart_title, row2_right_chart_title, detail_button=st.columns([2,1,1])
        with detail_button:
            if st.button('View Potential Saving Customer'):
                st.query_params.update(page='Potential Saving Customer')
        row2_left_chart, row2_right_chart = st.columns([1,1])
        with row2_left_chart:
            bar_chart(age_group_data['Age Group'],age_group_data['Customer Count'],chart_title='Customer by Age Group')
        with row2_right_chart:
            stack_bar_chart(geo_saving_data,chart_title='Geography by Saving Status')
        col_space, creditscore_group_btn, saving_group_btn, income_group_btn=st.columns([3,1.5,1,1.5])
        if 'selected_group' not in st.session_state:
            st.session_state.selected_group = 'credit'
        with col_space:
            pass
        with creditscore_group_btn:
            if st.button('Credit Score Group',key='creditbtn'):
                st.session_state.selected_group = 'credit'
        with saving_group_btn:
            if st.button('Saving Group',key='savingbtn'):
                st.session_state.selected_group = 'saving'
        with income_group_btn:
            if st.button('Income Group',key='incomebtn'):
                st.session_state.selected_group = 'income'
        row3_left_chart, row3_right_chart = st.columns([1,1])
        with row3_left_chart:
            stack_bar_chart(products_gender_data,chart_title='Gender by Product Holder')
        with row3_right_chart:
            option = st.session_state.selected_group
            if option == 'credit':
                bar_chart(creditscore_group_data['Credit Score Group'],creditscore_group_data['Customer Count'],chart_title='Customer by Credit Score Group')
            if option == 'saving':
                bar_chart(saving_group_data['Saving Group'],saving_group_data['Customer Count'],chart_title='Customer by Saving Group')
            if option == 'income':
                bar_chart(income_group_data['Income Group'],income_group_data['Customer Count'],chart_title='Customer by Income Group')   
        with recommendation_pane:
            st.markdown("""
                ##### Findings
**Active Rate**
- Current active rate: **52%**
- If the bank is **modern/digital**, this rate is **low**
- If the bank is **traditional**, this rate is **acceptable**
- Active rate is **directly correlated** with **credit card ownership**
                
**Customer Profile**
- **Savings & income** around **€100,000**
- **50%** are aged **33–44**
                
**Geographic Insights**
- **Germany**: Customers hold **only savings products**
- **Spain & France**: Potential to acquire **non-savings customers**
                
**Product Holding Patterns**
- **Females**: More likely to hold **>2 products**
- **Males**: Typically hold **1–2 products**

##### Recommendations
**Increase Engagement**
- Provide or expand **credit card plans**
- Especially important if the bank is **modern/digital**
                
**Targeted Marketing**
- Focus on age group **33–44**
- Customize offers based on lifestyle and financial habits
                
**Regional Product Strategy**
- In **Germany**: Introduce **non-savings products**
- In **Spain & France**: Target **non-savings customers** with tailored campaigns
                
**Gender-Based Product Bundling**
- For **females**: Offer **multi-product bundles** and loyalty rewards
- For **males**: Promote upgrades from **1 to 2 products**
""")    
if page == 'Churn Analysis':
    metric1, metric2, geography_filter, age_filter, tenure_filter, numofproducts_filter, gender_filter = st.columns(7)
    with geography_filter:
        select_geography = st.selectbox('Select Geography',['All'] + df['geography'].unique().tolist())
    with age_filter:  
        select_age_group = st.selectbox('Select Age Group',['All'] + df['age_group'].unique().tolist())
    with tenure_filter:
        select_tenure = st.selectbox('Select Tenure',['All'] + df['tenure'].unique().tolist())
    with numofproducts_filter:
        select_product_holdings = st.selectbox('Select Product Holdings',['All'] + df['numofproducts'].unique().tolist())
    with gender_filter:
        select_gender = st.selectbox('Select Gender',['All'] + df['gender'].unique().tolist())        
    churn_df = df.copy()
    filter_column = [
            ('tenure',select_tenure),
            ('geography',select_geography),
            ('gender',select_gender),
            ('numofproducts',select_product_holdings),
            ('age_group',select_age_group)]
    for col,selected_value in filter_column:
        churn_df = apply_filter(churn_df,col,selected_value)
    strong_engagement_customers = churn_df[(churn_df['creditscore']>700) & (churn_df['balance']>100000) & (churn_df['estimatedsalary']>100000) & (churn_df['isactivemember']==1)]
    strong_engagement_churn_customers = churn_df[(churn_df['creditscore']>700) & (churn_df['balance']>100000) & (churn_df['estimatedsalary']>100000) & (churn_df['isactivemember']==1)&(churn_df['exited']==1)]
    churn_customers = churn_df[churn_df['exited']==1]
    strong_engagement_churn_rate_in_churn_customers = f'{round(len(strong_engagement_churn_customers)/len(churn_customers)*100,2)}%'
    strong_engagement_churn_rate_in_engagement_customers = f"{round(len(strong_engagement_churn_customers)/len(strong_engagement_customers)*100,2)}%"
    metric1,metric2,very_strong_influence_btn,strong_influence_btn,moderate_influence_btn,low_influence_btn,no_influence_btn = st.columns(7)
    metric1.metric('High Engagement (Churners) %',strong_engagement_churn_rate_in_churn_customers)
    metric2.metric('Strong Engaged Churn %',strong_engagement_churn_rate_in_engagement_customers)
    churn_categorical_columns = ['gender','geography','isactivemember','age_group', 'tenure', 'numofproducts', 'saving_group', 'creditscore_group','estimatedsalary_group']
    contingency_table = {}
    for col in churn_categorical_columns:
        contingency_table[col] = pd.crosstab(churn_df[col],churn_df['exited'])
    chi2_results = []
    for col in churn_categorical_columns:
        chi2_stats, chi2_p_val, dof, expected = (chi2_contingency(contingency_table[col]))
        if chi2_stats > 1000:
            influence_type = 'very strong'
        elif chi2_stats > 500:
            influence_type = 'strong'
        elif chi2_stats > 100:
            influence_type = 'moderate'
        elif chi2_p_val < 0.05:
            influence_type = 'low'
        else :
            influence_type = 'no'
        chi2_results.append({'variable':col,'p_val':chi2_p_val,'stats':chi2_stats,'influence_type':influence_type})
    chi2_results = pd.DataFrame(chi2_results)
    chi2_results['p_val'] = chi2_results['p_val'].apply(lambda x:f"{x:.2e}")  
    if 'influence_option' not in st.session_state:
        st.session_state.influence_option = 'very strong'
    with very_strong_influence_btn:
        if st.button('Very Strong Influence',key='very_strong_btn'):
            st.session_state.influence_option = 'very strong'
    with strong_influence_btn:
        if st.button('Strong Influence',key='strong_btn'):
            st.session_state.influence_option = 'strong'
    with moderate_influence_btn:
        if st.button('Moderate Influence',key='moderate_btn'):
            st.session_state.influence_option = 'moderate'
    with low_influence_btn:
        if st.button('Low Influence',key='low_btn'):
            st.session_state.influence_option = 'low'
    with no_influence_btn:
        if st.button('No Influence',key='no_btn'):
            st.session_state.influence_option = 'no'
    left_section, right_section = st.columns([2,5])
    influence_option = st.session_state.influence_option
    chi2_filter = chi2_results[chi2_results['influence_type'] == influence_option]
    with left_section:
        churn_status_df = churn_df['exited'].value_counts().reset_index()
        churn_status_df.columns = ['Churn Status','Customer Count']
        churn_status_df = churn_status_df.rename(index={0:'Non Churner',1:'Churner'})
        pie_chart(churn_status_df.index,churn_status_df['Customer Count'],f"Total Customers<br>{len(df)}",chart_title='Churner Vs Non Churner')
        engagement_geography_age_data = pd.crosstab(strong_engagement_churn_customers['age_group'],strong_engagement_churn_customers['geography'],rownames=['Age Group'],colnames=['Geography'])
        stack_bar_chart(engagement_geography_age_data,chart_title='Strong Engagement Customers by Age Group and Geography')
        st.markdown("""
#### Key Findings

- **Strong churn drivers**: Product count, age group (Chi² > 1000)
- **Moderate influence**: Geography, active saving, gender (Chi² 100–300)
- **Low relevance**: Credit score, income, tenure type
- **High-value churn**: 3.14%  
- **Engaged churn**: 15.88%  
- **Overall churn**: 20.4% (industry standard)  
- **Germany churn**: Above average — needs competitor/service review

#### Recommended Actions

- Target product & age segments for retention
- Deep dive into Germany churn causes
- Share insights with Ops & Customer Service teams
- Track high-value and engaged churn for recovery
""")
    with right_section:
        st.subheader('Churn Status Distribution by Hypothesis Test Influence Level')
        if not chi2_filter.empty:
            col_count = st.columns(len(chi2_filter))
            for col, variable in zip(col_count, chi2_filter['variable']):
                with col:
                    influence_chart_data = pd.crosstab(df[variable], df['exited'],rownames=[variable], colnames=['Churn Status']).reset_index()
                if variable in influence_chart_data.columns:
                    influence_chart_data = influence_chart_data.set_index(variable)
                    if variable == 'isactivemember':
                        influence_chart_data = influence_chart_data.rename(index={0:'Inactive',1:'Active'})
                    influence_chart_data = influence_chart_data.rename(columns={0:'No Churner', 1:'Churner'})
                    stack_bar_chart(influence_chart_data, chart_title=f'Churn Status by {variable.capitalize()}')
                else:
                    st.warning(f"Column '{variable}' not found in chart data.")
        st.dataframe(chi2_filter.drop(columns=['influence_type']),use_container_width=True)
        high_value_lost_customer = strong_engagement_churn_customers[['customerid','surname','geography', 'gender', 'age', 'numofproducts', 'hascrcard', 'tenure', 'balance', 'creditscore', 'estimatedsalary']].reset_index(drop=True)
        st.subheader(f'High Value Lost Customer - {len(high_value_lost_customer)}')
        st.write(high_value_lost_customer)
if page == 'Churn Prediction':
    filter_pane, col_space, visual_pane, insight_pane = st.columns([2,1,4,2])
    with filter_pane:
        st.write("#### Filter")
        select_geography = st.selectbox('Select Geography',['All'] + df['geography'].unique().tolist())
        select_gender = st.selectbox('Select Gender',['All'] + df['gender'].unique().tolist())
        active_status_display = {'Active':1,'Inactive':0}
        select_active_status = st.selectbox('Select Active Status',['All'] + list(active_status_display.keys()))
        active_churn_display = {'Churner':1,'Non Churner':0}
        select_churn_status = st.selectbox('Select Churn Status',['All'] + list(active_churn_display.keys()))
        select_product_holdings = st.selectbox('Select Product Holdings',['All'] + df['numofproducts'].unique().tolist())
        select_age_group = st.selectbox('Select Age Group',['All'] + df['age_group'].unique().tolist())
        select_creditscore_group = st.selectbox('Select Credit Score Group',['All'] + df['creditscore_group'].unique().tolist())
        select_saving_group = st.selectbox('Select Saving Group',['All'] + df['saving_group'].unique().tolist())
        select_income_group = st.selectbox('Select Income Group',['All'] + df['estimatedsalary_group'].dropna().unique().tolist())
        filter_column = [
            ('geography',select_geography),
            ('gender',select_gender),
            ('isactivemember',select_active_status),
            ('exited',select_churn_status),
            ('numofproducts',select_product_holdings),
            ('age_group',select_age_group),
            ('creditscore_group',select_creditscore_group),
            ('saving_group',select_saving_group),
            ('estimatedsalary_group',select_income_group)]
        filtered_df = df.copy()
        for col,selected_value in filter_column:
            if col == 'isactivemember':
                selected_value = active_status_display.get(selected_value,selected_value)
            if col == 'exited':
                selected_value = active_churn_display.get(selected_value,selected_value)
            filtered_df = apply_filter(filtered_df,col,selected_value)
    with visual_pane:
        st.markdown("<div style='margin-top:50px'></div>",unsafe_allow_html=True)
        kpi1,kpi2 = st.columns(2)
        kpi1.metric('Avg Churn Probabilities',f"{filtered_df['exit_proba'].mean():.2%}")
        kpi2.metric('Avg Non_Churn Probabilities',f"{filtered_df['stay_proba'].mean():.2%}")
        feature_importance_data = pd.read_csv("feature_coefficient.csv")
        feature_importance_data['coefficient'] = feature_importance_data['coefficient'].apply(lambda x: round(x,2))
        feature_importance_data = feature_importance_data.sort_values(by='coefficient',ascending=True)
        horizontal_bar_chart(feature_importance_data['coefficient'],feature_importance_data['feature'],chart_title='Features Influence of Churn Status')
        potential_high_risk_customers = filtered_df[(filtered_df['exit_proba']>=0.7)&(filtered_df['exited']==0)]
        st.subheader(f'Potential High Risk Customers - {len(potential_high_risk_customers)}')
        potential_high_risk_customers['active_status'] = filtered_df['isactivemember'].map({1:'Active',0:'Inactive'})
        potential_high_risk_customers = potential_high_risk_customers[['customerid', 'surname', 'active_status','geography', 'gender', 'age', 'numofproducts', 'tenure', 'balance','creditscore', 'estimatedsalary']]
        st.write(potential_high_risk_customers)
    with insight_pane:
       st.markdown("""
#### Model Selection

- **Chosen Model**: Logistic Regression  
- **Reason**: Easier feature interpretation for actionable insights  
- **Performance**:  
  - Accuracy: **0.81**  
  - Precision: **0.53**  
  - Recall: **0.69**  
  - F1 Score: **0.60**

#### Churn Risk Insights

- **Missed churners**: ~38% — high-risk group needs attention  
- **Exit probability >70%**: Flag for **Marketing**, **Sales**, **Customer Service**, and **Operations**

#### Key Patterns

- **Product holding**:  
  - **Female with multiple products** → higher churn  
  - **2-product holders** → more likely to stay

- **Geography**:  
  - **Germany & Spain** → elevated churn risk

- **Age group**:  
  - **45–60** and **33–44** → higher churn than younger customers  
  - Possible **competitor influence**

- **Savings behavior**:  
  - **Extreme savers** (>200k or 50k) → more likely to churn  
  - **Mid-range savers** (50k–150k) → more likely to stay
""")
if page == 'Strong Engagement Customer':
    strong_engagement_customers = df[(df['isactivemember']==1) & (df['balance']>100000) & (df['creditscore']>=700) & (df['estimatedsalary']>=100000) & (df['exited']==0)]
    left_session,right_session = st.columns([2,1])
    with left_session:
        row1_filter, col_space, row1_kpi_card, row1_chart = st.columns([2,1,2,4])
        with row1_filter:
            st.markdown("<div style='margin-top:50px'></div>",unsafe_allow_html=True)
            select_geography = st.selectbox('Select Geography',['All'] + df['geography'].unique().tolist())
            select_gender = st.selectbox('Select Gender',['All'] + df['gender'].unique().tolist())
            select_product_holdings = st.selectbox('Select Product Holdings',['All'] + df['numofproducts'].unique().tolist())
            select_age_group = st.selectbox('Select Age Group',['All'] + df['age_group'].unique().tolist())
            filter_column = [
            ('geography',select_geography),
            ('gender',select_gender),
            ('numofproducts',select_product_holdings),
            ('age_group',select_age_group),]      
            filtered_engagement_df = strong_engagement_customers.copy()
        for col,selected_value in filter_column:
            filtered_engagement_df = apply_filter(filtered_engagement_df,col,selected_value)
        with row1_kpi_card:
            st.markdown("<div style='margin-top:50px'></div>",unsafe_allow_html=True)
            total_strong_engagement_customers = len(filtered_engagement_df)
            engagement_rate = f"{round(total_strong_engagement_customers/len(df)*100,2)}%"
            row1_kpi_card.metric('Total Strong Engagement Customers',total_strong_engagement_customers)
            row1_kpi_card.metric('Engagement Rate',engagement_rate)
        with row1_chart:
            st.markdown("<div style='margin-top:20px'></div>",unsafe_allow_html=True)
            potential_agegroup_gender = pd.crosstab(filtered_engagement_df['age_group'],filtered_engagement_df['gender'],rownames=['Age Group'],colnames=['Gender']).reset_index()
            potential_agegroup_gender.index = potential_agegroup_gender['Age Group']
            potential_agegroup_gender = potential_agegroup_gender.drop(columns='Age Group')
            stack_bar_chart(potential_agegroup_gender,chart_title='Strong Engagement Customer by Savings and Gender')
        potential_table = filtered_engagement_df[['customerid', 'surname', 'geography', 'gender', 'age', 'numofproducts', 'tenure', 'balance','creditscore', 'estimatedsalary']]
        st.write('Strong Engagement Customers List')
        st.dataframe(potential_table, width=1000, height=500)
    with right_session:
        st.markdown("""
#### Findings

- **3.4%** of customers show **strong engagement**:
  - **Active**, **not churned**
  - **Credit score >700**, **salary & balance >€100k**
  - **Age 24–60**
- **Male engagement** is higher than female in this segment

#### Recommendations

- Prioritize this segment for **product launches**, **promotions**, and **campaigns**
- Share detailed profiles with **Product, Marketing, and Sales teams**
- Use the provided table to support **targeting and awareness**

#### Actions

- **Flag segment** in dashboard with filters: engagement level, credit score, salary, age, gender  
- **Notify cross-functional teams** with summary and table for campaign planning  
- **Design tailored outreach** (e.g. email, app banners) focused on high-value male customers  
- **Schedule follow-up analysis** post-campaign to measure conversion and retention impact
""")
if page == 'Potential Active Customer':
    potential_active_customers = df[(df['hascrcard']==0) & (df['exited']==0) & (df['creditscore']>=700) & (df['estimatedsalary']>=100000) & (df['balance']>=100000)]
    left_session,right_session = st.columns([2,1])
    with left_session:
        row1_filter, col_space, row1_kpi_card, row1_chart = st.columns([2,1,2,4])
        with row1_filter:
            st.markdown("<div style='margin-top:50px'></div>",unsafe_allow_html=True)
            select_geography = st.selectbox('Select Geography',['All'] + df['geography'].unique().tolist())
            select_gender = st.selectbox('Select Gender',['All'] + df['gender'].unique().tolist())
            select_product_holdings = st.selectbox('Select Product Holdings',['All'] + df['numofproducts'].unique().tolist())
            select_age_group = st.selectbox('Select Age Group',['All'] + df['age_group'].unique().tolist())
            filter_column = [
            ('geography',select_geography),
            ('gender',select_gender),
            ('numofproducts',select_product_holdings),
            ('age_group',select_age_group), ]      
            filtered_potential_df = potential_active_customers.copy()
        for col,selected_value in filter_column:
            filtered_potential_df = apply_filter(filtered_potential_df,col,selected_value)
        with row1_kpi_card:
            st.markdown("<div style='margin-top:50px'></div>",unsafe_allow_html=True)
            total_potential_active_customers = len(filtered_potential_df)
            potential_active_rate = f"{round(total_potential_active_customers/len(df)*100,2)}%"
            row1_kpi_card.metric('Total Potential Active Customers',total_potential_active_customers)
            row1_kpi_card.metric('Potential Active Rate',potential_active_rate)
        with row1_chart:
            st.markdown("<div style='margin-top:20px'></div>",unsafe_allow_html=True)
            potential_geography_gender = pd.crosstab(filtered_potential_df['geography'],filtered_potential_df['gender'],rownames=['Geography'],colnames=['Gender']).reset_index()
            potential_geography_gender.index = potential_geography_gender['Geography']
            potential_geography_gender = potential_geography_gender.drop(columns='Geography')
            stack_bar_chart(potential_geography_gender,chart_title='Potential Customer by Geography and Gender')
        potential_table = filtered_potential_df[['customerid', 'surname', 'geography', 'gender', 'age', 'numofproducts', 'tenure', 'balance','creditscore', 'estimatedsalary']]
        potential_table['balance'] = potential_table['balance'].apply(lambda x: f"$ {(x)}")
        potential_table['estimatedsalary'] = potential_table['estimatedsalary'].apply(lambda x: f"$ {(x)}")
        st.write('Potential Customer List')
        st.dataframe(potential_table, width=1000, height=500)
    with right_session:
        st.markdown("""
#### Findings

- **High-Potential Inactive Segment**
  - **2.43%** of customers are:
    - **Inactive but retained** and Have **credit scores >700**
    - Hold **€100k+ in salary and savings**

- **Gender & Regional Distribution**
  - **France & Spain**: More **male** potential customers
  - **Germany**: More **female** potential customers
  - Marketing efforts should be tailored by **region and gender**

- **Age Group Alignment**
  - Potential segment falls within the **33–44 age group**

- **Credit Card Engagement**
  - There is a **100% correlation** between **active status** and **credit card ownership**

#### Actions

- **Credit Card Campaign**
  - Design and launch a **targeted credit card plan** for the 2.43% high-potential segment
  - Highlight benefits that match their financial profile and lifestyle

- **Segmented Marketing Strategy**
  - In **France & Spain**: Focus on **male customers**
  - In **Germany**: Focus on **female customers**

- **Age-Based Targeting**
  - Prioritize the **33–44 age group** in all outreach efforts
  - Use personalized campaigns to increase relevance and conversion

- **Data Exploration**
  - **Marketing, Key Account, and Sales teams** can explore the **Potential Customers Table** for detailed insights
  - **Please handle sensitive customer data with care** and follow data privacy guidelines
""")
if page == 'Potential Saving Customer': 
    potential_saving_customers = df[(df['balance']>0) & (df['creditscore']>=700) & (df['estimatedsalary']>=100000) & (df['exited']==0)]
    left_session,right_session = st.columns([2,1])
    with left_session:
        row1_filter, col_space, row1_kpi_card, row1_chart = st.columns([2,1,2,4])
        with row1_filter:
            st.markdown("<div style='margin-top:50px'></div>",unsafe_allow_html=True)
            select_geography = st.selectbox('Select Geography',['All'] + df['geography'].unique().tolist())
            select_gender = st.selectbox('Select Gender',['All'] + df['gender'].unique().tolist())
            active_status_display = {'Active':1,'Inactive':0}
            select_active_status = st.selectbox('Select Active Status',['All'] + list(active_status_display.keys()))
            select_product_holdings = st.selectbox('Select Product Holdings',['All'] + df['numofproducts'].unique().tolist())
            select_age_group = st.selectbox('Select Age Group',['All'] + df['age_group'].unique().tolist())
            filter_column = [
            ('geography',select_geography),
            ('gender',select_gender),
            ('isactivemember',select_active_status),
            ('numofproducts',select_product_holdings),
            ('age_group',select_age_group)]      
            filtered_potential_df = potential_saving_customers.copy()
        for col,selected_value in filter_column:
            if col == 'isactivemember':
                selected_value = active_status_display.get(selected_value,selected_value)
            filtered_potential_df = apply_filter(filtered_potential_df,col,selected_value)
        with row1_kpi_card:
            st.markdown("<div style='margin-top:50px'></div>",unsafe_allow_html=True)
            total_potential_saving_customers = len(filtered_potential_df)
            potential_saving_rate = f"{round(total_potential_saving_customers/len(df)*100,2)}%"
            row1_kpi_card.metric('Total Potential Saving Customers',total_potential_saving_customers)
            row1_kpi_card.metric('Potential Saving Rate',potential_saving_rate)
        with row1_chart:
            st.markdown("<div style='margin-top:20px'></div>",unsafe_allow_html=True)
            potential_agegroup_gender = pd.crosstab(filtered_potential_df['age_group'],filtered_potential_df['gender'],rownames=['Age Group'],colnames=['Gender']).reset_index()
            potential_agegroup_gender.index = potential_agegroup_gender['Age Group']
            potential_agegroup_gender = potential_agegroup_gender.drop(columns='Age Group')
            stack_bar_chart(potential_agegroup_gender,chart_title='Potential Customer by Savings and Gender')
        filtered_potential_df['active_status'] = filtered_potential_df['isactivemember'].map({1:'Active',0:'Inactive'})
        potential_table = filtered_potential_df[['customerid', 'surname', 'active_status','geography', 'gender', 'age', 'numofproducts', 'tenure', 'balance','creditscore', 'estimatedsalary']]
        st.write('Potential Customer List')
        st.dataframe(potential_table, width=1000, height=500)
    with right_session:
        st.markdown("""
#### Findings

- Potential saving customers with **>700 credit score** and **€100k+ salary**
- Includes **inactive customers** — saving is **not directly tied** to active status
- Age range: **24–60**
- Targeting this group can **increase saving rate by ~4.8%**

#### Actions

- Provide **credit plans** to inactive high-potential savers
- Target age group **24–60** to boost engagement and savings
- Use tailored outreach to convert into active, multi-product holders
""")
