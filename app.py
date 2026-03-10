import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(page_title="B2B Client Intelligence Lab", layout="wide")

st.title("B2B Client Intelligence Dashboard")
st.write("Client Risk Monitoring • Churn Prediction • Retention Insights")

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("B2B_Client_Churn_5000.csv")

# -----------------------------------
# CREATE RISK SCORE
# -----------------------------------

def risk_calc(row):

    risk = 0

    if row["Payment_Delay_Days"] > 20:
        risk += 2

    if row["Monthly_Usage_Score"] < 40:
        risk += 2

    if row["Support_Tickets_Last30Days"] > 4:
        risk += 3

    if row["Contract_Length_Months"] < 8:
        risk += 2

    return risk

df["Risk_Score"] = df.apply(risk_calc, axis=1)

def risk_label(x):

    if x >= 6:
        return "High"
    elif x >= 3:
        return "Medium"
    else:
        return "Low"

df["Risk_Level"] = df["Risk_Score"].apply(risk_label)

df["Churn"] = df["Renewal_Status"].map({"Yes":0,"No":1})

# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------

st.sidebar.header("Dashboard Filters")

region_filter = st.sidebar.multiselect(
"Region",
df["Region"].unique(),
default=df["Region"].unique()
)

industry_filter = st.sidebar.multiselect(
"Industry",
df["Industry"].unique(),
default=df["Industry"].unique()
)

risk_filter = st.sidebar.multiselect(
"Risk Level",
df["Risk_Level"].unique(),
default=df["Risk_Level"].unique()
)

rev_min = int(df["Monthly_Revenue_USD"].min())
rev_max = int(df["Monthly_Revenue_USD"].max())

revenue_filter = st.sidebar.slider(
"Revenue Range",
rev_min,
rev_max,
(rev_min,rev_max)
)

contract_filter = st.sidebar.slider(
"Contract Length (Months)",
int(df["Contract_Length_Months"].min()),
int(df["Contract_Length_Months"].max()),
(
int(df["Contract_Length_Months"].min()),
int(df["Contract_Length_Months"].max())
)
)

# -----------------------------------
# APPLY FILTERS
# -----------------------------------

filtered_df = df[
(df["Region"].isin(region_filter)) &
(df["Industry"].isin(industry_filter)) &
(df["Risk_Level"].isin(risk_filter)) &
(df["Monthly_Revenue_USD"] >= revenue_filter[0]) &
(df["Monthly_Revenue_USD"] <= revenue_filter[1]) &
(df["Contract_Length_Months"] >= contract_filter[0]) &
(df["Contract_Length_Months"] <= contract_filter[1])
]

# -----------------------------------
# KPI METRICS
# -----------------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Clients",len(filtered_df))

col2.metric(
"High Risk Clients",
len(filtered_df[filtered_df["Risk_Level"]=="High"])
)

col3.metric(
"Average Revenue",
round(filtered_df["Monthly_Revenue_USD"].mean(),2)
)

col4.metric(
"Churn Rate %",
round(filtered_df["Churn"].mean()*100,2)
)

st.divider()

# -----------------------------------
# TABS
# -----------------------------------

tab1,tab2,tab3,tab4 = st.tabs(
["Risk Overview","Business Segments","Churn Prediction","Client Explorer"]
)

# -----------------------------------
# TAB 1 : RISK OVERVIEW
# -----------------------------------

with tab1:

    st.subheader("Risk Category Distribution")

    risk_counts = filtered_df["Risk_Level"].value_counts()

    fig1 = plt.figure()

    plt.bar(risk_counts.index, risk_counts.values)

    plt.xlabel("Risk Category")
    plt.ylabel("Number of Clients")
    plt.title("Client Risk Distribution")

    st.pyplot(fig1)

    st.subheader("Revenue Distribution by Risk Level")

    fig2 = plt.figure()

    groups = [
    filtered_df[filtered_df["Risk_Level"]=="Low"]["Monthly_Revenue_USD"],
    filtered_df[filtered_df["Risk_Level"]=="Medium"]["Monthly_Revenue_USD"],
    filtered_df[filtered_df["Risk_Level"]=="High"]["Monthly_Revenue_USD"]
    ]

    plt.violinplot(groups)

    plt.xticks([1,2,3],["Low","Medium","High"])
    plt.ylabel("Revenue")

    st.pyplot(fig2)

# -----------------------------------
# TAB 2 : BUSINESS SEGMENTS
# -----------------------------------

with tab2:

    st.subheader("Industry vs Region Risk Bubble Chart")

    pivot = pd.pivot_table(
    filtered_df,
    index="Industry",
    columns="Region",
    values="Risk_Score",
    aggfunc="mean"
    )

    fig3 = plt.figure()

    for i,industry in enumerate(pivot.index):
        for j,region in enumerate(pivot.columns):

            size = pivot.loc[industry,region] * 120

            plt.scatter(j,i,s=size)

    plt.xticks(range(len(pivot.columns)),pivot.columns,rotation=45)
    plt.yticks(range(len(pivot.index)),pivot.index)

    plt.title("Industry vs Region Risk Levels")

    st.pyplot(fig3)

    st.subheader("Contract Length vs Churn Rate")

    bins = pd.cut(filtered_df["Contract_Length_Months"],bins=6)

    churn_rate = filtered_df.groupby(bins)["Churn"].mean()*100

    fig4 = plt.figure()

    plt.plot(range(len(churn_rate)),churn_rate.values,marker="o")

    plt.xticks(range(len(churn_rate)),[str(x) for x in churn_rate.index],rotation=45)

    plt.ylabel("Churn Rate (%)")

    st.pyplot(fig4)

# -----------------------------------
# TAB 3 : MACHINE LEARNING
# -----------------------------------

with tab3:

    st.subheader("Decision Tree Churn Prediction")

    features = [
    "Monthly_Usage_Score",
    "Payment_Delay_Days",
    "Contract_Length_Months",
    "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD",
    "Risk_Score"
    ]

    X = df[features]

    y = df["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2
    )

    model = DecisionTreeClassifier(max_depth=6)

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    st.metric("Model Accuracy",round(acc*100,2))

    cm = confusion_matrix(y_test,pred)

    st.subheader("Confusion Matrix")

    st.write(cm)

    importance = pd.Series(
    model.feature_importances_,
    index=features
    ).sort_values(ascending=False)

    st.subheader("Feature Importance")

    st.bar_chart(importance)

# -----------------------------------
# TAB 4 : CLIENT EXPLORER
# -----------------------------------

with tab4:

    st.subheader("Top 20 High Risk Clients")

    high_risk = filtered_df[filtered_df["Risk_Level"]=="High"]

    st.dataframe(
    high_risk.sort_values("Risk_Score",ascending=False).head(20)
    )

    st.subheader("Client Lookup")

    client = st.selectbox(
    "Select Client",
    filtered_df["Client_ID"]
    )

    row = filtered_df[filtered_df["Client_ID"]==client]

    st.write(row)

    st.subheader("Retention Suggestions")

    if row["Risk_Level"].values[0]=="High":

        st.error("High Risk Client")

        st.write("Offer payment flexibility")
        st.write("Assign dedicated account manager")
        st.write("Provide renewal discounts")

    elif row["Risk_Level"].values[0]=="Medium":

        st.warning("Moderate Risk")

        st.write("Increase product engagement")
        st.write("Provide training support")

    else:

        st.success("Low Risk Client")

        st.write("Maintain relationship")
        st.write("Offer loyalty benefits")
