import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import  datetime as dt
import subprocess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from sklearn.tree import DecisionTreeClassifier
import pickle

m_df = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/Customer Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_icon=":bulb:",page_title="CUSTOMER SEGMENTATION ")
st.markdown(
    """
        <style>
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }
         .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    choose = option_menu("HOME", ["About", "Customer Wise Data Analysis", "Region Wise Data Analysis", "Product Wise Data Analysis","MALL CUSTOMERS ANALYSIS" ,
                                         "MALL CUSTOMERS DATA VISUALIZATION","MACHINE LEARNING(MALL DATASET)","Data_analysis(Market Data)","Data_Visualization(Market Data)","Clustering(Market Data)","Data Analysis (STORE)","PREDICTION (SALES)","Walmart_Sales_Visualization"
                                            ,"Walmart_Sales_Prediction","Contact","LOGOUT"],
                         icons=['house', 'activity ', 'geo-alt-fill', 'bag-fill','person-lines-fill','file-bar-graph-fill','arrow-clockwise','pie-chart-fill','bar-chart-steps','collection','shop','cpu','reception-2','gpu-card','person-rolodex','arrow-right-square-fill'],
                         menu_icon="house", default_index=0,
                         styles={
        # "container": {"padding": "5!important", "background-color": "#fafafa"},
        # "icon": {"color": "orange", "font-size": "25px"},
        # "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        # "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

df = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/train.csv")

def about():
    st.title("Customer Segmentation ")
    st.write("""
       Customer Segmentation is a project aimed at analyzing customer data and segmenting customers into distinct groups based on various characteristics such as demographics, behavior, and preferences. The goal is to uncover meaningful insights that can drive targeted marketing strategies, improve customer engagement, and enhance overall business performance.

       ### Objectives:

       1. **Understand Customer Base**: Gain insights into the diverse customer base by analyzing demographic information such as age, gender, income, and location.

       2. **Identify Customer Segments**: Segment customers into distinct groups based on shared characteristics, including purchase behavior, frequency of purchases, product preferences, and engagement with marketing campaigns.

       3. **Personalized Marketing**: Develop personalized marketing strategies tailored to different customer segments to improve campaign effectiveness and drive higher conversion rates.

       4. **Enhance Customer Experience**: Identify opportunities to enhance the customer experience by delivering personalized recommendations, offers, and services based on individual preferences and behavior.

       5. **Optimize Resource Allocation**: Allocate resources more effectively by focusing marketing efforts and resources on high-value customer segments with the greatest potential for revenue growth and profitability.

       ### Methodology:

       1. **Data Collection**: Gather customer data from various sources, including transactional data, customer profiles, and marketing interactions.

       2. **Data Preprocessing**: Cleanse and preprocess the data to ensure accuracy and consistency. Handle missing values, outliers, and data inconsistencies.

       3. **Exploratory Data Analysis (EDA)**: Perform exploratory data analysis to uncover patterns, trends, and relationships within the data. Visualize key metrics and distributions to gain a deeper understanding of the customer base.

       4. **Customer Segmentation**: Utilize clustering algorithms such as K-means, hierarchical clustering, or Gaussian mixture models to segment customers into distinct groups based on predefined features.

       5. **Evaluation and Validation**: Evaluate the quality of customer segments using appropriate metrics and validation techniques.

       """)

def customer():
    st.title("Customer Wise Data Analysis")
    st.header("DATASET")
    st.write(df.head())

    df["sales_year"] =df["Order Date"].str.split("/").str[2]
    st.subheader("YEAR WISE SALES")
    year_sales = df.groupby(["sales_year","Sales"]).size().value_counts()
    # st.write(year_sales)

    total_sales_per_year = df.groupby("sales_year")["Sales"].sum()
    st.write(total_sales_per_year)

    st.subheader("YEAR WISE SALES GRAPH")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create bar plot using Matplotlib
    fig, ax = plt.subplots()
    ax.bar(total_sales_per_year.index, total_sales_per_year.values)

    # Customize the plot
    ax.set_title('Total Sales per Year', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Total Sales', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.2)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # st.subheader("Highest spender customer names/ID( table )")
    highest_spendor=df.groupby("Customer ID")["Order ID"].count()
    # st.write(highest_spendor)


    st.subheader("TOP 10 HIGHEST SPENDER")
    top_10_spenders = highest_spendor.nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    st.write(top_10_spenders)
    # Create bar plot using Seaborn
    sns.barplot(x=top_10_spenders.index, y=top_10_spenders.values, ax=ax)

    ax.set_title('Top 10 Spenders', fontsize=16)
    ax.set_xlabel('Customer ID', fontsize=14)
    ax.set_ylabel('Total Orders', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)

    st.subheader("10 LOWEST SPENDERS ")
    low_spendor_top10 =highest_spendor.nsmallest(20)
    st.write(low_spendor_top10)
    st.subheader("10 LOWEST SPENDERS GRAPH ")
    low_spendor_top10.plot(kind="bar")
    st.bar_chart(low_spendor_top10)

def region():
    st.title("Region Wise Data Analysis")
    st.subheader("STATE WISE SALES")
    state_wise_sales = df.groupby("State")["Sales"].count()
    st.write(state_wise_sales)
    top_40_sales = state_wise_sales.nlargest(10)
    st.subheader("Top 10 StateWise Sales")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Custom colors
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

    # Plotting pie chart using matplotlib
    plt.pie(top_40_sales, labels=top_40_sales.index, autopct='%1.2f%%', startangle=140, colors=colors, shadow=True)
    plt.title("Top 40 Sales")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

    top_postal_code = df.groupby("Postal Code")["Sales"].count()
    top_5_postal_code = top_postal_code.nlargest(5)
    st.subheader("Top 5 postal code by sales")
    st.write(top_5_postal_code)

    st.subheader("Top 5 Postal Codes")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set Seaborn style
    sns.set_style("whitegrid")

    # Create bar plot using Seaborn
    sns.barplot(x=top_5_postal_code.index, y=top_5_postal_code.values, ax=ax)

    # Set plot title and labels
    ax.set_title('Top 5 Postal Codes', fontsize=16)
    ax.set_xlabel('Postal Code', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)

    # Set grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Display plot
    st.pyplot(fig)

    st.subheader("Region wise sale")
    region_wise_sales = df["Region"].value_counts()
    st.write(region_wise_sales)

    st.subheader("Region-wise Sales")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set Seaborn style
    sns.set_style("whitegrid")

    # Custom colors
    colors = sns.color_palette("husl", len(region_wise_sales))

    # Plotting pie chart using matplotlib
    plt.pie(region_wise_sales, labels=region_wise_sales.index, autopct='%1.1f%%', colors=colors, shadow=True)
    plt.title("Region-wise Sales")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display plot
    st.pyplot(fig)

    st.subheader("CITY WISE SALES ")
    city_wise_order_count = df.groupby("City")["Order ID"].count()
    top_40_city_count = city_wise_order_count.nlargest(40)
    st.write(top_40_city_count)
    st.subheader("Top 40 City Countsplot")
    top_40_city_count = df["City"].value_counts().head(40)
    sns.set_style("whitegrid")
    st.bar_chart(top_40_city_count)

def product():
    st.title("Product Wise Data Analysis")
    st.subheader("Category Wise Sales")
    category_wise_order = df.groupby("Category")["Order ID"].count()
    st.write(category_wise_order)

    # Category-wise Orders
    st.subheader("Category-wise Orders")
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    plt.pie(category_wise_order, labels=category_wise_order.index, autopct='%1.1f%%', shadow=True, colors=colors)
    plt.title("Category-wise Orders")
    plt.axis('equal')
    st.pyplot(fig)
    st.subheader("Highest sale Product")

    highest_sell_product = df["Product ID"].value_counts()
    st.write(highest_sell_product)
    st.subheader("Top 10 Highest Selling Product")
    top_selling = highest_sell_product.nlargest(10)
    st.write(top_selling)

    st.subheader("Top Selling Products")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("whitegrid")
    sns.barplot(x=top_selling.index, y=top_selling.values)
    plt.xlabel("Product ID")
    plt.ylabel("Orders Quantity")
    plt.title("Top Selling Products")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def contact():
    st.title("Contact Info")
    st.write("Please fill out the form below to get in touch with us.")

    # Input fields
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    # Submit button
    if st.button("Submit"):
        if name and email and message:
            st.success("Thank you! Your message has been submitted.")
            # You can add code here to handle the submission, such as sending an email or saving to a database
        else:
            st.error("Please fill out all fields.")


def mall_data_analysis():
    st.title("MALL CUSTOMERS ANALYSIS")


    st.write("")
    st.subheader("MALL CUSTOMER DATASET")
    mall_data = pd.read_csv("Datasets/Mall_Customers.csv")
    st.dataframe(mall_data)
    st.subheader("SHAPE OF DATASET")
    st.write("ROWS")
    st.subheader(mall_data.shape[0])
    st.write("Columns")
    st.subheader(mall_data.shape[1])
    st.subheader("NULL VALUES BY COLUMNS")

    st.dataframe(mall_data.isnull().sum())
    st.subheader("Annual income & spending score")
    temp = mall_data.iloc[:,3:5]
    st.dataframe(temp)
    st.subheader("DATA INFO")
    st.write(mall_data.describe())
    st.subheader("GENDER VALUE COUNT")
    st.write(mall_data["Gender"].value_counts())
    st.subheader("Age and Annual Income.")
    st.dataframe(mall_data.iloc[:,2:4])
    print(df.columns)
    st.subheader("Top 10 most spending score customer by customer id.")
    top_10_customers = mall_data.nlargest(10, "Spending Score (1-100)")

    # Print only CustomerID and Spending Score
    st.write(top_10_customers[['CustomerID', 'Spending Score (1-100)']])

    st.subheader("Top 10 customers with the highest annual income")

    # Get the top 10 customers based on annual income
    top_10_income_customers = mall_data.nlargest(10, "Annual Income (k$)")

    # Print CustomerID and Annual Income for top 10 customers
    st.write(top_10_income_customers[['CustomerID', 'Annual Income (k$)']])

def MALL_CUSTOMERS_DATA_VISUALIZATION():
    # Plot Histogram of Age
    mall_data = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/Mall_Customers.csv")

    st.subheader("Histogram of Age")
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.hist(mall_data['Age'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(fig)
    # Plot Scatter graph of Annual Income and Spending Score
    st.subheader("Scatter graph of Annual Income and Spending Score")
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.scatter(mall_data['Annual Income (k$)'], mall_data['Spending Score (1-100)'], color='green')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    st.pyplot(fig)

    # Plot Bar graph of Gender and Spending Score
    st.subheader("Bar graph of Gender and Spending Score")
    gender_spending = mall_data.groupby('Gender')['Spending Score (1-100)'].mean()
    fig, ax = plt.subplots(figsize=(8, 8))

    gender_spending.plot(kind='bar', color=['blue', 'pink'])
    plt.xlabel('Gender')
    plt.ylabel('Average Spending Score')
    st.pyplot(fig)

    # Plot Bar graph of Top 10 highest income customers
    st.subheader("Bar graph of Top 10 highest income customers")
    top_10_income_customers = mall_data.nlargest(10, 'Annual Income (k$)')
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.bar(top_10_income_customers['CustomerID'], top_10_income_customers['Annual Income (k$)'], color='orange')
    plt.xlabel('CustomerID')
    plt.ylabel('Annual Income (k$)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Pie chart of Top 10 highest spending customers
    st.subheader("Pie chart of Top 10 highest spending customers")
    top_10_spending_customers = mall_data.nlargest(10, 'Spending Score (1-100)')
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.pie(top_10_spending_customers['Spending Score (1-100)'], labels=top_10_spending_customers['CustomerID'],
            autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    st.pyplot(fig)

    # Plot Bar graph of Age Groups and Spending Scores
    st.subheader("Bar graph of Age Groups and Spending Scores")
    age_bins = pd.cut(mall_data['Age'], bins=[0, 20, 30, 40, 50, 60, 70])
    age_groups_spending = mall_data.groupby(age_bins)['Spending Score (1-100)'].mean()
    fig, ax = plt.subplots(figsize=(8, 8))

    age_groups_spending.plot(kind='bar', color='purple')
    plt.xlabel('Age Groups')
    plt.ylabel('Average Spending Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def ml():
    st.title("MALL CUSTOMER MACHINE LEARNING")

    # Sample DataFrame
    mall_data = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/Mall_Customers.csv")

    # Perform KMeans clustering
    X = mall_data[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=5, random_state=0)
    mall_data['Cluster'] = kmeans.fit_predict(X)

    # Plotting the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(mall_data['Annual Income (k$)'], mall_data['Spending Score (1-100)'], c=mall_data['Cluster'],
                cmap='viridis', s=50)
    plt.title('Income Wise Spending Clusters')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.colorbar(label='Cluster')
    st.pyplot(plt)

    # Streamlit app
    st.title("Income Wise Spending Clusters")

    # Allow user to insert details
    st.subheader("Insert Your Details")
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    annual_income = st.number_input("Annual Income (k$)", min_value=5, max_value=150, value=50)
    spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

    # Predict cluster for user input
    user_data = np.array([[annual_income, spending_score]])
    user_cluster = kmeans.predict(user_data)[0]

    # Display user's cluster
    st.subheader("Your Cluster")
    st.write(f"You belong to Cluster {user_cluster}")

    # Display meanings of clusters
    st.subheader("Cluster Meanings")
    st.write("Cluster 0: Low Income, Low Spending")
    st.write("Cluster 1: High Income, High Spending")
    st.write("Cluster 2: High Income, Low Spending")
    st.write("Cluster 3: Low Income, High Spending")
    st.write("Cluster 4: Moderate Income, Moderate Spending")

def Data_analysis():
    st.subheader("MARKET CUSTOMER DATASET")
    st.dataframe(m_df)

    st.subheader("DATA CLEANING")
    m_df["MINIMUM_PAYMENTS"] = m_df["MINIMUM_PAYMENTS"].fillna(m_df["MINIMUM_PAYMENTS"].mean())
    m_df["CREDIT_LIMIT"] = m_df["CREDIT_LIMIT"].fillna(m_df["CREDIT_LIMIT"].mean())
    st.subheader("FILL MINIMUM PAYMENT WITH MEAN")
    st.dataframe(m_df["MINIMUM_PAYMENTS"])

    st.subheader("FILL CREDIT LIMIT WITH MEAN")

    st.dataframe(m_df["CREDIT_LIMIT"])

    st.subheader("AFTER CLEANING NULL VALUES BY COLUMNS")
    st.dataframe(m_df.isnull().sum())
def Data_visualization():
    st.subheader("Plots")
    showPyplotGlobalUse = False
    # Plot for balance
    fig, ax = plt.subplots(figsize=(15, 5))

    st.subheader('Histogram of BALANCE')
    plt.hist(m_df['BALANCE'], bins=50)
    st.pyplot(fig)



    st.subheader('Boxplot of PURCHASES grouped by TENURE')
    m_df.boxplot(column='PURCHASES', by='TENURE')
    st.pyplot()

    st.subheader("PAIR PLOT")
    sns.pairplot(m_df[['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                               'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                               'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS',
                               'PRC_FULL_PAYMENT', 'TENURE']], kind='scatter')
    st.pyplot()


    sns.violinplot(x='cluster', y=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                                   'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                                   'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX',
                                   'PURCHASES_TRX', 'PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'], data=m_df)
    st.pyplot()
def clustering():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans


    # Display raw data
    st.title("Customer Segmentation based on Balance and Purchases")

    # Display raw data
    st.subheader("Raw Data")
    # Assuming m_df is already defined and imported
    m_df.dropna(axis=1, inplace=True)
    st.dataframe(m_df)

    # Selecting only 'BALANCE' and 'PURCHASES' columns
    m_df_selected = m_df[['BALANCE', 'PURCHASES']]

    # Scaling data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(m_df_selected)

    # Elbow method to determine optimal number of clusters
    wcss = []
    max_clusters = 10  # Maximum number of clusters to try
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(scaled_df)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    st.pyplot(fig)

    # Perform KMeans clustering
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=4, step=1)
    kmeans_model = KMeans(n_clusters=num_clusters)
    kmeans_model.fit(scaled_df)

    # Add the cluster labels to the original dataframe
    m_df_selected['cluster'] = kmeans_model.labels_

    # Display clustering results
    st.subheader("Clustering Results")
    st.write(m_df_selected)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(m_df_selected['BALANCE'], m_df_selected['PURCHASES'], c=m_df_selected['cluster'], cmap='viridis')
    plt.xlabel('Balance')
    plt.ylabel('Purchases')
    plt.title('Customer Segmentation Results')
    plt.colorbar(ticks=range(num_clusters))
    st.pyplot(plt)


def retail_analysis():
    sl = pd.read_csv('F:/DEGREE/SEM - 8/Project/CS/Datasets/sales.csv')

    import streamlit as st
    st.header("Sales Dataset")
    st.write(sl.head(5))

    st.subheader("Sales Dataset Coloumn info")
    st.write('Segment')
    st.write(sl['Segment'].value_counts())
    st.write('Region')
    st.write(sl['Region'].value_counts())
    st.write('Category')
    st.write(sl['Category'].value_counts())
    st.write('Sub-Category')
    st.write(sl['Sub-Category'].value_counts())
    st.write('Ship Mode')
    st.write(sl['Ship Mode'].value_counts())

    st.title('Sales in U.S.')
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Initialize Plotly in Jupyter Notebook mode
    import plotly.io as pio
    pio.renderers.default = 'notebook_connected'

    # Create a mapping for all 50 states
    all_state_mapping = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
        "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
        "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
    }

    # Add the Abbreviation column to the DataFrame
    sl['Abbreviation'] = sl['State'].map(all_state_mapping)

    # Group by state and calculate the sum of sales
    sum_of_sales = sl.groupby('State')['Sales'].sum().reset_index()

    # Add Abbreviation to sum_of_sales
    sum_of_sales['Abbreviation'] = sum_of_sales['State'].map(all_state_mapping)

    # Create a choropleth map using Plotly
    fig = go.Figure(data=go.Choropleth(
        locations=sum_of_sales['Abbreviation'],
        locationmode='USA-states',
        z=sum_of_sales['Sales'],
        hoverinfo='location+z',
        showscale=True
    ))

    fig.update_geos(projection_type="albers usa")
    fig.update_layout(
        geo_scope='usa',
        title='Total Sales by U.S. State'
    )

    st.write(fig)

    st.title('Sales in U.S. Bar Graph')
    # Group by state and calculate the sum of sales
    sum_of_sales = sl.groupby('State')['Sales'].sum().reset_index()

    # Sort the DataFrame by the 'Sales' column in descending order
    sum_of_sales = sum_of_sales.sort_values(by='Sales', ascending=False)

    # Create a horizontal bar graph
    plt.figure(figsize=(10, 13))
    ax = sns.barplot(x='Sales', y='State', data=sum_of_sales, ci=None)

    plt.xlabel('Sales', color="white")
    plt.ylabel('State', color="white")
    plt.title('Total Sales by State', color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")

    st.pyplot(plt.gcf(), transparent=True)

    st.header('Summary of Category and Sub-Category')
    sl_summary = sl.groupby(['Category', 'Sub-Category'])['Sales'].sum().reset_index()
    st.write(sl_summary)

    import plotly.express as px
    st.header('Pie of Category and Sub-Category')
    # Create a nested pie chart
    fig = px.sunburst(
        sl_summary,
        path=['Category', 'Sub-Category'],
        values='Sales',
    )

    st.write(fig)

def sales_pred():
    import plotly.graph_objs as go
    from prophet import Prophet

    def load_data(file_path):
        df = pd.read_csv(file_path)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
        monthly_data = df.resample('M', on='Order Date').sum()
        monthly_data.reset_index(inplace=True)
        monthly_data = monthly_data.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
        return monthly_data

    # Function to train the Prophet model and make predictions
    def prophet_forecast(data_train, periods):
        model = Prophet()
        model.fit(data_train)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        return forecast

    # Main function to run the Streamlit app
    def main():
        st.title("Sales Forecasting App")

        # Sidebar section for uploading file and setting parameters

        uploaded_file = "F:/DEGREE/SEM - 8/Project/CS/Datasets/sales.csv"
        if uploaded_file is not None:
            data_train = load_data(uploaded_file)

            # Display uploaded data
            st.subheader("Uploaded Data")
            st.write(data_train)
            periods = st.slider("Number of months to forecast", min_value=1, max_value=36, value=12)

            # Train model and make predictions
            forecast = prophet_forecast(data_train, periods)

            # Plotting with Plotly
            fig = go.Figure()

            # Add actual sales data
            fig.add_trace(go.Scatter(x=data_train['ds'], y=data_train['y'], mode='lines', name='Actual Sales'))

            # Add forecasted sales data
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

            # Add uncertainty interval
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_lower'], forecast['yhat_upper'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Uncertainty Interval'
            ))

            # Update layout
            fig.update_layout(title="Sales Forecast",
                              xaxis_title="Date",
                              yaxis_title="Sales",
                              legend=dict(x=0, y=1, traceorder="normal"),
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)',
                              )

            # Display plot
            st.plotly_chart(fig)
    if __name__ == "__main__":
        main()

def walmart_visual():
    import plotly.express as px

    st.title('Walmart sales visualization')
    # Load features.csv
    features_df = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/features.csv")

    # Load stores.csv
    stores_df = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/stores.csv")

    holiday_pie_chart = px.pie(features_df, names='IsHoliday', title='Holiday Distribution')
    st.plotly_chart(holiday_pie_chart)

    # Histogram for store sizes
    store_size_histogram = px.histogram(stores_df, x='Size', title='Store Size Distribution')
    st.plotly_chart(store_size_histogram)

    # Scatter plot for temperature vs. sales
    temperature_sales_scatter = px.scatter(features_df, x='Temperature', y='Fuel_Price',
                                           title='Temperature vs. Fuel Price')
    st.plotly_chart(temperature_sales_scatter)

    # Merge features_df with stores_df to get the store type
    merged_df = pd.merge(features_df, stores_df, on='Store')

    # Calculate total sales for each store grouped by store type
    total_sales_by_store_type = merged_df.groupby(['Type', 'Store'])['Temperature'].sum().reset_index()

    # Find the store with the highest sales within each store type
    highest_sales_store_by_type = total_sales_by_store_type.groupby('Type')['Temperature'].idxmax()

    # Get the details of the stores with the highest sales within each store type
    stores_with_highest_sales = total_sales_by_store_type.loc[highest_sales_store_by_type]

    # Create a bar chart to visualize the stores with the highest sales within each store type
    sales_by_store_type_bar_chart = px.bar(stores_with_highest_sales, x='Type', y='Temperature', color='Store',
                                           title='Stores with Highest Sales by Type')
    st.plotly_chart(sales_by_store_type_bar_chart)

    # Allow the user to select a store
    selected_store = st.selectbox('Select a store:', stores_df['Store'])

    # Filter the features dataframe for the selected store
    selected_store_data = features_df[features_df['Store'] == selected_store]

    # Create a line chart for unemployment over time for the selected store
    unemployment_line_chart = px.line(selected_store_data, x='Date', y='Unemployment',title=f'Unemployment Over Time for Store {selected_store}')
    st.plotly_chart(unemployment_line_chart)
def Walmart_Pred():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import streamlit as st

    # Load training data
    def load_train_data():
        train_data = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/train_w.csv")
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        train_data['IsHoliday'] = train_data['IsHoliday'].astype(int)
        return train_data

    # Load test data
    def load_test_data():
        test_data = pd.read_csv("F:/DEGREE/SEM - 8/Project/CS/Datasets/test.csv")
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        test_data['IsHoliday'] = test_data['IsHoliday'].astype(int)
        return test_data

    # Train model
    def train_model(train_data):
        X = train_data.drop(columns=['Weekly_Sales', 'Date'])
        y = train_data['Weekly_Sales']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model, X_val, y_val

    # Make predictions
    def make_predictions(model, test_data):
        test_predictions = model.predict(test_data.drop(columns=['Date']))
        test_data['Predicted_Weekly_Sales'] = test_predictions
        return test_data

    def main():
        st.title("Walmart Sales Prediction")

        # Load data
        train_data = load_train_data()
        test_data = load_test_data()

        # Train model
        st.subheader("Training Model")
        model, X_val, y_val = train_model(train_data)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        st.write("Mean Absolute Error on Validation Data:", mae)

        # Make predictions
        st.subheader("Making Predictions on Test Data")
        test_predictions = make_predictions(model, test_data)
        st.write(test_predictions.sort_values(by='Date'))

    if __name__ == "__main__":
        main()

if choose == "About":
    about()
elif choose == "Customer Wise Data Analysis":
    customer()
elif choose == "Region Wise Data Analysis":
    region()
elif choose == "Product Wise Data Analysis":
    # Execute the Signup.py script as a separate process
    product()
elif choose == "Contact":
    contact()
elif choose == "MALL CUSTOMERS ANALYSIS":
    mall_data_analysis()
elif choose == "MALL CUSTOMERS DATA VISUALIZATION":
    MALL_CUSTOMERS_DATA_VISUALIZATION()
elif choose == "MACHINE LEARNING(MALL DATASET)":
    ml()
elif  choose == "Data_analysis(Market Data)":
    Data_analysis()
elif choose == "Data_Visualization(Market Data)":
    Data_visualization()

elif choose == "Clustering(Market Data)":
    clustering()

elif  choose == "Data Analysis (STORE)":
    retail_analysis()
elif choose =="PREDICTION (SALES)":
    sales_pred()

elif choose == "Walmart_Sales_Visualization":
    walmart_visual()

elif choose == "Walmart_Sales_Prediction":
    Walmart_Pred()
elif choose == "LOGOUT":
    st.markdown(f'<meta http-equiv="refresh" content="2;url=http://localhost:8501/Login">', unsafe_allow_html=True)
    st.header("Redirecting...")