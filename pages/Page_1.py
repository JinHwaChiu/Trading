import streamlit as st
from utils import spx_history,single_bar_stats,day_range_stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="SPX Range Analysis", layout="wide")
st.title("SPX Monthly Candle")
st.write("This page contains candlestick plots.")


# if st.sidebar.button("執行分析"):

# Load your data into a DataFrame
# Ensure your DataFrame has columns: 'Date', 'Open', 'High', 'Low', 'Close'
# and that 'Date' is of datetime type.
spx = spx_history('^GSPC', period='1y', interval='1mo')
df = single_bar_stats(spx)
### 20 Days Vivration Statistic (H - L)
# Calculate the daily price range (High - Low)
stats_df = day_range_stats(20,spx_history('^GSPC', period='1mo', interval='1d'))

# Create subplots with 1 row and 2 columns
# Create subplots with 2 rows and 2 columns
fig = make_subplots(
    rows=2, cols=2,
    column_widths=[0.3, 0.7],  # Adjust the width ratio between table and chart
    specs=[
        [{"type": "table"}, {"type": "candlestick"}],
        [{"type": "table"}, None]
    ],
    subplot_titles=("Month K", "Month K", "20 days range", "")  # Placeholder for subplot titles
)

# Add table to the first column
# Transpose the DataFrame to have dates as columns and price types as rows
df_fil = df[['Date','Open','High','Low','Median','%','C-O']]
df_transposed = df_fil.set_index('Date').T
# Convert the transposed DataFrame values to a list of lists
transposed_values = df_transposed.values.tolist()
# Transpose the list of lists to switch rows and columns
transposed_list = list(map(list, zip(*transposed_values)))
fig.add_trace(
    go.Table(
        
        header=dict(
            values=[''] + [date.strftime('%Y-%m-%d') for date in df['Date']],
            align='center'
        ),
        cells=dict(
            values=[df_transposed.index] + transposed_list,
            align='center'
        )
    ),
    row=1, col=1
)


 # Add statistical summary table to the first column
fig.add_trace(
    go.Table(
        header=dict(
            values=['Statistic', 'Value'],
            align='center'
        ),
        cells=dict(
            values=[stats_df.index, stats_df['Value']],
            align='center'
        )
    ),
    row=2, col=1
)


# Add candlestick chart to the second column
fig.add_trace(
    go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlesticks',
        
    ),
    row=1, col=2
)

# Add median values as a scatter plot on the candlestick chart
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Median'],
        mode='markers',
        marker=dict(size=6, color='blue'),
        name='Median'
    ),
    row=1, col=2
)


# Customize layout and set figure size
# Update layout for a tighter fit
fig.update_layout(
    title='',
    xaxis_title='Date',
    xaxis_rangeslider_visible=False,
    showlegend=False,
    width=1200,   # Reduced width
    height=700,   # Reduced height
    margin=dict(l=20, r=20, t=20, b=20)  # Tight margins
)


# Format x-axis to display only the month
# fig.update_xaxes(tickformat='%b')  # '%b' displays the abbreviated month name



# Show the figure
st.plotly_chart(fig, use_container_width=True)
