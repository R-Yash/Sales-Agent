# Sales Agent Chatbot

This chatbot is designed to assist customers by answering questions about products from a specific seller.

## Solution Approach
The first step was to extract the sample data. I wrote a basic python script to send request to the url and get the json data. This data was cleaned a bit and a CSV file was obtained. This is used as the context for the RAG System.

The next step was to build a Chatbot. The chatbot uses RAG to get rows from the table, relevent to the query. It uses these rows, along with important points from chat history to answer any questions that the user may have regarding the products.

## Prerequisites
- Python
- An OpenAI API key

## Steps to Run

1. **Install Dependencies**  
   Install all the required packages by running:  
   `pip install -r requirements.txt`
   
3. **Run the `get_data.py` file to get products data as a csv**  

4. **Run `sales_agent.py` to start the bot and ask any questions. To exit the bot, just type exit**  

