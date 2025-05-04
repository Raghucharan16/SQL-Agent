import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sqlparse
import psycopg2
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load fine-tuned model
model_path = "./sqlcoder_finetuned"
base_model = AutoModelForCausalLM.from_pretrained(
    "defog/sqlcoder-7b",
    load_in_4bit=True,
    device_map="cpu",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Database schema (replace with your schema)
SCHEMA = """Table: customers
- customer_id (INT, Primary Key): Unique identifier for a customer.
- name (VARCHAR): Customer's full name.
- email (VARCHAR): Customer's email address.

Table: orders
- order_id (INT, Primary Key): Unique identifier for an order.
- customer_id (INT, Foreign Key: references customers.customer_id): ID of the customer who placed the order.
- order_date (DATE): Date the order was placed.
- amount (DECIMAL): Total order amount."""

# Database connection
def execute_query(query):
    try:
        conn = psycopg2.connect(
            dbname="your_db",
            user="your_user",
            password="your_password",
            host="localhost"
        )
        cur = conn.cursor()
        cur.execute(query)
        results = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        return f"Error: {str(e)}"

# Generate SQL query
def generate_sql_query(question):
    prompt = f"### Schema:\n{SCHEMA}\n\n### Question:\n{question}\n\n### SQL Query:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract SQL query from response
    sql_query = sql_query.split("### SQL Query:")[-1].strip()
    return sql_query

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_question = data.get("question", "")
    try:
        sql_query = generate_sql_query(user_question)
        # Validate SQL syntax
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return jsonify({"error": "Invalid SQL query", "query": sql_query})
        # Execute query
        results = execute_query(sql_query)
        return jsonify({"query": sql_query, "results": results})
    except Exception as e:
        return jsonify({"error": str(e), "query": sql_query})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
