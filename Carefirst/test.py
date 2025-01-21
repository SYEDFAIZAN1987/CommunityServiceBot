import pandas as pd
import giskard
from giskard.rag import generate_testset, KnowledgeBase, QATestset

# Load the knowledge base from base.csv
file_path = "base.csv"  
df = pd.read_csv(file_path)

# Initialize the Giskard Knowledge Base
knowledge_base = KnowledgeBase.from_pandas(df, columns=["question", "answer"])

# Generate a test set with 60 questions
testset = generate_testset(
    knowledge_base,
    num_questions=60,  # Change if needed
    language="en",  # Auto-detect if not provided
    agent_description="A customer support chatbot for Carefirst Ontario services.",  # Helps generate better questions
)

# Save the generated test set
testset.save("my_testset.jsonl")
print("✅ Test set saved as 'my_testset.jsonl'.")

# Load the test set back
loaded_testset = QATestset.load("my_testset.jsonl")

# Convert test set to pandas DataFrame for analysis
df_test = loaded_testset.to_pandas()

# Save test dataset as CSV for review
df_test.to_csv("testset_review.csv", index=False)
print("✅ Test set converted to CSV as 'testset_review.csv'.")

# Display the first few rows
print(df_test.head())
