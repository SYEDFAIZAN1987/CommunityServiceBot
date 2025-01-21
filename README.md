# Carefirst Chatbot - Proprietary AI Chatbot

![Carefirst Logo](carefirst_logo.png)

## 📌 Project Overview

The **Carefirst Chatbot** is a **proprietary AI-driven assistant** designed for **Carefirst Ontario Services**, primarily serving **Chinese-speaking seniors** in Scarborough, Markham, Richmond Hill, Newmarket, North York, and Downtown Toronto.

This chatbot is developed under the supervision of **Dr. Yvonne Leung** and is the **intellectual property of Dr. Yvonne Leung**. Unauthorized use, modification, or distribution is strictly prohibited.

The chatbot leverages **Giskard** for robust **Retrieval-Augmented Generation (RAG) evaluation** using **RAGAS metrics** to ensure accuracy and relevancy of responses.

---

## 🚀 Features

- **Conversational AI** powered by `LangChain` and `OpenAI GPT-4o`
- **Retrieval-Augmented Generation (RAG)** for knowledge-based responses
- **Vector Search** using `FAISS`
- **Live Web Search** using `Tavily API`
- **RAG Evaluation** using `Giskard` and `RAGAS`
- **Pre-trained Knowledge Base** from Carefirst Ontario's data

---

## 📂 Project Structure

```
Carefirst/
│── app.py              # Chatbot implementation
│── eval.py             # Giskard-based evaluation script
│── rag_eval_report.html # RAGAS evaluation report (output)
│── base.csv            # Knowledge base for RAG testing
│── my_testset.jsonl    # Generated test set
│── carefirst_logo.png  # Branding image
│── requirements.txt    # Dependencies list
│── README.md           # This file
```
---

## 🔧 Installation & Setup

### 1️⃣ Prerequisites

Ensure you have **Python 3.8+** installed along with `pip`. Install dependencies:

```bash
pip install -r requirements.txt
2️⃣ Set Up API Keys
Set up OpenAI API Key and Tavily API Key in your environment:

bash
Copy
Edit
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
For Windows (PowerShell):

powershell
Copy
Edit
$env:OPENAI_API_KEY="your-openai-api-key"
$env:TAVILY_API_KEY="your-tavily-api-key"
🎯 Running the Chatbot
Launch the chatbot using:

bash
Copy
Edit
python app.py
This will start a Gradio web app, allowing users to interact with the chatbot.

📊 RAG Evaluation with Giskard
Step 1: Generate the Test Set
Run eval.py to generate and evaluate the test set using Giskard:

bash
Copy
Edit
python eval.py
This performs:

Test Set Generation (my_testset.jsonl)
RAG Evaluation (stored in rag_eval_report.html)
🧪 RAGAS Metrics Evaluation
The chatbot’s accuracy is validated using RAGAS metrics, including:

Context Recall (relevant document retrieval)
Faithfulness (ground truth consistency)
Context Precision (correct document matching)
Answer Relevancy (response appropriateness)
To run RAGAS-based evaluation:

bash
Copy
Edit
python ragas.py
This generates a detailed RAGAS evaluation report, stored as:

yaml
Copy
Edit
ragas_eval_report.html
View it in any web browser.

⚠️ Proprietary Notice
⚠️ IMPORTANT: This chatbot is the intellectual property of Dr. Yvonne Leung. Unauthorized reproduction, distribution, or modification is strictly prohibited.

For inquiries, contact Dr. Yvonne Leung.

🛠️ Future Enhancements
Improve multi-language support for users with limited English proficiency
Optimize vector search for faster response retrieval
Expand knowledge base for broader service coverage
Enhance accuracy of RAGAS metrics with fine-tuned models
👨‍💻 Developers
Syed Faizan (Lead Developer)
Dr. Yvonne Leung (Supervisor & Owner)
✅ Key Inclusions
✅ Branding & Proprietary Notice
✅ Evaluation Process using Giskard & RAGAS
✅ Installation & Setup Guide
✅ Project Structure & Key Files


