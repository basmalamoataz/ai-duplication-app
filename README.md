# 🤖 AI Duplication Finder

This project is an intelligent web application designed to identify similar or duplicate work orders from an Excel file. In industrial maintenance and operations, duplicate work orders are often created for the same issue, leading to wasted resources and confusion. This tool uses a powerful AI language model to understand the *meaning* of the work order descriptions, finding similarities that simple keyword searches would miss.

---

## ✨ Key Features

* **AI-Powered Semantic Search:** Utilizes a state-of-the-art `SentenceTransformer` model to understand the context and meaning of work order descriptions, not just keywords.
* **Intelligent Grouping:** Automatically groups work orders by **Location** and **Asset** to ensure that only relevant tasks are compared.
* **Smart Filtering:**
    * Automatically excludes work orders with a status of "COMP" (Completed) or "CAN" (Cancelled).
    * Automatically excludes scheduled Preventive Maintenance tasks by filtering out rows that contain a `PM Number`.
* **Interactive Web Interface:** A user-friendly GUI built with Streamlit that allows non-technical users to upload their data and get results with the click of a button.
* **Clear Results Table:** Displays the found pairs of similar work orders in an organized, easy-to-read table, sorted by similarity score.
* **CSV Export:** Allows the user to download the results table as a CSV file for further analysis or reporting.

---

## 🚀 Live Application

You can try the live application here:
**(https://your-username-ai-duplication-app-xyz.streamlit.app/)** *<-- Replace this with the actual public link to your Streamlit app!*

### Application Screenshot

*(This is a placeholder - on GitHub, you can upload a screenshot of your app and link to it here)*

---

## 🛠️ How to Use the Deployed App

1.  **Open the Link:** Navigate to the public Streamlit URL.
2.  **Upload File:** Click the "Browse files" button and select the Excel file (`.xlsx`) containing your work order data.
3.  **Start Analysis:** Click the "Start" button to begin the AI analysis.
4.  **Review and Download:** The results will appear in a table on the screen. You can use the "Download results as CSV" button to save the findings.

---

## 💻 How to Run Locally (for Developers)

To run this application on your own computer, follow these steps.

**Prerequisites:**
* Python 3.8+ installed
* `pip` (Python package installer)

**1. Clone the Repository:**
```bash
git clone [https://github.com/your-username/ai-duplication-app.git](https://github.com/your-username/ai-duplication-app.git)
cd ai-duplication-app
