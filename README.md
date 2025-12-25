# NexKey DealMatch — Deep Learning Real Estate Chatbot

NexKey DealMatch is an end-to-end **deep learning–powered chatbot recommender system** for real estate wholesaling.
Users describe their buy box in natural language, and the system returns the **top matching deals** using a modern **retrieve → rerank** architecture.

This project demonstrates a complete **Machine Learning Engineer workflow**:
data exploration, model training, evaluation, production inference, and a real frontend UI.

---

## 🚀 What This Project Does

- Accepts user input like:
  > “3 bed in AZ under 350k, entry under 20k, payment under 2500”
- Retrieves candidate deals using a **Dual Encoder**
- Reranks candidates using a **Cross Encoder**
- Returns the **top 5 deals**
- Asks clarifying questions when prompts are vague

---

## 🧠 ML Architecture

User → Next.js Chat UI → FastAPI Backend  
→ Dual Encoder Retrieval → Cross Encoder Reranking → Top-K Deals

### Models
- **Dual Encoder** (fast semantic retrieval)
- **Cross Encoder** (high-accuracy reranking)

---

## 📁 Project Structure

```
NexKey-DealMatch/
├── data/
├── models/
├── notebooks/
├── src/
│   └── app/
├── frontend/
├── requirements.txt
└── README.md
```

---

## ▶️ Run From Scratch (Local)

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.app.main:app --reload --port 8000
```

Check:
- http://localhost:8000/health
- http://localhost:8000/version

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open:
- http://localhost:3000

---

## 💬 Example Prompts

- 3 bed in AZ under 350k, entry under 20k, payment under 2500
- Phoenix AZ 4 bed, ARV 550k+, entry under 25k
- Subto deal in Arizona, 3 bed minimum

---

## 📊 Evaluation

Evaluation is performed in `notebooks/10_final_test_report.ipynb` using Recall@K and NDCG@K.

---

## 🛠 Tech Stack

- PyTorch
- FastAPI
- Next.js + Tailwind CSS
- Pandas / NumPy

---

## 📌 Notes

- Dataset is synthetic and intended for learning
- Architecture mirrors real-world recommender systems
- Models load once at API startup for efficiency
