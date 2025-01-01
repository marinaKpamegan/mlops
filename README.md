# **Iris Model Training and Prediction API**

This project provides an end-to-end solution for training, serving, and managing machine learning models for the classic **Iris dataset**. The system uses **FastAPI** as a backend, **MLFlow** for tracking experiments, and **Streamlit** for the client-side interface.

---

## **Features**
- Train and evaluate models on the Iris dataset:
  - Supports algorithms like **KNN**, **Decision Tree**, and **Random Forest**.
  - Automatically logs metrics and artifacts using **MLFlow**.
- Predict iris species based on input features.
- Dynamically update models from MLFlow using version control.
- Interactive user interface with **Streamlit**.

---

## **Dataset**
The project uses the **Iris dataset**, which includes:
- **Features**:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Target**: Species of iris flowers (`setosa`, `versicolor`, `virginica`).

---

## **Architecture**

- **Backend (FastAPI)**:
  - API endpoints for training, predictions, and model management.
  - Integration with MLFlow for model tracking and versioning.
- **Frontend (Streamlit)**:
  - User-friendly interface to train models, test predictions, and update models.
- **MLFlow**:
  - Tracks experiments, stores metrics, and manages model versions in sqlite db.
- **Docker Compose**:
  - Containerized services for seamless deployment.

---

## **Setup Instructions**

### **Prerequisites**
- Python 3.8+
- Docker or Docker Desktop installed.

---

### **Local Setup**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   docker-compose up --build

### **Some images**

![alt text](<Capture d'écran 2024-12-29 191202.png>) ![alt text](<Capture d'écran 2024-12-29 183017.png>) ![alt text](<Capture d'écran 2024-12-29 183113.png>) ![alt text](<Capture d'écran 2024-12-29 191132.png>)

