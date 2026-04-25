# NETCONF RAG GUI Demo  
Agentic RAG → NETCONF XML → Orchestrator‑Worker DAG execution with a NiceGUI frontend, powered by Microsoft Agent Framework (MAF) 1.2.0.

---

## 🚀 Overview  
This project demonstrates an **Agentic RAG–driven network automation workflow** using the **Microsoft Agent Framework (MAF) 1.2.0**.  
Natural‑language intent is converted into **NETCONF XML**, executed through a **two‑task DAG**, and visualized through a **NiceGUI web interface**.

It is designed as a **minimal, reproducible demo** showing how AI‑assisted intent, NETCONF automation, and orchestrator‑worker patterns can be combined for modern network operations.

---

## 🎥 Demo Video  (55 sec)  

https://github.com/user-attachments/assets/d058a8bd-f54a-428c-853e-14c2a3186989

> **Demonstration: Multi-Task DAG Execution via MAF 1.2.0**
> Watch how a simple intent ("Delete VLAN 50 and add VLAN 100") is autonomously handled by MAF agents, with full visibility through NiceGUI and JupyterHub.

- NiceGUI UI  
- Natural‑language input  
- RAG‑generated NETCONF XML  
- DAG execution (2 tasks)  
- Before/after CLI output via JupyterHub  

---

## ✨ Features  

### 🔹 Agentic RAG  
- Converts natural‑language intent into structured NETCONF XML  
- Uses a local FAISS vector store (excluded from repo)

### 🔹 NETCONF Automation  
Demo workflow:  
- **Task 1:** Delete VLAN 50  
- **Task 2:** Add VLAN 100  

### 🔹 Orchestrator‑Worker DAG (MAF 1.2.0)  
- Two‑task DAG executed by MAF  
- Clear separation of orchestration and execution  
- Ideal for multi‑step network workflows

### 🔹 NiceGUI Frontend  
- Input natural‑language intent  
- Preview generated XML  
- Trigger DAG execution  
- Display results

### 🔹 JupyterHub CLI Integration  
- Shows before/after device state  
- Makes the automation process transparent

---

## 🏗️ Architecture
```
Natural Language Input
│
▼
Agentic RAG
│
▼
NETCONF XML Generation
│
▼
MAF Orchestrator
│
▼
DAG (2 Tasks)
├── Task 1: Delete VLAN 50
└── Task 2: Add VLAN 100
│
▼
Network Device
```
---

## ⚙️ Quickstart

### 1. Clone the repository  
```bash
git clone https://github.com/hidemi-k/maf-netconf-rag-gui.git
cd maf-netconf-rag-gui
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Prepare configuration
```bash
cp config.ini.example config.ini
```
### 4. Run the NiceGUI app
```bash
python maf_netconf_rag_gui.py
```

### 5. Open the UI

Navigate to `http://localhost:8080` in your browser.  
You will be automatically redirected to `http://localhost:8080/netconf_rag`.

> **Note:** On startup, the following warning from the BERT model loader can be safely ignored:
> ```
> embeddings.position_ids | UNEXPECTED
> ```
> This is expected behavior when loading `BAAI/bge-large-en-v1.5` and does not affect functionality.

### 📁 Repository Structure
```
maf-netconf-rag-gui/
 ├── maf_netconf_rag_gui.py      # Main NiceGUI + RAG + NETCONF app
 ├── config.ini.example          # Example configuration
 ├── requirements.txt            # Python dependencies required for the UI, RAG engine, and NETCONF automation
 ├── LICENSE                     # MIT License
 ├── .gitignore                  # Includes faiss_db/ exclusion
 ├── faiss_db/ (ignored)         # Local vector store
 └── .ipynb_checkpoints/         # Jupyter artifacts
```
### 🧩 Requirements
- Python 3.10+
- NiceGUI
- FAISS
- Microsoft Agent Framework (MAF) 1.2.0
- NETCONF-capable network device
- JupyterHub (optional)

### 📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.

### 🙌 Acknowledgements
- Microsoft Agent Framework (MAF) team
- NiceGUI project
- Network automation community
