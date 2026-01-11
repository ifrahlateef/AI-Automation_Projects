# Survey Insights AI: Orchestrating Automated NLP & Sentiment Intelligence

## Executive Summary & Strategic Leadership
Recognizing a critical operational bottleneck in the quarterly survey cycle, I **proactively initiated and spearheaded** the digital transformation of our feedback analysis workflow. Previously, the process relied on manual qualitative review—a tedious task that took over 168 man-hours per quarter and was prone to subjective bias.

I engineered an **End-to-End AI Pipeline** that automates recipient selection and utilizes **Natural Language Processing (NLP)** to extract actionable intelligence from thousands of verbatim comments. 

###  High-Level Impact:
* **Operational Velocity:** Reduced the analysis and reporting cycle from **1 week to 1 hour** (99% efficiency gain).
* **Data-Driven Consistency:** Replaced subjective manual reviews with a repeatable, mathematically grounded clustering model.
* **Democratized AI:** Empowered non-technical HR and Success teams to execute advanced Python-based analytics through a custom-built, "One-Click" interface.

---

##  Technical Architecture & Advanced Analytics
I designed a multi-layered architecture that bridges cloud-scale data movement with local machine learning execution.

### 1. Automated Data Orchestration
* **Azure Data Factory (ADF):** Engineered a robust pipeline to ingest raw survey responses from diverse sources into a centralized environment.
* **Power Automate Integration:** Developed a seamless trigger flow to manage data movement and stakeholder notifications.
* **Intelligent Recipient Logic:** Designed and deployed a custom algorithm to programmatically identify and select target survey participants for each quarterly cycle.

### 2. NLP & Unsupervised Machine Learning Pipeline
To handle the high-volume qualitative feedback, I developed an analytical engine using:
* **Text Pre-processing:** Implemented a sophisticated NLP stack including **Tokenization, Stop-word Removal, and TF-IDF (Term Frequency-Inverse Document Frequency)** for vectorization.
* **Dimensionality Reduction (PCA):** Applied **Principal Component Analysis** to reduce feature noise and identify the most significant variance in feedback topics.
* **K-Means Clustering:** Developed an unsupervised learning model to automatically group comments into "Pain Point Clusters," identifying recurring themes without human intervention.
* **Statistical Correlation:** Conducted correlation analysis to isolate the specific drivers (features) that most significantly impact the **Net Promoter Score (NPS)**.

---

##  Design for Self-Sufficiency (The "One-Click" Solution)
A key senior-level achievement was making this advanced technology accessible to non-technical Business Delivery and HR teams:
* **CLI Interface:** Built a simplified command-line interface where users can simply drop an Excel file and click **"RUN"** to execute the entire Python backend.
* **Knowledge Transfer & Literacy:** Facilitated hands-on workshops where I translated technical concepts into business logic (e.g., explaining Clustering as "automated grouping of similar concerns").
* **User Documentation:** Authored a concise User Guide to ensure 100% self-sufficiency, fostering a culture of technical independence.

---

##  Business Value & ROI
* **Immediate Insight Generation:** Stakeholders now receive NPS drivers and thematic patterns on day one of data collection.
* **Strategic Alignment:** By identifying the core factors driving NPS, leadership can now prioritize resource allocation based on actual customer pain points.
* **Strengthened Partnerships:** Bridged the gap between technical and non-technical departments, creating a unified, data-driven approach to Customer Success.

---

##  Technical Stack
* **Cloud & Automation:** Azure Data Factory, Power Automate
* **Machine Learning:** Python (Scikit-Learn, NLTK, Pandas)
* **Algorithms:** K-Means Clustering, PCA, TF-IDF, Correlation Analysis
* **Deployment:** CLI-based Python Application
