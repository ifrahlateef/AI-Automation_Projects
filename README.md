# Managed Service Provider (MSP) Financial & Operational Analytics Pipeline

##  Project Overview & Business Value
This project automates the financial auditing and performance monitoring of Desktop Support Services. Previously, tracking service tasks and associated costs was a manual, error-prone process that required **5-6 working days** of manual effort per month.

###  Key Transformations:
* **Efficiency:** Reduced reporting turnaround from **40 hours to near real-time** via automated scheduling.
* **Accuracy:** Eliminated human error in "Activity Point" calculations, for subsequent  monetary conversion ensuring 100% reconciliation with third-party billing.
* **Scalability:** Replaced crashing Excel workbooks with a robust Python/SQL backend capable of handling high-volume data loads.
* **Auditability:** Introduced a structured "Override" mechanism for transparency in last-minute adjustments.

---

##  System Architecture & Workflow
The solution is built as a modular Python-based ETL (Extract, Transform, Load) pipeline hosted on a central server for global accessibility.



### 1. Data Ingestion & Modularisation
* **Custom Library Architecture:** Developed a proprietary Python repository containing reusable modules and utility functions to ensure code maintainability and reusable principles.
* **SQL Integration:** Automated extraction from SQL Server, performing complex **multi-table joins** to synthesize a holistic view of hardware, software, and service tickets.

### 2. Business Logic & Categorization
* **Segmented Analysis:** Custom functions categorize tasks based on distinct business environments: **Corporate** vs. **Contact Center**.
* **Advanced Pattern Matching:** Utilized the 're' (Regular Expression) library to parse unstructured task descriptions and assign accurate classifications.
* **Rolling 12-Month Horizon:** Maintained a dynamic 12-month window to provide stakeholders with year-over-year (YoY) trend analysis and seasonality insights.

### 3. Financial Engineering (The Activity Point Model)
The core of the system is an automated valuation engine that assigns "Activity Points" (converted to £) based on task complexity:
* **Dynamic Tiering:** Automatically classifies tasks into **Simple, Moderate, or Complex** tiers based on completion methods.
* **Discrepancy Management:** Integrated a "Override" feature. This allows the Desktop Support Lead to manually adjust figures with an audit trail, providing flexibility for ad-hoc work or avoid billing disputes.

### 4. Automation & Deployment
* **Server-Side Hosting:** The script is hosted on a central server, ensuring high availability.
* **Task Scheduling:** Orchestrated via **Windows Task Scheduler**, the pipeline runs daily to ensure that the "Latest Figures" are always available to the leadership team without manual intervention.

---

## Technical Stack
* **Backend:** Python (Pandas, NumPy, SQLAlchemy)
* **Database:** SQL Server (Complex Joins, View Optimization)
* **Text Processing:** Regular Expressions (`re`)
* **DevOps:** Windows Task Scheduler, Local Server Hosting
* **Version Control:** Internal Python Library Repository

---

## Analytical Insights Provided
* **Cost Variance Analysis:** Comparison between internal calculated costs and MSP-provided invoices.
* **Volume Distribution:** Task breakdown by complexity tier and business unit.
* **Resource Utilization:** Highlighting monthly peaks in desktop support demand.
