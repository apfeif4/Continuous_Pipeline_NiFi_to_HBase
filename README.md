---

# 🔄 Continuous Data Pipeline: NiFi → HDFS → Hive → PySpark → HBase

This project demonstrates the design and implementation of a continuous data pipeline in a big data ecosystem, using tools such as Apache NiFi, HDFS, Hive, PySpark, and HBase. It highlights how data can flow from ingestion to predictive analytics and finally to persistent storage for retrieval.

---

## ✅ Project Status

**Completed**

---

## 🎯 Objective

The goal of this project is to build an end-to-end continuous data pipeline in a big data environment. The process involves:

1. Ingesting a dataset into HDFS using Apache NiFi
2. Loading and querying data with Apache Hive
3. Performing machine learning using PySpark
4. Storing and retrieving model metrics in HBase

This pipeline demonstrates a typical real-time or batch data flow often used in enterprise analytics platforms.

---

## 🧪 Methods Used

* Predictive Modeling
* Data Engineering
* Performance Optimization

---

## 🧰 Technologies

* Apache NiFi
* Hadoop HDFS
* Apache Hive
* PySpark
* Apache HBase

---

## 📄 Project Description

This project uses a modified **Heart Attack dataset** for predictive modeling. Due to resource limitations in the training environment, the dataset was shortened and pre-processed outside the system. The pipeline performs the following steps:

* **NiFi** handles data ingestion and flow control into HDFS
* **Hive** structures the data for queryable access
* **PySpark** processes the data and applies machine learning models
* **HBase** stores the output metrics for fast access

---

## 🧩 Challenges & Solutions

* **Issue:** NiFi's InvokeHTTP processor kept looping data

  * **Solution:** Configure it to run only once for initial ingestion
* **Issue:** PySpark session frequently crashed

  * **Solution:** Add manual garbage collection steps to prevent memory overload

---

## 📌 Project Requirements

* Data ingestion and cleaning
* Descriptive statistics
* Predictive modeling
* Reporting and pipeline visualization

---

## 🚀 Getting Started

1. Use the `NiFi_Flow.json` file to import the full NiFi processor group.
2. The dataset used is included in `data/medicaldata.json` (already pre-converted for convenience).
3. Pipeline screenshots and terminal outputs are included in the accompanying documentation for visual reference.

---

## 📢 Final Notes

This project is a practical demonstration of how data engineering and machine learning workflows can be automated in a distributed environment. The use of HBase for storing model metrics enables scalable and low-latency access for future applications.

---
