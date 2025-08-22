# Requirement Tracking AI (JSS)

This repository contains the source code for an AI model designed to automate the process of traceability link recovery between requirements and specifications in complex engineering systems, particularly within the defense industry. The project is associated with the research paper, "Automating Requirement-to-Specification Traceability Recovery in Complex Engineering Systems Using Domain-Adapted Pretrained Models: An Empirical Study on Defense Projects."

This project leverages Natural Language Processing (NLP) and pretrained language models to reduce the errors and inefficiencies inherent in manual traceability management, aiming to enhance quality assurance and configuration management in large-scale projects.

---

## Table of Contents
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Core Components](#core-components)

---

## Key Features

- **Automated Document Parsing**: Automatically extracts requirements and design specifications from various standard defense document formats (.docx, .hwp), including SSRS, SSS, SSDD, HRS, HDD, and SRS documents.
- **Domain-Specific Dictionary Builder**: Scrapes the DAPA (Defense Acquisition Program Administration) terminology website to build a custom dictionary for domain-specific morphological analysis.
- **Dataset Generation**: Creates a labeled sentence-pair dataset for model training by systematically pairing parsed requirements and specifications.
- **Model Training and Evaluation**: Trains and evaluates various pretrained language models (e.g., Ko-SBERT, Qwen) for the traceability classification task, allowing for comprehensive performance comparisons across different preprocessing strategies and model architectures.

---

## Project Architecture

The project follows a structured pipeline:

- **Dictionary Construction (Dict/)**
    - `KoreaDefenseDictCollector.py`: Scrapes defense terminology from the official DAPA website.
    - `DictionaryBuilder.py`: Processes the scraped terms into a user dictionary for the KoNLPy Komoran morphological analyzer, enabling accurate recognition of domain-specific nouns.

- **Document Parsing (Document_parsing_rules/, Word/)**
    - Defines parsing rules for various requirement and design documents (SSRS, SSS, etc.).
    - `DocxParser.py`: A utility to extract tables from between specified section titles in Word documents.

- **Data Generation (Output_generation/)**
    - Structures parsed data into `Output` objects and establishes parent-child relationships based on traceability links.
    - `DatasetGenerator.py`: Creates all possible requirement pairs from the structured data and assigns 'Relevant' (1) or 'Irrelevant' (0) labels based on ground-truth traceability tables to create the final training dataset.

- **Model Experimentation (ResearchQuestions/, Model/)**
    - `RQ1.py`: The main script that orchestrates a comprehensive comparison across different models (KR-SBERT, Ko-SRoBERTa, etc.), preprocessing strategies (full text, noun/verb only, etc.), and the number of additional neural network layers.
    - `Training.py`: Defines the PyTorch-based `CustomModel` class and contains the logic for training and evaluating the model with the specified data.

---

## Prerequisites

The following libraries are required to run this project:

- Python 3.8+
- PyTorch
- transformers
- pandas
- scikit-learn
- olefile
- pycryptodome
- python-docx
- pywin32
- konlpy
- datasets
- requests
- beautifulsoup4
- openpyxl

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/RequirementTrackingAI.git](https://github.com/your-username/RequirementTrackingAI.git)
    cd RequirementTrackingAI
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided. You will need to install the packages listed under Prerequisites manually.)*

3.  **Install Java (JDK):**
    A Java Development Kit (JDK) version 8 or higher is required for KoNLPy to function correctly.

---

## Usage

- **Prepare the Data:**
    - Place your dataset file (e.g., `Dataset.xlsx`) in the project's root directory. This file should contain `src_sentence`, `tar_sentence`, and `Label` columns.
    - Place the defense terminology file (`dapa_exact_terms.txt`) in the `Dict/` folder. If this file does not exist, you can generate it by running `Dict/KoreaDefenseDictCollector.py`.

- **Run the Experiment:**
    To execute the comprehensive comparison experiment, run the `RQ1.py` script from the `ResearchQuestions` directory.
    ```bash
    python RequirementTrackingAI/ResearchQuestions/RQ1.py
    ```
    This script will automatically train and evaluate all defined models, preprocessing strategies, and layer configurations. The final results will be saved to a file named `comprehensive_model_preprocessing_analysis.json`.

---

## Core Components

- **`ResearchQuestions/RQ1.py`**: This is the main executable file that drives the entire experiment. It systematically tests combinations of models, preprocessing methods, and layer depths to identify the optimal configuration for requirement traceability.
- **`Model/Training.py`**: Defines the `CustomModel`, which adds supplementary Feed-Forward layers on top of a base transformer model to perform the classification task. It also includes conditional logic for handling specific models like Qwen.
- **`Dict/DictionaryBuilder.py`**: A key preprocessing module that integrates a custom defense dictionary with the `Komoran` morphological analyzer from `KoNLPy`. This ensures that domain-specific compound nouns are correctly identified.
- **`Document_parsing_rules/`**: This directory contains modules with rules for extracting data from various defense document standards (HRS, SDD, etc.). These modules serve as the primary data input pipeline for the project.
