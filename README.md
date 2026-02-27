# Armous - Forensic Multi-Layer Similarity Engine

A Python-based forensic text analysis system that detects content reuse, plagiarism, and paraphrasing across articles and research documents using multiple similarity detection algorithms.

## Overview

Armous implements a sophisticated multi-layered approach to text similarity detection, combining lexical analysis, semantic analysis, and stylometric fingerprinting. The system is designed to identify various forms of content reuse ranging from verbatim reproduction to heavily paraphrased text.

## Architecture

The system follows a three-stage pipeline:

1. Data Ingestion and Normalization
2. Multi-layer Feature Extraction
3. Forensic Similarity Analysis

## Core Components

### app.py

Main Streamlit web interface providing the forensic analysis engine with two primary workflows.

Key Functions:
- Radar Plot Visualization: Displays similarity metrics across six dimensions (Jaccard 3-gram, Semantic Similarity, Paragraph Ratio, Entity Retention, MinHash, Stylometry)
- PDF Structural Analysis: Extracts and normalizes text from research PDFs with automatic section detection (Abstract, Introduction, Methodology, Results, Discussion, Conclusion)
- Multi-Input Support: Accepts PDF files, JSON articles, HTML content, or live URLs
- Similarity Engine: Compares uploaded content against corpus using multiple algorithms

Technical Details:
- Uses pdfplumber for structural PDF parsing with regex-based section detection
- Implements BeautifulSoup for HTML scraping and cleaning
- TF-IDF vectorization for semantic similarity computation
- Multi-metric scoring with weighted confidence calculation (20% Jaccard 3-gram, 20% Semantic, 15% Paragraph Ratio, 15% Entity Retention, 15% MinHash, 15% SimHash normalization)

Reuse Classification Thresholds:
- Verbatim Reuse: Jaccard > 0.85 and Semantic > 0.85
- Light Paraphrase: Semantic > 0.75 and Preservation Ratio > 0.7
- Moderate Paraphrase: Semantic > 0.55
- Heavy Transformation: Semantic > 0.35
- Independent Content: Below all thresholds

Output Metrics (for each match):
1. Jaccard 3-gram: Lexical overlap of 3-word sequences
2. Jaccard 5-gram: Lexical overlap of 5-word sequences
3. MinHash Similarity: Probabilistic document fingerprint similarity
4. SimHash Distance: Hamming distance of semantic hash values
5. Semantic Similarity: Mean cosine similarity across paragraphs
6. Meaning Preservation Ratio: Proportion of paragraphs exceeding 0.5 similarity
7. Paragraph Ratio: Structural size alignment
8. Sentence Ratio: Fine-grained structural alignment
9. Entity Retention Ratio: Named entity overlap using Jaccard
10. Stylometric Distance: Euclidean distance of POS tag distributions

### preprocess.py

Batch preprocessing pipeline for converting raw corpus documents into forensic fingerprints.

Input Handling:
- Reads JSON documents from raw_content/ folder
- Expects clean_text field containing normalized document text

Processing Steps:
1. Tokenization and linguistic annotation using spaCy (en_core_web_sm)
2. Paragraph segmentation by double newlines
3. Sentence boundary detection
4. Alphabetic token extraction (lowercased)

Feature Extraction:

Lexical Features:
- 3-gram and 5-gram extraction from tokens with deduplication

Entity Recognition:
- Named entity extraction and lowercasing
- Stores both ordered list and unique set

Structural Features:
- Paragraph count calculation
- Sentence count calculation

Syntactic Features:
- Dependency relation signatures using Counter frequency distribution

Stylometric Analysis:
- Average sentence length computation
- Sentence length variance calculation
- POS tag distribution normalization (frequency/total tokens)

Fingerprinting Methods:
- SimHash: Integer hash value for semantic document similarity
- MinHash: 128-permutation signatures for Jaccard similarity estimation

Output Format:
Stores processed documents as JSON in processed_content/ folder with filename pattern {id}-processed.json containing all extracted features and signatures. Includes NumPy type conversion for JSON serialization.

### upload_preprocess.py

Real-time preprocessing module for documents uploaded through the Streamlit interface.

Function: process_uploaded_article(file, file_type)

Input Formats:
- JSON: Expects clean_text and optional id fields
- HTML: Extracts text after removing script/style tags

Processing:
Identical feature extraction pipeline to preprocess.py using spaCy and statistical methods.

Output Storage:
Saves processed document to uploads/ folder with filename derived from SHA256 content hash, ensuring deduplication of identical documents.

Return Value:
Returns complete processed feature dictionary for immediate similarity analysis.

### clean_id.py

Data preparation utility for standardizing document identifiers in raw corpus.

Operation:
- Reads all JSON files from raw_content/ folder
- Renames files to sequential BBC-{index:02d}.json format (BBC-01, BBC-02, etc.)
- Updates internal id field within JSON to match filename
- Removes original files after successful migration
- Maintains alphabetical sort for consistent numbering

Purpose:
Ensures corpus documents follow standardized naming convention for repeatable processing and easy reference.

## Data Flow

### Input Processing

Raw Documents Flow:
raw_content/ (raw JSON/HTML) → clean_id.py (standardization) → preprocess.py (feature extraction) → processed_content/ (forensic fingerprints)

User Uploads Flow:
Streamlit Interface → upload_preprocess.py (real-time feature extraction) → uploads/ (temporary storage) → app.py (comparison against corpus)

### Similarity Analysis

1. Load user-submitted document via upload_preprocess.py
2. Iterate through all processed_content/ files
3. Compute six independent similarity metrics
4. Calculate weighted confidence score
5. Classify reuse type based on thresholds
6. Sort by confidence and display top 4 matches
7. Render radar chart visualization

## Feature Specifications

Lexical Analysis:
- N-gram extraction for detecting word-sequence overlap
- Jaccard similarity: intersection/union of token sets
- Handles deduplication and set operations

Semantic Analysis:
- TF-IDF vectorization with maximum 2000 features
- Cosine similarity matrix computation
- Paragraph-level semantic matching
- Mean maximum similarity aggregation

Fingerprinting:
- SimHash: 64-bit semantic hash for quick comparison
- MinHash: 128-permutation Jaccard approximation
- Enables efficient large-scale similarity searches

Entity Analysis:
- spaCy named entity recognition
- Case-normalized entity sets
- Jaccard similarity on entity overlap

Stylometry:
- POS tag distribution analysis
- Sentence length statistics (mean and variance)
- Euclidean distance on normalized POS vectors
- Captures writing style characteristics

## Dependencies

Core Libraries:
- streamlit: Web interface framework
- spacy: Natural language processing with en_core_web_sm model
- scikit-learn: TF-IDF vectorization and cosine similarity
- datasketch: MinHash probabilistic fingerprinting
- simhash: Semantic hashing
- pdfplumber: Research PDF structural parsing
- beautifulsoup4: HTML parsing and cleaning
- numpy: Numerical computations

## Directory Structure
