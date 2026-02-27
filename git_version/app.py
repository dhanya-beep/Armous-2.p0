import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
import pdfplumber 
import re 
from bs4 import BeautifulSoup
from io import StringIO
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from upload_preprocess import process_uploaded_article

PROCESSED_FOLDER = "processed_content"

st.set_page_config(layout="wide")

st.markdown("""
<style>
body { background-color: #0e0e0e; color: white; }
.stApp { background-color: #0e0e0e; }
h1,h2,h3 { color: #ff0055; }
</style>
""", unsafe_allow_html=True)

st.title("Forensic Multi-Layer Similarity Engine")

tab1, tab2 = st.tabs(["Upload Paper","Upload Article"])

# -----------------------------
# Radar Plot
# -----------------------------
def radar_plot(match):

    labels = ["J3","Semantic","Paragraph",
              "Entity","MinHash","Stylometry"]

    styl_sim = 1 - min(match[11], 1)

    values = [
        match[2],
        match[6],
        match[8],
        match[10],
        match[4],
        styl_sim
    ]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values += values[:1]
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(5,5),
                           subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title("Forensic Similarity Signature")

    return fig


with tab1:

    st.subheader("Research-Grade PDF Structural Analysis")

    pdf_file = st.file_uploader("Upload Research PDF", type=["pdf"])

    if pdf_file:

        st.info("Performing structural PDF parsing...")

        try:
            full_text = ""

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

            # Normalize whitespace
            full_text = re.sub(r'\s+', ' ', full_text)

            # -------------------------
            # SECTION DETECTION
            # -------------------------

            section_pattern = r'(?=(\b(Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion|Related Work)\b))'
            sections = re.split(section_pattern, full_text)

            structured_sections = {}
            current_section = "Unknown"

            for part in sections:
                if part is None:
                    continue
                part = part.strip()
                if not part:
                    continue

                header_match = re.match(r'^(Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion|Related Work)', part)
                if header_match:
                    current_section = header_match.group(1)
                    structured_sections[current_section] = ""
                else:
                    structured_sections.setdefault(current_section, "")
                    structured_sections[current_section] += " " + part

            # -------------------------
            # REMOVE REFERENCES
            # -------------------------

            for ref_label in ["References", "Bibliography", "Acknowledgment", "Acknowledgements"]:
                if ref_label in structured_sections:
                    del structured_sections[ref_label]

            # Extract abstract if exists
            abstract_text = structured_sections.get("Abstract", "")

            # Reconstruct cleaned document (exclude references)
            cleaned_text = " ".join(structured_sections.values())

            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)

            pdf_data = {
                "id": pdf_file.name,
                "url": None,
                "title": pdf_file.name,
                "category": "research_pdf",
                "publication_date": None,
                "collected_at": datetime.utcnow().isoformat(),
                "raw_html": None,
                "clean_text": cleaned_text,
                "word_count": word_count,
                "char_count": char_count,
                "language": "en",
                "sections_detected": list(structured_sections.keys()),
                "abstract_extracted": bool(abstract_text)
            }

            raw_json_string = json.dumps(pdf_data, indent=4)

            st.success("Structural parsing completed")
            st.write(f"Sections detected: {pdf_data['sections_detected']}")
            st.write(f"Word count (cleaned): {word_count}")

            st.download_button(
                label="Download Structured PDF JSON",
                data=raw_json_string,
                file_name="structured_pdf_extracted.json",
                mime="application/json"
            )

            json_buffer = StringIO(raw_json_string)
            uploaded = process_uploaded_article(json_buffer, "json")

            st.success("PDF fingerprinted and ready for similarity analysis")

        except Exception as e:
            st.error(f"PDF parsing failed: {e}")

with tab2:

    st.subheader("Input Article")

    article = st.file_uploader("Upload Article (JSON or HTML)",
                               type=["json","html"])

    url_input = st.text_input("OR Enter Article URL")

    uploaded = None
    raw_json_string = None

    # -----------------------------
    # URL INGESTION PIPELINE
    # -----------------------------
    if url_input:

        st.info("Fetching URL...")
        try:
            response = requests.get(url_input, timeout=10)

            st.success("URL fetched successfully")

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts/styles
            for tag in soup(["script","style","noscript"]):
                tag.decompose()

            text = soup.get_text(separator="\n")

            st.info("HTML parsed and cleaned")

            # Metadata extraction
            title = soup.title.string.strip() if soup.title else "Unknown"

            publication_date = None
            meta_date = soup.find("meta", property="article:published_time")
            if meta_date and meta_date.get("content"):
                publication_date = meta_date["content"]

            url_data = {
                "id": url_input,
                "url": url_input,
                "title": title,
                "category": None,
                "publication_date": publication_date,
                "collected_at": datetime.utcnow().isoformat(),
                "raw_html": response.text,
                "clean_text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
                "language": "en"
            }

            raw_json_string = json.dumps(url_data, indent=4)

            st.success("Raw structured content created")

            st.download_button(
                label="Download Raw JSON",
                data=raw_json_string,
                file_name="live_ingested_article.json",
                mime="application/json"
            )

            json_buffer = StringIO(raw_json_string)
            uploaded = process_uploaded_article(json_buffer, "json")

            st.success("Article processed and fingerprinted")

        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    # -----------------------------
    # FILE UPLOAD PIPELINE
    # -----------------------------
    elif article:

        file_type = "json" if article.name.endswith(".json") else "html"
        uploaded = process_uploaded_article(article, file_type)
        st.success("Uploaded article processed")

    # -----------------------------
    # SIMILARITY ENGINE
    # -----------------------------
    if uploaded:

        def jaccard(a,b):
            return len(set(a)&set(b))/len(set(a)|set(b)) if a and b else 0

        def minhash_similarity(a,b):
            matches = sum(1 for i,j in zip(a,b) if i==j)
            return matches/len(a)

        def simhash_distance(a,b):
            return bin(a^b).count("1")

        def classify_reuse(confidence, j3, semantic, preservation_ratio):
            if j3 > 0.85 and semantic > 0.85:
                return "Verbatim reuse"
            if semantic > 0.75 and preservation_ratio > 0.7:
                return "Light paraphrase"
            if semantic > 0.55:
                return "Moderate paraphrase"
            if semantic > 0.35:
                return "Heavy transformation"
            return "Independent content"

        scores = []

        st.info("Running multi-layer similarity analysis...")

        for file in os.listdir(PROCESSED_FOLDER):

            with open(os.path.join(PROCESSED_FOLDER,file)) as f:
                base = json.load(f)

            j3 = jaccard(uploaded["lexical"]["ngrams_3"],
                         base["lexical"]["ngrams_3"])

            j5 = jaccard(uploaded["lexical"]["ngrams_5"],
                         base["lexical"]["ngrams_5"])

            min_sim = minhash_similarity(
                uploaded["fingerprints"]["minhash_signature"],
                base["fingerprints"]["minhash_signature"]
            )

            sim_dist = simhash_distance(
                uploaded["fingerprints"]["simhash"],
                base["fingerprints"]["simhash"]
            )

            combined = uploaded["paragraphs"] + base["paragraphs"]
            vec = TfidfVectorizer(max_features=2000)
            tfidf = vec.fit_transform(combined)

            up = tfidf[:len(uploaded["paragraphs"])]
            bp = tfidf[len(uploaded["paragraphs"]):]

            matrix = cosine_similarity(up, bp)

            semantic = float(np.mean(np.max(matrix,axis=1)))

            preservation_ratio = float(
                np.sum(np.max(matrix,axis=1)>0.5)
                / len(uploaded["paragraphs"])
            )

            para_ratio = min(uploaded["structure"]["paragraph_count"],
                             base["structure"]["paragraph_count"]) / \
                         max(uploaded["structure"]["paragraph_count"],
                             base["structure"]["paragraph_count"])

            sent_ratio = min(uploaded["structure"]["sentence_count"],
                             base["structure"]["sentence_count"]) / \
                         max(uploaded["structure"]["sentence_count"],
                             base["structure"]["sentence_count"])

            entity_ratio = jaccard(
                uploaded["entities"]["entity_set"],
                base["entities"]["entity_set"]
            )

            up_pos = uploaded["stylometry"]["pos_distribution"]
            base_pos = base["stylometry"]["pos_distribution"]

            all_tags = set(up_pos.keys()).union(set(base_pos.keys()))
            up_vec = np.array([up_pos.get(tag, 0.0)
                               for tag in all_tags])
            base_vec = np.array([base_pos.get(tag, 0.0)
                                 for tag in all_tags])

            styl_dist = np.linalg.norm(up_vec - base_vec)

            confidence = (
                0.2*j3 +
                0.2*semantic +
                0.15*para_ratio +
                0.15*entity_ratio +
                0.15*min_sim +
                0.15*(1 - sim_dist/64)
            )

            reuse_label = classify_reuse(
                confidence, j3,
                semantic, preservation_ratio
            )

            scores.append((
                file,confidence,j3,j5,min_sim,
                sim_dist,semantic,preservation_ratio,
                para_ratio,sent_ratio,entity_ratio,
                styl_dist,reuse_label
            ))

        scores.sort(key=lambda x:x[1],reverse=True)
        top_matches = scores[:4]

        st.success("Similarity analysis completed")

        for idx, match in enumerate(top_matches):

            st.markdown("---")
            st.subheader(f"Match {idx+1}: {match[0]}")
            st.write(f"Confidence Score: {round(match[1],3)}")
            st.write(f"Classification: {match[12]}")

            col1, col2 = st.columns([1,1])

            with col1:
                st.markdown("### Metrics")
                st.write(f"""
                Jaccard 3-gram: {round(match[2],3)}  
                Jaccard 5-gram: {round(match[3],3)}  
                MinHash similarity: {round(match[4],3)}  
                SimHash distance: {match[5]}  
                Semantic similarity: {round(match[6],3)}  
                Meaning preservation ratio: {round(match[7],3)}  
                Paragraph ratio: {round(match[8],3)}  
                Sentence ratio: {round(match[9],3)}  
                Entity retention ratio: {round(match[10],3)}  
                Stylometric distance: {round(match[11],3)}
                """)

            with col2:
                st.pyplot(radar_plot(match))
