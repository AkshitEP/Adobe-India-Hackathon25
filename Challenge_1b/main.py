import json
import os
from datetime import datetime
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model (pre-included in Docker image)
model = SentenceTransformer('./models/all-MiniLM-L6-v2')

def extract_outline(pdf_path):
    """Extract headings from a PDF using font size heuristics."""
    doc = fitz.open(pdf_path)
    outline = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    text = " ".join([span["text"] for span in line["spans"]]).strip()
                    font_size = max([span["size"] for span in line["spans"]], default=10)
                    if font_size > 12:  # Heuristic for headings
                        level = "H1" if font_size > 14 else "H2"
                        outline.append({"text": text, "level": level, "page": page_num})
    doc.close()
    return outline

def extract_sections(doc, outline):
    """Extract section content between headings."""
    page_texts = [page.get_text() for page in doc]
    full_text = "\n".join(page_texts)
    sections = []
    heading_positions = []
    
    for heading in outline:
        page_num = heading["page"] - 1
        page_text = page_texts[page_num]
        start_idx = page_text.find(heading["text"])
        if start_idx != -1:
            global_start = sum(len(page_texts[i]) + 1 for i in range(page_num)) + start_idx
            heading_positions.append((global_start, heading))
    
    heading_positions.sort(key=lambda x: x[0])
    
    for i in range(len(heading_positions)):
        start = heading_positions[i][0] + len(heading_positions[i][1]["text"])
        end = heading_positions[i + 1][0] if i < len(heading_positions) - 1 else len(full_text)
        content = full_text[start:end].strip()
        sections.append({
            "heading": heading_positions[i][1],
            "content": content,
            "document": doc.name
        })
    return sections

def build_top_level_sections(sections):
    """Organize sections into a hierarchy with H1 as top-level and H2 as sub-sections."""
    top_level_sections = []
    current_top = None
    
    for section in sections:
        level = section["heading"]["level"]
        if level == "H1":
            if current_top:
                top_level_sections.append(current_top)
            current_top = {
                "title": section["heading"]["text"],
                "page": section["heading"]["page"],
                "content": section["content"],
                "sub_sections": [],
                "document": section["document"]
            }
        elif current_top and level == "H2":
            current_top["sub_sections"].append({
                "title": section["heading"]["text"],
                "page": section["heading"]["page"],
                "content": section["content"]
            })
    if current_top:
        top_level_sections.append(current_top)
    return top_level_sections

def tokenize(text):
    """Tokenize text for BM25 scoring."""
    return text.lower().split()

# Load input JSON
with open('./input/challenge1b_input.json', 'r') as f:
    input_data = json.load(f)

# Extract document paths, persona, and job-to-be-done
documents = [os.path.join('./input/PDFs', doc['filename']) for doc in input_data['documents']]
persona = input_data['persona']['role']
job = input_data['job_to_be_done']['task']
query = persona + " " + job  # "Travel Planner Plan a trip of 4 days for a group of 10 college friends."

# Process all documents
all_sections = []
for doc_path in documents:
    doc = fitz.open(doc_path)
    outline = extract_outline(doc_path)
    sections = extract_sections(doc, outline)
    top_level_sections = build_top_level_sections(sections)
    all_sections.extend(top_level_sections)
    doc.close()

# Compute relevance scores
section_texts = [s['title'] + " " + s['content'][:200] for s in all_sections]
query_embedding = model.encode([query])[0]
section_embeddings = model.encode(section_texts)
similarities = [np.dot(query_embedding, sec_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(sec_emb)) 
                for sec_emb in section_embeddings]

bm25 = BM25Okapi([tokenize(s) for s in section_texts])
bm25_scores = bm25.get_scores(tokenize(query))

query_keywords = set(tokenize(query))
keyword_matches = [len(set(tokenize(s['title'])).intersection(query_keywords)) for s in all_sections]

# Combine scores for ranking
relevance_scores = [bm25_scores[i] + similarities[i] + keyword_matches[i] for i in range(len(all_sections))]

# Rank sections by relevance
ranked_sections = sorted(zip(all_sections, relevance_scores), key=lambda x: x[1], reverse=True)

# Generate structured output
output = {
    "metadata": {
        "input_documents": [doc['filename'] for doc in input_data['documents']],
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.now().isoformat()
    },
    "extracted_sections": []
}

for rank, (section, _) in enumerate(ranked_sections[:10], start=1):  # Limit to top 10 sections
    extracted_section = {
        "document": os.path.basename(section['document']),
        "page_number": section['page'],
        "section_title": section['title'],
        "importance_rank": rank,
        "sub_sections": [
            {
                "sub_section_title": sub['title'],
                "refined_text": sub['content'][:200]  # Truncate for brevity
            } for sub in section['sub_sections']
        ]
    }
    output["extracted_sections"].append(extracted_section)

# Write output JSON
with open('./output/challenge1b_output.json', 'w') as f:
    json.dump(output, f, indent=2)