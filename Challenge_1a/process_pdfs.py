import fitz  # PyMuPDF
import os
import json
import pandas as pd
import joblib

# Load model and label encoder
clf = joblib.load('heading_classifier.pkl')
le = joblib.load('label_encoder.pkl')

def extract_lines_with_features(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    font_sizes = []
                    is_bold = False
                    is_italic = False
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                        if span["flags"] & 2**4:  # bold
                            is_bold = True
                        if span["flags"] & 2**1:  # italic
                            is_italic = True
                    if font_sizes:
                        font_size = max(set(font_sizes), key=font_sizes.count)
                    else:
                        font_size = 0
                    y_position = line["bbox"][1]  # top y-coordinate
                    lines.append({
                        "text": line_text.strip(),
                        "font_size": font_size,
                        "is_bold": is_bold,
                        "is_italic": is_italic,
                        "y_position": y_position,
                        "page_number": page_num + 1
                    })
    doc.close()
    return lines

def compute_features(line):
    text = line["text"]
    return {
        "font_size": line["font_size"],
        "is_bold": 1 if line["is_bold"] else 0,
        "is_italic": 1 if line["is_italic"] else 0,
        "y_position": line["y_position"],
        "text_length": len(text),
        "is_all_caps": 1 if text.isupper() else 0,
        "is_title_case": 1 if text.istitle() else 0,
        "page_number": line["page_number"]
    }


# Input and output directories in Docker
input_dir = '/app/input'
output_dir = '/app/output'

# Ensure output directory exists (for local testing, Docker handles this via mount)
os.makedirs(output_dir, exist_ok=True)

# Process all PDFs in input directory
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)
    lines = extract_lines_with_features(pdf_path)
    df = pd.DataFrame([compute_features(line) for line in lines])
    X = df.values
    y_pred = clf.predict(X)
    y_pred_labels = le.inverse_transform(y_pred)
    
    title = None
    outline = []
    for line, label in zip(lines, y_pred_labels):
        if label == "title" and title is None:  # Take first title
            title = line["text"]
        elif label in ["H1", "H2", "H3"]:
            outline.append({
                "level": label,
                "text": line["text"],
                "page": line["page_number"]
            })
    
    # Fallback: Use first H1 if no title found
    if title is None and outline:
        title = outline[0]["text"]
    
    # Create JSON output
    output_data = {
        "title": title if title else "Untitled",
        "outline": outline
    }
    
    # Save JSON
    output_file = os.path.splitext(pdf_file)[0] + '.json'
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)