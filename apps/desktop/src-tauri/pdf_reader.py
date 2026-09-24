import sys
import os
import json
import io

def read_pdf(filepath):
    """Read a PDF file and return its text content."""
    filepath = filepath.strip().strip('"').strip("'")
    if not os.path.exists(filepath):
        return json.dumps({"error": f"File not found: {filepath}"})
    
    try:
        import PyPDF2
    except ImportError:
        return json.dumps({
            "error": "PyPDF2 is not installed; declare PyPDF2 in the generated tool dependencies"
        })
    
    try:
        reader = PyPDF2.PdfReader(filepath)
        pages = []
        image_only_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                pages.append({"page": i + 1, "text": text})
            else:
                image_only_pages.append(i + 1)
        
        result = {
            "total_pages": len(reader.pages),
            "content": pages,
            "image_only_pages": image_only_pages,
            "ocr_required": bool(image_only_pages)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Accept JSON input via stdin (for generated tool) or command-line args
    if sys.stdin.isatty():
        # Command-line mode
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Usage: pdf_reader.py <filepath>"}))
            sys.exit(1)
        filepath = sys.argv[1]
    else:
        # JSON stdin mode (generated tool)
        try:
            input_data = json.loads(sys.stdin.read())
            filepath = input_data.get("filepath", "")
        except json.JSONDecodeError:
            print(json.dumps({"error": "Invalid JSON input"}))
            sys.exit(1)
    
    if not filepath:
        print(json.dumps({"error": "No filepath provided"}))
        sys.exit(1)
    
    print(read_pdf(filepath))
