"""Read plain text out of uploaded .txt, .md, .pdf and .docx files."""
import io


def read_upload(storage):
    name = (storage.filename or "").lower()
    data = storage.read()
    if name.endswith((".txt", ".md", ".text")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    raise ValueError("Unsupported file type. Use .txt, .pdf or .docx.")
