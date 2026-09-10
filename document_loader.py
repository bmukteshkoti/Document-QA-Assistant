from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(file):
    file.seek(0)

    pdf_bytes = file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_txt(file):
    file.seek(0)

    text = file.read().decode("utf-8")

    return text.strip()