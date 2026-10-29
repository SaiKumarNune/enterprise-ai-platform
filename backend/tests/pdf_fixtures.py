from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_with_pages(*texts: str) -> bytes:
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 24 Tf 72 700 Td ({safe}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = stream
        font = DictionaryObject()
        f1 = DictionaryObject()
        f1[NameObject("/Type")] = NameObject("/Font")
        f1[NameObject("/Subtype")] = NameObject("/Type1")
        f1[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font[NameObject("/F1")] = f1
        page[NameObject("/Resources")][NameObject("/Font")] = font
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
