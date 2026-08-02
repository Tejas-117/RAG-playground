## Supported Documents

## 1. Document & Text Formats 

* Portable Document Format (.pdf): The most uploaded format; requires layout-aware splitting.
* Microsoft Word (.docx, .doc): Standard for business, articles, and resumes.
* Rich Text & Plain Text (.txt, .rtf): Low-overhead, easy to chunk.
* E-books (.epub, .mobi): Great for users testing RAG on full books. 

## 2. Developer & Technical Content

* Markdown (.md): Vital for documentation and notes (e.g., Obsidian vaults).


## 3. Tabular & Structured Data

* Spreadsheets (.xlsx, .xls): Financial models, user lists, and metrics.
* Delimited Files (.csv, .tsv): Clean structured data that requires row-by-row or sub-table chunking strategies.
* Data Interchange (.json): Raw data exports from APIs or apps. 

## 4. Presentations & Visuals (The Playground Differentiator)

* PowerPoint (.pptx, .ppt): Crucial for corporate strategy decks; requires slide-by-slide processing.
* Images (.jpg, .png, .webp): Infographics, receipts, or memes that need either an OCR or a Vision LLM reader.

------------------------------
## Essential Architecture Pillars for a Playground
Since you don't know what users will upload, your pipeline needs these automated safety nets:

* Automated OCR Routing: If a PDF or image has zero selectable text, automatically route it through an optical character recognition engine (like Tesseract or a cloud Vision API).
* Table Extraction Strategy: CSVs and Excel files break standard token chunking. Implement a strategy that converts rows into pseudo-sentences or JSON objects before embedding.
* Size & Token Budgeting: Users will try to upload 100MB files. Implement a background worker (like Celery or Redis Queue) to handle heavy parsing without timing out the playground UI. 

Would you like recommendations on managed parsing APIs (like Unstructured.io, LlamaParse, or Azure Document Intelligence) that can handle all of these formats with a single endpoint, or are you planning to build the open-source parsing stack yourself?

------------------------------
### The supported formats as of now
* .pdf, .epub, .mobi -> pymupdf
* .docx -> python-docx
* .pptx -> python-pptx
* .txt -> inbuilt function
* .md -> inbuilt function

### The below formats will be supported later on
* csv
* xlsx
* Configuration Files (.yaml, .toml, .ini): For users testing system prompt/log RAGs.
* Web Formats (.html, .xml): Saved web pages or blog article downloads.
