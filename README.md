# Persian Debit Card Extractor (Gemini)

This project extracts key fields from **Iranian (Persian) bank debit cards** using **Google Gemini** (LLM vision), then writes the results into an **Excel** file.  
Optionally, if you have a ground-truth Excel (`expected_result.xlsx`), it can generate an **accuracy report** comparing predictions vs expected values.

## Features

- Reads debit card images from a folder (batch) or a single image.
- Extracts:
  - `bank_name` (Persian)
  - `card_number` (16 digits, no spaces)
  - `cardholder_name` (Persian)
  - `expiration_date` (typically `YY/MM`)
  - `sheba_number` (IBAN starting with `IR`, if visible)
- Saves results to `extracted_cards.xlsx`
- If `extracted_cards.xlsx` already exists and `expected_result.xlsx` is present, generates `accuracy_report.xlsx` (summary + per-card details)

## Project Structure

- `card_extractor.py` — main extractor + evaluation logic
- `run_project.py` — installs dependencies and runs the extractor
- `check_models.py` — prints available Gemini models for your API key
- `requirements.txt` — dependencies
- `cards/` — (you create) put card images here for batch extraction
- `expected_result.xlsx` — (optional) your ground-truth labels for evaluation

## Requirements

- Python 3.9+ recommended
- A valid Google Gemini API key

## Setup

1) **Create a `.env` file** in the project root:

```env
GOOGLE_API_KEY=YOUR_KEY_HERE

# Optional:
GEMINI_MODEL=gemini-2.5-flash
CARDS_DIR=cards
TARGET_IMAGE=card1.jpg
EXTRACTED_XLSX=extracted_cards.xlsx
EXPECTED_XLSX=expected_result.xlsx
REPORT_XLSX=accuracy_report.xlsx
REQUEST_DELAY_SECONDS=6
MAX_RETRIES=3
