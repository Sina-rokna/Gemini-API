import os
import json
import logging
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DebitCardExtractor:
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("API Key not found. Please set GOOGLE_API_KEY in .env file.")
            raise ValueError("Google API Key is missing.")

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.request_delay_s = float(os.getenv("REQUEST_DELAY_SECONDS", "6"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.mask_sensitive = False

        genai.configure(api_key=self.api_key)
        self.generation_config = {
            "temperature": 0.1, 
            "response_mime_type": "application/json",
        }

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        logger.info("DebitCardExtractor initialized successfully.")

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"File not found: {image_path}")
            
        try:
            img = Image.open(path)
            logger.info(f"Image loaded successfully: {image_path}")
            return img
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise

    def extract_info(self, image_path: str) -> Dict[str, Any]:
        img = self._load_image(image_path)
        
        prompt = """
        You are an expert OCR system specialized in Persian financial documents.
        Analyze this image of a bank debit card. 
        
        Extract the following information:
        1. Bank Name (in Persian)
        2. Card Number (16 digits) - Format as a clean string without spaces
        3. Cardholder Name (in Persian)
        4. Expiration Date (usually in YY/MM format)
        5. Sheba Number (IBAN), if visible (starts with IR)

        Use the following JSON schema:
        {
            "bank_name": "string",
            "card_number": "string",
            "cardholder_name": "string",
            "expiration_date": "string",
            "sheba_number": "string"
        }
        If a field is not visible, set it to null.
        """

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("Sending image to Google Gemini API for processing... (attempt %s/%s)", attempt, self.max_retries)
                response = self.model.generate_content([prompt, img])
                raw_text = (response.text or "").strip()
                raw_text = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw_text, flags=re.IGNORECASE | re.MULTILINE).strip()
                if not raw_text.startswith("{"):
                    start = raw_text.find("{")
                    end = raw_text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        raw_text = raw_text[start:end + 1]

                data = json.loads(raw_text)
                logger.info("Data extracted successfully from API response.")
                return data

            except json.JSONDecodeError as e:
                last_err = e
                logger.error("Failed to parse JSON response. Raw text: %s", getattr(response, "text", ""))
            except Exception as e:
                last_err = e
                logger.error(f"API call failed: {e}")
                if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
                    logger.error(f"Prompt Feedback: {e.response.prompt_feedback}")
            sleep_s = self.request_delay_s * attempt
            logger.info("Retrying after %.1fs...", sleep_s)
            time.sleep(sleep_s)

        raise RuntimeError(f"Failed to extract info from {image_path} after {self.max_retries} attempts") from last_err

    def save_to_excel(self, data: Any, output_path: str = "extracted_cards.xlsx", append: bool = False):
        try:
            if isinstance(data, pd.DataFrame):
                df_new = data
            elif isinstance(data, list):
                df_new = pd.DataFrame(data)
            else:
                df_new = pd.DataFrame([data])

            if append and os.path.exists(output_path):
                df_existing = pd.read_excel(output_path)
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_final = df_new

            df_final.to_excel(output_path, index=False)
            logger.info("Data successfully saved to %s", output_path)

        except Exception as e:
            logger.error("Failed to save to Excel: %s", e)
            raise

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _normalize_text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value)
    s = s.translate(_PERSIAN_DIGITS)
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه").replace("ة", "ه")
    s = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _digits_only(value: Any) -> Optional[str]:
    s = _normalize_text(value)
    if not s:
        return None
    d = re.sub(r"\D+", "", s)
    return d or None


def _normalize_bank(value: Any) -> Optional[str]:
    s = _normalize_text(value)
    if not s:
        return None
    s = s.replace("بانك", "بانک")
    s = re.sub(r"\bبانک\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


def _normalize_sheba(value: Any) -> Optional[str]:
    s = _normalize_text(value)
    if not s:
        return None
    s = s.upper().replace(" ", "")
    return s or None


def _parse_expiry(value: Any) -> Optional[tuple[int, int]]:
    s = _normalize_text(value)
    if not s:
        return None
    nums = re.findall(r"\d+", s)
    if len(nums) < 2:
        return None
    a, b = nums[0], nums[1]
    ai, bi = int(a), int(b)
    if ai > 12 or len(a) == 4:
        year, month = ai, bi
    elif bi > 12 or len(b) == 4:
        month, year = ai, bi
    else:
        year, month = ai, bi
    yy = year % 100
    mm = month
    if not (1 <= mm <= 12):
        return None
    return yy, mm


def _levenshtein_ratio(a: Optional[str], b: Optional[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a, b = str(a), str(b)
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ca = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[m]
    return 1.0 - (dist / max(n, m))


def _token_f1(a: Optional[str], b: Optional[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ta = [t for t in str(a).split() if t]
    tb = [t for t in str(b).split() if t]
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    p = inter / len(sa)
    r = inter / len(sb)
    return 2 * p * r / (p + r)


def _mask_number(s: Optional[str], keep_last: int = 4) -> Optional[str]:
    if not s:
        return None
    s = str(s)
    if len(s) <= keep_last:
        return "*" * len(s)
    return "*" * (len(s) - keep_last) + s[-keep_last:]


class DebitCardEvaluator:
    FIELDS = ["bank_name", "card_number", "cardholder_name", "expiration_date", "sheba_number"]

    def __init__(self, mask_sensitive_in_report: bool = True):
        self.mask_sensitive_in_report = False

    def evaluate(self, expected_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        expected_df = expected_df.copy()
        pred_df = pred_df.copy()

        if "image_file" in expected_df.columns and "image_file" in pred_df.columns:
            merged = expected_df.merge(pred_df, on="image_file", how="outer", suffixes=("_expected", "_pred"))
        else:
            expected_df["_row"] = range(len(expected_df))
            pred_df["_row"] = range(len(pred_df))
            merged = expected_df.merge(pred_df, on="_row", how="outer", suffixes=("_expected", "_pred"))

        rows = []
        for _, r in merged.iterrows():
            e_bank = _normalize_bank(r.get("bank_name" + "_expected"))
            p_bank = _normalize_bank(r.get("bank_name" + "_pred"))
            e_card = _digits_only(r.get("card_number" + "_expected"))
            p_card = _digits_only(r.get("card_number" + "_pred"))
            e_name = _normalize_text(r.get("cardholder_name" + "_expected"))
            p_name = _normalize_text(r.get("cardholder_name" + "_pred"))
            raw_e_name = r.get("cardholder_name_expected")
            raw_p_name = r.get("cardholder_name_pred")
            if isinstance(raw_e_name, float) and pd.isna(raw_e_name):
                raw_e_name = None
            if isinstance(raw_p_name, float) and pd.isna(raw_p_name):
                raw_p_name = None

            e_exp = _parse_expiry(r.get("expiration_date" + "_expected"))
            p_exp = _parse_expiry(r.get("expiration_date" + "_pred"))
            if e_exp is None and p_exp is None:
                exp_match = True
            elif e_exp is None or p_exp is None:
                exp_match = False
            else:
                exp_match = (e_exp == p_exp) or (e_exp == (p_exp[1] % 100, p_exp[0]))

            e_sheba = _normalize_sheba(r.get("sheba_number" + "_expected"))
            p_sheba = _normalize_sheba(r.get("sheba_number" + "_pred"))
            bank_match = (e_bank == p_bank) if (e_bank or p_bank) else True
            bank_contains = False
            if e_bank and p_bank:
                bank_contains = (e_bank in p_bank) or (p_bank in e_bank)
                if bank_contains:
                    bank_match = True

            card_match = (e_card == p_card) if (e_card or p_card) else True
            sheba_match = (e_sheba == p_sheba) if (e_sheba or p_sheba) else True
            name_exact = (e_name == p_name) if (e_name or p_name) else True
            name_f1 = _token_f1(e_name, p_name)
            name_lev = _levenshtein_ratio(e_name, p_name)
            if not name_exact and name_f1 >= 0.9:
                name_exact = True

            required_ok = all([bank_match, card_match, name_exact, exp_match])
            all_fields_ok = required_ok and sheba_match

            def show(v: Optional[str], kind: str) -> Optional[str]:
                if not self.mask_sensitive_in_report:
                    return v
                if kind in {"card", "sheba"}:
                    return _mask_number(v)
                return v

            rows.append({
                "image_file": r.get("image_file") if "image_file" in merged.columns else None,
                "bank_expected": show(e_bank, "text"),
                "bank_pred": show(p_bank, "text"),
                "bank_match": bool(bank_match),
                "card_expected": show(e_card, "card"),
                "card_pred": show(p_card, "card"),
                "card_match": bool(card_match),
                "name_expected": raw_e_name,
                "name_pred": raw_p_name,
                "name_match": bool(name_exact),
                "name_token_f1": round(float(name_f1), 3),
                "name_lev_ratio": round(float(name_lev), 3),
                "exp_expected": r.get("expiration_date_expected"),
                "exp_pred": r.get("expiration_date_pred"),
                "exp_match": bool(exp_match),
                "sheba_expected": show(e_sheba, "sheba"),
                "sheba_pred": show(p_sheba, "sheba"),
                "sheba_match": bool(sheba_match),
                "required_fields_ok": bool(required_ok),
                "all_fields_ok": bool(all_fields_ok),
            })

        per_card = pd.DataFrame(rows)

        summary = pd.DataFrame([
            {
                "n_expected": int(len(expected_df)),
                "n_predicted": int(len(pred_df)),
                "bank_accuracy": float(per_card["bank_match"].mean()) if len(per_card) else 0.0,
                "card_accuracy": float(per_card["card_match"].mean()) if len(per_card) else 0.0,
                "name_accuracy": float(per_card["name_match"].mean()) if len(per_card) else 0.0,
                "expiry_accuracy": float(per_card["exp_match"].mean()) if len(per_card) else 0.0,
                "sheba_accuracy": float(per_card["sheba_match"].mean()) if len(per_card) else 0.0,
                "required_fields_exact_row_accuracy": float(per_card["required_fields_ok"].mean()) if len(per_card) else 0.0,
                "all_fields_exact_row_accuracy": float(per_card["all_fields_ok"].mean()) if len(per_card) else 0.0,
            }
        ])

        return summary, per_card


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def extract_directory(extractor: DebitCardExtractor, cards_dir: str) -> pd.DataFrame:
    p = Path(cards_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Cards directory not found: {cards_dir}")

    image_files = sorted([x for x in p.iterdir() if x.is_file() and _is_image(x)], key=lambda x: _natural_key(x.name))
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {cards_dir}")

    results = []
    for i, img_path in enumerate(image_files, start=1):
        logger.info("[%s/%s] Extracting: %s", i, len(image_files), img_path.name)
        data = extractor.extract_info(str(img_path))
        data = dict(data or {})
        data["image_file"] = img_path.name
        data["extracted_at"] = datetime.now().isoformat(timespec="seconds")
        results.append(data)
        if i < len(image_files):
            time.sleep(extractor.request_delay_s)

    return pd.DataFrame(results)


def write_report(summary: pd.DataFrame, per_card: pd.DataFrame, report_path: str):
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")
        per_card.to_excel(writer, index=False, sheet_name="per_card")


if __name__ == "__main__":
    try:
        cards_dir = os.getenv("CARDS_DIR", "cards")
        expected_xlsx = os.getenv("EXPECTED_XLSX", "expected_result.xlsx")
        extracted_xlsx = os.getenv("EXTRACTED_XLSX", "extracted_cards.xlsx")
        report_xlsx = os.getenv("REPORT_XLSX", "accuracy_report.xlsx")

        if os.path.exists(extracted_xlsx):
            df_pred = pd.read_excel(extracted_xlsx)
            if os.path.exists(expected_xlsx):
                expected_df = pd.read_excel(expected_xlsx)
                evaluator = DebitCardEvaluator(mask_sensitive_in_report=False)
                summary, per_card = evaluator.evaluate(expected_df, df_pred)
                write_report(summary, per_card, report_xlsx)
                print("\n=== Accuracy summary ===")
                print(summary.to_string(index=False))
                print(f"\nDetailed report saved to: {report_xlsx}")
            else:
                print(f"\nNote: '{expected_xlsx}' not found; skipping accuracy evaluation.")
            print(f"\nUsing existing extraction file: {extracted_xlsx}")
        else:
            extractor = DebitCardExtractor()
            if os.path.isdir(cards_dir):
                df_pred = extract_directory(extractor, cards_dir)
                extractor.save_to_excel(df_pred, extracted_xlsx, append=False)
            else:
                target_image = os.getenv("TARGET_IMAGE", "card1.jpg")
                if not os.path.exists(target_image):
                    raise FileNotFoundError(
                        f"Neither cards directory '{cards_dir}' exists, nor TARGET_IMAGE '{target_image}' was found."
                    )
                data = extractor.extract_info(target_image)
                data = dict(data or {})
                data["image_file"] = os.path.basename(target_image)
                data["extracted_at"] = datetime.now().isoformat(timespec="seconds")
                df_pred = pd.DataFrame([data])
                extractor.save_to_excel(df_pred, extracted_xlsx, append=False)

            print(f"\nExtraction saved to: {extracted_xlsx}")
            print(f"\nNote: '{extracted_xlsx}' was created in this run; accuracy evaluation is skipped.")

    except Exception as e:
        print(f"\nCritical Error: {e}")