import json
import logging
import time
from pathlib import Path

from .config import get_api_key
from .types import TitleResult

logger = logging.getLogger(__name__)


class GeminiTitleLLM:
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None, model_name: str = DEFAULT_MODEL):
        try:
            import google.generativeai as genai

            self.genai = genai
            # Try multiple sources for API key - check both old and new env var names
            key = (
                api_key
                or get_api_key("GEMINI_API_KEY")
                or get_api_key("GOOGLE_API_KEY")
            )
            if not key:
                logger.warning(
                    "API key not found. Checked: GEMINI_API_KEY, GOOGLE_API_KEY in environment variables, "
                    ".env files in project/tools/home directories"
                )
                self.model = None
            else:
                genai.configure(api_key=key)
                # Try to create the model and handle potential model name issues
                try:
                    self.model = genai.GenerativeModel(model_name)
                except (ValueError, RuntimeError, OSError) as e:
                    logger.warning(f"Failed to create model '{model_name}': {e}")
                    # Try fallback model names
                    fallback_models = [
                        "gemini-2.5-flash",
                        "gemini-2.0-flash",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro",
                        "gemini-pro",
                    ]
                    for fallback in fallback_models:
                        try:
                            logger.info(f"Trying fallback model: {fallback}")
                            self.model = genai.GenerativeModel(fallback)
                            logger.info(f"Successfully using model: {fallback}")
                            break
                        except (ValueError, RuntimeError, OSError) as fallback_error:
                            logger.warning(
                                f"Fallback model '{fallback}' failed: {fallback_error}"
                            )
                            continue
                    else:
                        logger.error("All model attempts failed")
                        self.model = None
        except ImportError:
            self.genai = None
            logger.error("google-generativeai package not installed")

    def extract_title(self, pdf_path: Path) -> TitleResult:
        if not self.genai or not self.model:
            return TitleResult(
                None, 0.0, "llm", "Gemini API not available or model not initialized"
            )

        uploaded_file = None
        try:
            logger.info(f"Uploading {pdf_path.name} to Gemini...")
            uploaded_file = self.genai.upload_file(
                path=str(pdf_path), mime_type="application/pdf"
            )

            # Wait for processing state if needed (usually fast for small PDFs)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_file = self.genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                return TitleResult(None, 0.0, "llm", "Gemini file processing failed")

            prompt = """
            Extract the main title of this document.
            Ignore headers/footers/generic text like "Draft" unless part of the title.
            Return ONLY a JSON object with this structure:
            {
                "title": "The exact title string",
                "confidence": 0.0 to 1.0,
                "reason": "Why you chose this title"
            }
            """

            response = self.model.generate_content(
                [uploaded_file, prompt],
                generation_config={"response_mime_type": "application/json"},
            )

            text = response.text
            try:
                data = json.loads(text)
                title = data.get("title")
                conf = float(data.get("confidence", 0.0))
                details = data.get("reason", "")

                if title:
                    return TitleResult(title, conf, "llm", f"Gemini: {details}")
                else:
                    return TitleResult(
                        None, 0.0, "llm", f"Gemini found no title: {details}"
                    )

            except json.JSONDecodeError:
                return TitleResult(
                    None,
                    0.0,
                    "llm",
                    f"Gemini response parsing failed: {text[:100]}",
                )

        except (ValueError, KeyError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Gemini LLM error: {e}")
            return TitleResult(None, 0.0, "llm", f"Gemini error: {e}")
        finally:
            if uploaded_file:
                try:
                    self.genai.delete_file(uploaded_file.name)
                except (RuntimeError, ValueError, OSError) as e:
                    logger.debug(
                        "Failed to delete uploaded file %s: %s", uploaded_file.name, e
                    )
