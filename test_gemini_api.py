"""
Gemini API setup checker for the CV Analyzer.

Run:
    python test_gemini_api.py

It:
1. Checks that GEMINI_API_KEY exists.
2. Imports google-genai.
3. Makes one tiny request to Gemini 3.5 Flash-Lite.
4. Reports whether authentication/model access works.

The API key itself is never printed.
"""

import os
import sys


MODEL = os.getenv(
    "GEMINI_SCREENING_MODEL",
    "gemini-3.5-flash-lite",
).strip()


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    print("=" * 60)
    print("CV Analyzer - Gemini API Setup Check")
    print("=" * 60)

    # --------------------------------------------------------
    # Key presence
    # --------------------------------------------------------
    if not api_key:
        print("❌ GEMINI_API_KEY is NOT set.")
        print()
        print("PowerShell:")
        print('$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"')
        return 1

    print("✅ GEMINI_API_KEY is set.")
    print(f"   Key length: {len(api_key)} characters")
    print("   Key value: [hidden]")

    # --------------------------------------------------------
    # SDK
    # --------------------------------------------------------
    try:
        from google import genai
    except ImportError:
        print("❌ google-genai is not installed.")
        print()
        print("Run:")
        print("pip install -U google-genai")
        return 1

    print("✅ google-genai SDK is installed.")
    print(f"✅ Model configured: {MODEL}")

    # --------------------------------------------------------
    # API call
    # --------------------------------------------------------
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL,
            contents=(
                "Reply with exactly these two words: "
                "GEMINI OK"
            ),
        )

        text = str(getattr(response, "text", "") or "").strip()

        if not text:
            print("❌ Gemini returned an empty response.")
            return 1

        print("✅ Gemini API request succeeded.")
        print(f"✅ Response received: {text[:120]}")

        print()
        print("🎉 API setup is working.")
        print()
        print("You can now run:")
        print("streamlit run app.py")
        return 0

    except Exception as exc:
        message = str(exc)

        print("❌ Gemini API request failed.")
        print()
        print(message)

        print()
        if "401" in message or "403" in message:
            print("➡️ This usually indicates an API-key/authentication or access problem.")
        elif "429" in message or "quota" in message.lower():
            print("➡️ The key works, but the quota/rate limit was reached.")
        elif "503" in message or "unavailable" in message.lower():
            print("➡️ Authentication likely worked, but the model was temporarily unavailable/high demand.")
            print("   Try again later or choose another available free-tier model.")
        elif "404" in message or "not found" in message.lower():
            print("➡️ The configured model name is unavailable for this API project.")
        else:
            print("➡️ Review the error above. Your API key was not printed.")

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
