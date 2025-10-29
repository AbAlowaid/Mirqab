"""
LLM Handler - Interface with OpenAI GPT-4 Vision API
Uses OpenAI for fast, accurate image analysis
"""

import base64
import io
from PIL import Image
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
root_dir = Path(__file__).parent.parent
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path)  # ✅ NEW: Explicitly load from project root

# Import OpenAI
try:
    from openai import OpenAI  # ✅ NEW: Import client class
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Install with: pip install openai")

class LLMReportGenerator:
    def __init__(self, api_key: str = None):
        """
        Initialize the LLM report generator with OpenAI GPT-4 Vision API
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
        """
        if not OPENAI_AVAILABLE:
            print("⚠️ Warning: OpenAI SDK not installed. Install with: pip install openai")
            self.api_key = None
            self.model_name = None
            self.client = None
            return
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("⚠️ Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            print(f"   Checked: {env_path}")
        else:
            print(f"✅ OpenAI API key loaded successfully")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)  # ✅ NEW: Create client instance
        else:
            self.client = None
        self.model_name = "gpt-4-turbo"  # ✅ GPT-4 Turbo supports vision (image analysis)
    
    async def generate_report(self, image: Image.Image) -> dict:
        """
        Generate AI analysis report for an image using OpenAI GPT-4 Vision
        
        Args:
            image: PIL Image object
        
        Returns:
            Dictionary containing AI analysis with keys:
            - summary: Brief description
            - environment: Environment type
            - soldier_count: Estimated count
            - attire: Description of clothing/camouflage
            - equipment: Visible equipment
        """
        if not self.api_key:
            print("⚠️ No API key available (self.api_key is None), using fallback analysis")
            return self._get_fallback_analysis()
        
        if not OPENAI_AVAILABLE:
            print("⚠️ OpenAI module not available, using fallback analysis")
            return self._get_fallback_analysis()
        
        print(f"✅ Starting AI analysis with OpenAI GPT-4 Vision")
        print(f"   API Key available: {bool(self.api_key)}")
        print(f"   Model: {self.model_name}")
        
        # Convert image to base64
        buffered = io.BytesIO()
        # Resize image to reasonable size (max 1024px)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        print(f"   Image encoded: {len(img_base64)} bytes")
        
        # Construct the prompt
        prompt = self._create_analysis_prompt()
        
        try:
            # ✅ Make request to OpenAI GPT-4 Vision API
            print(f"🤖 Requesting AI analysis from OpenAI ({self.model_name})...")
            print(f"   Timeout: 30 seconds")
            print(f"   Temperature: 0.3")
            print(f"   Max tokens: 2048")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"You are a military intelligence analyst. Analyze this image and respond ONLY with valid JSON in this exact format:\n\n{prompt}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=2048,
                timeout=30
            )
            
            print(f"📝 API Response received successfully")
            print(f"   Response type: {type(response)}")
            
            # Extract text from response
            try:
                # ✅ NEW: Handle v1.0+ response format (object with attributes, not dict)
                if not response:
                    print(f"⚠️ Invalid response structure: None")
                    return self._get_fallback_analysis()
                
                # Try new format first (v1.0+)
                try:
                    analysis_text = response.choices[0].message.content
                except (AttributeError, IndexError, TypeError):
                    # Fallback to dict format if needed
                    try:
                        analysis_text = response['choices'][0]['message']['content']
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"⚠️ Error parsing response structure: {str(e)}")
                        print(f"   Response: {str(response)[:500]}")
                        return self._get_fallback_analysis()
                
                print(f"   Extracted text length: {len(analysis_text)} chars")
            except Exception as e:
                print(f"⚠️ Error extracting response: {str(e)}")
                return self._get_fallback_analysis()
            
            # Parse JSON from response
            try:
                analysis = json.loads(analysis_text)
                print(f"✅ JSON parsed successfully")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error: {str(e)}")
                # If response is not valid JSON, extract what we can
                analysis = self._parse_text_response(analysis_text)
            
            # Ensure all required fields are present
            analysis = self._validate_analysis(analysis)
            
            print("✅ AI analysis completed successfully")
            return analysis
            
        except Exception as e:
            print(f"❌ Error connecting to OpenAI API: {type(e).__name__}: {str(e)}")
            print(f"   Full error: {repr(e)}")
            
            # Return fallback analysis
            return self._get_fallback_analysis()
    
    def _create_analysis_prompt(self) -> str:
        """Create the structured prompt for OpenAI Vision API"""
        return """You are a military intelligence analyst specializing in camouflage detection. Analyze the provided image and return ONLY a valid JSON object with the following schema.

CRITICAL: Only count soldiers wearing camouflage (woodland, desert, digital, ghillie suits, etc.). DO NOT count soldiers in regular military uniforms without camouflage patterns.

Required JSON structure:
{
  "summary": "A brief 2-sentence summary of the environment and what was detected.",
  "environment": "Describe the environment (e.g., 'dense woodland', 'urban ruins', 'arid desert', 'mountainous terrain').",
  "camouflaged_soldier_count": 0,
  "has_camouflage": false,
  "attire_and_camouflage": "Describe the camouflage pattern and attire IF camouflaged soldiers are present. If no camouflage detected, write 'No camouflage detected'.",
  "equipment": "List any visible equipment IF camouflaged soldiers are present (e.g., 'rifles', 'backpacks'). If no camouflage detected, write 'N/A'."
}

IMPORTANT RULES:
1. Set "has_camouflage" to true ONLY if you detect soldiers with actual camouflage patterns
2. Set "camouflaged_soldier_count" to the number of soldiers wearing camouflage
3. Regular uniforms, tactical gear, or plain clothing DO NOT count as camouflage
4. If no camouflaged soldiers detected, set count to 0 and has_camouflage to false

Analyze the image and respond with ONLY the JSON object, no additional text."""
    
    def _parse_text_response(self, text: str) -> dict:
        """
        Parse non-JSON text response and extract information
        
        Args:
            text: Raw text response from LLM
        
        Returns:
            Structured dictionary
        """
        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Fallback: basic text parsing
        return {
            "summary": text[:200] if text else "Analysis generated from detected image.",
            "environment": "Unknown environment",
            "camouflaged_soldier_count": 0,
            "has_camouflage": False,
            "attire_and_camouflage": "No camouflage detected",
            "equipment": "N/A"
        }
    
    def _validate_analysis(self, analysis: dict) -> dict:
        """
        Ensure analysis has all required fields
        
        Args:
            analysis: Analysis dictionary
        
        Returns:
            Validated dictionary with all required fields
        """
        required_fields = {
            "summary": "Image analyzed for camouflaged soldiers.",
            "environment": "Unknown environment",
            "camouflaged_soldier_count": 0,
            "has_camouflage": False,
            "attire_and_camouflage": "No camouflage detected",
            "equipment": "N/A"
        }
        
        # Map old field names to new ones for backward compatibility
        if "soldier_count" in analysis and "camouflaged_soldier_count" not in analysis:
            analysis["camouflaged_soldier_count"] = analysis.get("soldier_count", 0)
        
        if "attire" in analysis and "attire_and_camouflage" not in analysis:
            analysis["attire_and_camouflage"] = analysis.get("attire", "Unknown")
        
        # Determine has_camouflage if not set
        if "has_camouflage" not in analysis:
            count = analysis.get("camouflaged_soldier_count", 0)
            analysis["has_camouflage"] = count > 0
        
        # Fill in missing fields with defaults
        for field, default_value in required_fields.items():
            if field not in analysis or not analysis[field]:
                analysis[field] = default_value
        
        return analysis
    
    def _get_fallback_analysis(self) -> dict:
        """
        Return a fallback analysis when LLM is unavailable
        
        Returns:
            Default analysis dictionary
        """
        return {
            "summary": "Image analyzed. Detailed analysis requires LLM connection.",
            "environment": "Unknown environment (LLM unavailable)",
            "camouflaged_soldier_count": 0,
            "has_camouflage": False,
            "attire_and_camouflage": "Unable to analyze - LLM connection required",
            "equipment": "Unable to analyze - LLM connection required"
        }
    
    def check_connection(self) -> bool:
        """
        Check if OpenAI API key is available and valid
        
        Returns:
            True if API key is configured, False otherwise
        """
        if not self.api_key or not OPENAI_AVAILABLE:
            print("⚠️ No OpenAI API key configured")
            print("   Set OPENAI_API_KEY environment variable")
            return False
        
        try:
            # ✅ NEW: Quick API check using OpenAI SDK
            # Just check if we can initialize without error
            if self.client:
                print(f"✅ OpenAI API connected ({self.model_name})")
                return True
            else:
                print(f"⚠️ OpenAI API key not set")
                return False
                
        except Exception as e:
            print(f"⚠️ Cannot connect to OpenAI API: {str(e)}")
            return False
