"""
Shelf Product Identification System with Reference Products
Uses object detection + OCR + reference matching
"""

import cv2
import numpy as np
import requests
import json
import base64
from typing import List, Dict, Tuple
import unicodedata

class ShelfProductIdentifier:
    def __init__(self, gemini_api_key: str):
        """
        Initialize the shelf product identifier.
        
        Args:
            gemini_api_key: Google Gemini API key for detection and OCR
        """
        self.gemini_api_key = gemini_api_key
        self.reference_products = {}  # Store reference products
        
    def add_reference_product(self, name: str, price: float, **metadata):
        """
        Add a reference product to the database.
        
        Args:
            name: Product name
            price: Product price
            **metadata: Additional product information (SKU, category, etc.)
        """
        normalized_name = self._normalize_text(name)
        self.reference_products[normalized_name] = {
            'original_name': name,
            'price': price,
            'metadata': metadata
        }
        
    def load_reference_products_from_list(self, products: List[Dict]):
        """
        Load multiple reference products from a list.
        
        Args:
            products: List of dicts with 'name', 'price', and optional metadata
        """
        for product in products:
            name = product.get('name')
            price = product.get('price')
            metadata = {k: v for k, v in product.items() if k not in ['name', 'price']}
            self.add_reference_product(name, price, **metadata)
            
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for consistent matching."""
        return (text.lower()
                .replace(" ", "")
                .replace("&", "and")
                .replace("+", "plus")
                .replace("-", ""))
    
    @staticmethod
    def _remove_currency_symbols(text: str) -> str:
        """Remove currency symbols from text."""
        return ''.join(
            ch for ch in text 
            if unicodedata.category(ch) != 'Sc'
        )
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def detect_shelf_labels(self, image: np.ndarray) -> List[Dict]:
        """
        Detect shelf labels/products using Gemini API with optimized prompts.
        
        Args:
            image: Input shelf image as numpy array
            
        Returns:
            List of detection dicts with 'bbox' and 'confidence'
        """
        h, w = image.shape[:2]
        base64_image = self._encode_image_to_base64(image)
        
        MODEL_ID = "gemini-2.5-flash"
        API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={self.gemini_api_key}"
        
        prompt = """You are an expert retail product detection system. Analyze this shelf image and detect ALL individual products.

CRITICAL INSTRUCTIONS:
1. DETECT INDIVIDUAL PRODUCTS: Each product bottle, box, or package should get its own bounding box
2. PRODUCT BOXES/CARDS: If there are product boxes or cards displayed above bottles (like hair oil boxes), detect each one separately
3. BOTTLES/CONTAINERS: Detect each individual bottle or container on the shelf
4. INCLUDE EVERYTHING:
   - Product boxes/packaging displayed on top shelves
   - Individual bottles on middle and lower shelves
   - Even partially visible products at edges
   - Products at different angles or orientations

5. BOUNDING BOX RULES:
   - Each box should tightly wrap around ONE product
   - Include the full product from top to bottom (cap to base for bottles)
   - Include the brand name and visible text on the product
   - Don't merge multiple products into one box
   - Don't include excessive empty space

6. BE THOROUGH: Count carefully and detect EVERY single product visible
   - Top shelf products (boxes/cards)
   - Middle shelf products (bottles)
   - Bottom shelf products (bottles)

Return ONLY the bounding boxes in the exact format: box_2d (normalized 0-1000 coordinates: [y1, x1, y2, x2])."""
        output_prompt = "\n\nOutput format: Return ONLY a JSON array with box_2d coordinates. No additional text or explanation."
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt + output_prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}
                ]
            }],
            "generationConfig": {"temperature": 0}
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
        detections = json.loads(raw_text.strip().removeprefix("```json").removesuffix("```").strip())
        
        # Convert normalized coordinates to pixel coordinates
        processed_detections = []
        for det in detections:
            y1, x1, y2, x2 = det["box_2d"]
            y1, x1, y2, x2 = y1/1000*h, x1/1000*w, y2/1000*h, x2/1000*w
            
            # Ensure correct ordering
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            processed_detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': 1.0
            })
            
        return processed_detections
    
    def extract_text_from_crop(self, crop: np.ndarray) -> List[Dict]:
        """
        Extract product names and prices from cropped product using OCR with optimized prompts.
        
        Args:
            crop: Cropped product image
            
        Returns:
            List of dicts with 'item_name' and 'price'
        """
        base64_crop = self._encode_image_to_base64(crop)
        
        MODEL_ID = "gemini-2.5-flash-lite"
        API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={self.gemini_api_key}"
        
        prompt_text = """You are an expert at reading product labels and extracting information. Carefully examine this product image.

TASK: Extract the complete product identification information.

EXTRACTION RULES:

1. BRAND NAME (REQUIRED):
   - Main brand (e.g., "Vatika", "Parachute", "Dabur", "Janet", "Kesh King")
   - Always include the brand as the first part of the product name

2. PRODUCT LINE/TYPE (REQUIRED):
   - What type of product is it? (e.g., "Hair Oil", "Hair Fall Control", "Herbal Hair Oil", "Natural Hair Oil")
   - Include specific product line names (e.g., "Ayurveda", "Naturals", "Gold")

3. VARIANT/FORMULATION:
   - Key ingredients or benefits (e.g., "Coconut", "Almond", "Hibiscus", "Nourish & Protect")
   - Special formulations (e.g., "7 Oils in One", "Enriched with Henna", "Kalonji")
   - Treatment type (e.g., "Hair Fall Control", "Anti-Dandruff", "Damage Repair")

4. SIZE/VOLUME (if visible):
   - Include measurements (e.g., "200mL", "100ml", "150ml", "300ml")
   - Pack size if mentioned (e.g., "3 pack", "Twin Pack")

5. PRICE (if visible):
   - Extract exact price with currency symbol
   - If no price visible on product itself, leave as empty string ""

6. SPECIAL NOTES:
   - For hair oils, hair care products: ALWAYS include "Hair Oil" or product type
   - Read text on bottles, boxes, labels carefully
   - Include text in ALL languages if present (English, Hindi, etc.)
   - Maintain proper spacing and capitalization

EXAMPLE OUTPUTS:
✓ "Vatika Enriched Coconut Hair Oil 200ml"
✓ "Parachute Advansed Almond Enriched Coconut Hair Oil"
✓ "Dabur Amla Hair Oil 275ml"
✓ "Kesh King Ayurvedic Scalp and Hair Oil"
✓ "Janet Ayurveda Hair Fall Control 100ml"

FORMAT: Return as JSON array with 'item_name' (complete name) and 'price' (string, use "" if not visible).
Be precise and extract the FULL product identity, not generic descriptions."""
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_crop}},
                    {"text": prompt_text}
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_name": {"type": "string"},
                            "price": {"type": "string"}
                        },
                        "required": ["item_name", "price"]
                    }
                },
                "temperature": 0
            }
        }
        
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        items = json.loads(data['candidates'][0]['content']['parts'][0]['text'])
        
        if not items:
            items = [{"item_name": "miscellaneous", "price": ""}]
            
        return items
    
    def match_with_reference(self, extracted_items: List[Dict]) -> List[Dict]:
        """
        Match extracted items with reference products.
        
        Args:
            extracted_items: List of extracted products with names and prices
            
        Returns:
            List of matched products with validation status
        """
        matched_results = []
        
        for item in extracted_items:
            normalized_name = self._normalize_text(item['item_name'])
            
            # Look up in reference database
            reference = self.reference_products.get(normalized_name)
            
            if not reference:
                result = {
                    'item_name': item['item_name'],
                    'extracted_price': item['price'],
                    'status': 'PRODUCT_NOT_FOUND',
                    'reference_price': None,
                    'match': False
                }
            else:
                try:
                    extracted_price = float(self._remove_currency_symbols(item['price']))
                    reference_price = float(reference['price'])
                    
                    if extracted_price == reference_price:
                        status = 'PRICE_MATCH'
                        match = True
                    else:
                        status = 'PRICE_MISMATCH'
                        match = False
                except ValueError:
                    status = 'INVALID_PRICE'
                    match = False
                    reference_price = reference['price']
                    
                result = {
                    'item_name': item['item_name'],
                    'extracted_price': item['price'],
                    'reference_price': reference.get('price'),
                    'status': status,
                    'match': match,
                    'reference_name': reference.get('original_name'),
                    'metadata': reference.get('metadata', {})
                }
                
            matched_results.append(result)
            
        return matched_results
    
    def process_shelf_image(self, image_path: str, output_path: str = None) -> Dict:
        """
        Complete pipeline: detect, extract, match, and visualize.
        
        Args:
            image_path: Path to shelf image
            output_path: Optional path to save annotated image
            
        Returns:
            Dict with detection results and matched products
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        print("Detecting shelf labels...")
        detections = self.detect_shelf_labels(image)
        print(f"Found {len(detections)} shelf labels")
        
        all_results = []
        annotated_image = image.copy()
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Add offset for better OCR
            offset = 10
            x1 = max(0, x1 - offset)
            y1 = max(0, y1 - offset)
            x2 = min(image.shape[1], x2 + offset)
            y2 = min(image.shape[0], y2 + offset)
            
            # Crop label
            crop = image[y1:y2, x1:x2]
            
            print(f"Processing label {idx+1}/{len(detections)}...")
            extracted_items = self.extract_text_from_crop(crop)
            matched_items = self.match_with_reference(extracted_items)
            
            # Determine overall status for this detection
            if not matched_items:
                status = 'NO_TEXT_FOUND'
                color = (128, 128, 128)  # Gray
            elif all(item['status'] == 'PRICE_MATCH' for item in matched_items):
                status = 'ALL_MATCH'
                color = (0, 255, 0)  # Green
            elif any(item['status'] == 'PRICE_MISMATCH' for item in matched_items):
                status = 'MISMATCH'
                color = (0, 0, 255)  # Red
            elif all(item['status'] == 'PRODUCT_NOT_FOUND' for item in matched_items):
                status = 'NOT_FOUND'
                color = (0, 255, 255)  # Yellow
            else:
                status = 'MIXED'
                color = (255, 0, 255)  # Magenta
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            all_results.append({
                'bbox': [x1, y1, x2, y2],
                'status': status,
                'items': matched_items
            })
        
        # Save annotated image if requested
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to {output_path}")
        
        return {
            'total_labels': len(detections),
            'results': all_results,
            'annotated_image': annotated_image
        }


# Example usage
if __name__ == "__main__":
    # Initialize system
    GEMINI_API_KEY = "your-gemini-api-key-here"
    identifier = ShelfProductIdentifier(GEMINI_API_KEY)
    
    # Add reference products for hair care/hair oil shelf
    reference_products = [
        {"name": "Vatika Enriched Coconut Hair Oil 200ml", "price": 250.00, "sku": "VAT001"},
        {"name": "Vatika Hibiscus Hair Oil", "price": 180.00, "sku": "VAT002"},
        {"name": "Parachute Advansed Almond Enriched Coconut Hair Oil", "price": 120.00, "sku": "PAR001"},
        {"name": "Dabur Amla Hair Oil 275ml", "price": 150.00, "sku": "DAB001"},
        {"name": "Kesh King Ayurvedic Scalp and Hair Oil", "price": 280.00, "sku": "KES001"},
        {"name": "Janet Ayurveda Hair Fall Control", "price": 220.00, "sku": "JAN001"},
        {"name": "Kashvi Herbal Hair Oil", "price": 95.00, "sku": "KAS001"},
        {"name": "Navratna Cool Hair Oil", "price": 85.00, "sku": "NAV001"},
        {"name": "Bajaj Almond Drops Hair Oil", "price": 110.00, "sku": "BAJ001"},
        {"name": "Sliming Herbal Hair Treatment", "price": 160.00, "sku": "SLI001"},
        {"name": "Genesha Hair Oil", "price": 140.00, "sku": "GEN001"}
    ]
    
    identifier.load_reference_products_from_list(reference_products)
    
    # Process shelf image
    results = identifier.process_shelf_image(
        image_path="shelf_image.jpg",
        output_path="shelf_annotated.jpg"
    )
    
    # Print results
    print("\n" + "="*60)
    print("SHELF PRODUCT IDENTIFICATION RESULTS")
    print("="*60)
    print(f"Total products detected: {results['total_labels']}")
    print("="*60)
    
    matches = 0
    mismatches = 0
    not_found = 0
    
    for idx, result in enumerate(results['results']):
        print(f"\n[Product {idx+1}] Status: {result['status']}")
        print("-" * 60)
        for item in result['items']:
            print(f"  Product: {item['item_name']}")
            print(f"  Extracted Price: {item['extracted_price']}")
            if item.get('reference_price'):
                print(f"  Reference Price: {item['reference_price']}")
                print(f"  Reference Name: {item.get('reference_name', 'N/A')}")
            print(f"  Validation: {item['status']}")
            
            if item['status'] == 'PRICE_MATCH':
                matches += 1
            elif item['status'] == 'PRICE_MISMATCH':
                mismatches += 1
            elif item['status'] == 'PRODUCT_NOT_FOUND':
                not_found += 1
        print("-" * 60)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Price Matches: {matches}")
    print(f"✗ Price Mismatches: {mismatches}")
    print(f"? Products Not Found: {not_found}")
    print("="*60)