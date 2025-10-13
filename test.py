"""
Food Recognition Testing System for Google Cloud Vision API
Tests accuracy on RAW household food ingredients from Canadian/African households
Auto-selects top images with batch review capability
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, asdict
from google.cloud import vision
from dotenv import load_dotenv
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()


@dataclass
class TestImage:
    """Represents a test image with metadata"""
    expected_ingredients: List[str]
    url: str
    source: str
    image_type: str
    downloaded_path: str = ""
    identified_labels: List[str] = None
    confidence_scores: List[float] = None
    is_correct: bool = False
    preview_url: str = ""


class ReviewServer(BaseHTTPRequestHandler):
    """HTTP server for batch image review"""
    marked_for_removal = set()
    images_data = None
    review_complete = False
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self._generate_html().encode())
        elif self.path == '/complete':
            ReviewServer.review_complete = True
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()
        data = json.loads(post_data)
        
        if data['action'] == 'toggle':
            idx = data['index']
            if idx in ReviewServer.marked_for_removal:
                ReviewServer.marked_for_removal.remove(idx)
            else:
                ReviewServer.marked_for_removal.add(idx)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'marked': list(ReviewServer.marked_for_removal)}).encode())
    
    def _generate_html(self):
        images = ReviewServer.images_data
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Review Auto-Selected Images</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: #1a1a2e;
                    color: white;
                    padding: 20px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    position: sticky;
                    top: 0;
                    background: #1a1a2e;
                    padding: 20px 0;
                    z-index: 100;
                    border-bottom: 3px solid #16213e;
                }
                .header h1 { font-size: 2.5em; margin-bottom: 10px; color: #00d4ff; }
                .header p { font-size: 1.1em; opacity: 0.8; margin-bottom: 15px; }
                .stats {
                    display: inline-block;
                    background: #16213e;
                    padding: 10px 30px;
                    border-radius: 25px;
                    margin: 10px;
                }
                .stats span { font-weight: bold; color: #00d4ff; }
                .complete-btn {
                    padding: 15px 40px;
                    font-size: 18px;
                    background: #00d4ff;
                    color: #1a1a2e;
                    border: none;
                    border-radius: 30px;
                    cursor: pointer;
                    font-weight: bold;
                    transition: all 0.3s;
                    margin-top: 15px;
                }
                .complete-btn:hover {
                    background: #00a8cc;
                    transform: scale(1.05);
                }
                .section {
                    margin-bottom: 50px;
                }
                .section-title {
                    font-size: 1.8em;
                    color: #00d4ff;
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #16213e;
                    border-radius: 10px;
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 20px;
                }
                .image-card {
                    background: #16213e;
                    border-radius: 15px;
                    overflow: hidden;
                    cursor: pointer;
                    transition: all 0.3s;
                    border: 3px solid transparent;
                    position: relative;
                }
                .image-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
                }
                .image-card.marked {
                    border-color: #ff4757;
                    opacity: 0.5;
                }
                .image-card.marked::after {
                    content: '✗ REMOVE';
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #ff4757;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 10px;
                    font-weight: bold;
                    font-size: 1.2em;
                }
                .image-card img {
                    width: 100%;
                    height: 200px;
                    object-fit: cover;
                    display: block;
                }
                .card-info {
                    padding: 12px;
                }
                .ingredient-name {
                    font-weight: bold;
                    color: #00d4ff;
                    margin-bottom: 5px;
                    font-size: 0.95em;
                }
                .source-badge {
                    display: inline-block;
                    padding: 3px 10px;
                    background: #0f3460;
                    border-radius: 12px;
                    font-size: 0.75em;
                    text-transform: uppercase;
                }
                .instructions {
                    background: #16213e;
                    padding: 20px;
                    border-radius: 15px;
                    margin-bottom: 30px;
                    border-left: 5px solid #00d4ff;
                }
                .instructions h3 {
                    color: #00d4ff;
                    margin-bottom: 10px;
                }
                .instructions li {
                    margin: 8px 0;
                    margin-left: 20px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🍽️ Review Auto-Selected Images</h1>
                <p>Click on images to mark for removal (bad quality, wrong item, etc.)</p>
                <div class="stats">
                    <span id="total">0</span> Total Images
                </div>
                <div class="stats">
                    <span id="marked">0</span> Marked for Removal
                </div>
                <div class="stats">
                    <span id="keeping">0</span> Will Keep
                </div>
                <br>
                <button class="complete-btn" onclick="completeReview()">✓ Complete Review & Continue</button>
            </div>
            
            <div class="instructions">
                <h3>📋 Quick Instructions:</h3>
                <ul>
                    <li><strong>Click any image</strong> to toggle removal (red border = will be removed)</li>
                    <li><strong>Look for:</strong> Wrong items, poor quality, cooked food instead of raw</li>
                    <li><strong>Keep scrolling</strong> to review all images before completing</li>
                    <li><strong>When done:</strong> Click the blue button at top to continue testing</li>
                </ul>
            </div>
        """
        
        # Group by type
        single_imgs = [img for img in images if img['image_type'] == 'single']
        multi_imgs = [img for img in images if img['image_type'] == 'multi']
        
        for section_name, section_imgs in [("Single Ingredient Images", single_imgs), 
                                            ("Multi-Ingredient Images", multi_imgs)]:
            html += f"""
            <div class="section">
                <div class="section-title">{section_name} ({len(section_imgs)} images)</div>
                <div class="grid">
            """
            
            for idx, img in enumerate(section_imgs):
                global_idx = images.index(img)
                ingredients = ', '.join(img['expected_ingredients'])
                
                html += f"""
                <div class="image-card" id="card-{global_idx}" onclick="toggleMark({global_idx})">
                    <img src="{img['preview_url']}" alt="{ingredients}" loading="lazy">
                    <div class="card-info">
                        <div class="ingredient-name">{ingredients}</div>
                        <span class="source-badge">{img['source']}</span>
                    </div>
                </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        html += f"""
            <script>
                let markedSet = new Set();
                const totalImages = {len(images)};
                
                function updateStats() {{
                    document.getElementById('total').textContent = totalImages;
                    document.getElementById('marked').textContent = markedSet.size;
                    document.getElementById('keeping').textContent = totalImages - markedSet.size;
                }}
                
                function toggleMark(index) {{
                    fetch('/', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{action: 'toggle', index: index}})
                    }})
                    .then(res => res.json())
                    .then(data => {{
                        markedSet = new Set(data.marked);
                        document.getElementById('card-' + index).classList.toggle('marked');
                        updateStats();
                    }});
                }}
                
                function completeReview() {{
                    if (confirm(`You are keeping ${{totalImages - markedSet.size}} images. Continue to testing?`)) {{
                        fetch('/complete').then(() => {{
                            document.body.innerHTML = '<div style="text-align:center; padding:100px;"><h1>✓ Review Complete!</h1><p>Proceeding to testing phase...</p></div>';
                        }});
                    }}
                }}
                
                updateStats();
            </script>
        </body>
        </html>
        """
        return html


class ImageCurator:
    """Handles auto-selection of top-rated images with batch review"""
    
    UNSPLASH_API = "https://api.unsplash.com/search/photos"
    PEXELS_API = "https://api.pexels.com/v1/search"
    
    UNSPLASH_DELAY = 1.2  # 50 per hour = 72 seconds, but we'll batch
    PEXELS_DELAY = 0.5    # 200 per hour = 18 seconds
    
    def __init__(self):
        self.unsplash_key = os.getenv('UNSPLASH_ACCESS_KEY')
        self.pexels_key = os.getenv('PEXELS_API_KEY')
        
        if not self.unsplash_key or not self.pexels_key:
            raise ValueError("Missing API keys in .env file")
    
    def auto_select_image(self, query: str) -> Dict:
        """Auto-select best image from search results"""
        # Try Unsplash first (generally higher quality)
        result = self._search_unsplash(query)
        if result:
            return result
        
        time.sleep(0.5)
        
        # Fallback to Pexels
        result = self._search_pexels(query)
        return result
    
    def _search_unsplash(self, query: str) -> Dict:
        """Search Unsplash and return top result"""
        try:
            headers = {'Authorization': f'Client-ID {self.unsplash_key}'}
            params = {'query': query, 'per_page': 1, 'order_by': 'relevant'}
            
            response = requests.get(self.UNSPLASH_API, headers=headers, params=params, timeout=10)
            time.sleep(self.UNSPLASH_DELAY)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    img = results[0]
                    return {
                        'url': img['urls']['raw'],
                        'preview_url': img['urls']['small'],
                        'source': 'unsplash',
                        'description': img.get('description') or img.get('alt_description') or query
                    }
        except Exception as e:
            print(f"⚠️  Unsplash error for '{query}': {e}")
        
        return None
    
    def _search_pexels(self, query: str) -> Dict:
        """Search Pexels and return top result"""
        try:
            headers = {'Authorization': self.pexels_key}
            params = {'query': query, 'per_page': 1}
            
            response = requests.get(self.PEXELS_API, headers=headers, params=params, timeout=10)
            time.sleep(self.PEXELS_DELAY)
            
            if response.status_code == 200:
                data = response.json()
                photos = data.get('photos', [])
                if photos:
                    img = photos[0]
                    return {
                        'url': img['src']['original'],
                        'preview_url': img['src']['medium'],
                        'source': 'pexels',
                        'description': query
                    }
        except Exception as e:
            print(f"⚠️  Pexels error for '{query}': {e}")
        
        return None
    
    def batch_auto_select(self, ingredients_list: List[Tuple[List[str], str]]) -> List[TestImage]:
        """Auto-select images for all ingredients with progress"""
        test_images = []
        total = len(ingredients_list)
        
        print(f"\n🤖 Auto-selecting top images for {total} ingredients...")
        print("This will take ~2-3 minutes with rate limiting...\n")
        
        for idx, (ingredients, img_type) in enumerate(ingredients_list, 1):
            query_base = " ".join(ingredients)
            query = f"raw {query_base}" if img_type == 'single' else f"raw food {query_base}"
            
            print(f"[{idx}/{total}] Fetching: {query_base}...", end=' ')
            
            result = self.auto_select_image(query)
            
            if result:
                test_images.append(TestImage(
                    expected_ingredients=ingredients,
                    url=result['url'],
                    source=result['source'],
                    image_type=img_type,
                    preview_url=result['preview_url']
                ))
                print(f"✓ ({result['source']})")
            else:
                print("✗ No results")
        
        print(f"\n✅ Auto-selected {len(test_images)} images")
        return test_images
    
    def review_selections(self, test_images: List[TestImage]) -> List[TestImage]:
        """Present all selections for batch review"""
        print("\n📋 Opening browser for batch review...")
        print("Click images to mark for removal, then click 'Complete Review'")
        
        # Prepare data for review
        ReviewServer.marked_for_removal = set()
        ReviewServer.review_complete = False
        ReviewServer.images_data = [
            {
                'expected_ingredients': img.expected_ingredients,
                'preview_url': img.preview_url,
                'source': img.source,
                'image_type': img.image_type
            }
            for img in test_images
        ]
        
        # Start server
        server = HTTPServer(('localhost', 8000), ReviewServer)
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        # Open browser
        import webbrowser
        webbrowser.open('http://localhost:8000')
        
        # Wait for review completion
        while not ReviewServer.review_complete:
            time.sleep(0.2)
        
        server.shutdown()
        
        # Remove marked images
        marked_indices = ReviewServer.marked_for_removal
        filtered_images = [
            img for idx, img in enumerate(test_images) 
            if idx not in marked_indices
        ]
        
        removed_count = len(test_images) - len(filtered_images)
        print(f"\n✓ Review complete! Removed {removed_count} images")
        print(f"📊 Proceeding with {len(filtered_images)} images\n")
        
        return filtered_images


class VisionAPITester:
    """Tests Google Cloud Vision API food recognition"""
    
    def __init__(self, dataset_dir: Path = Path("test_dataset")):
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env")
        
        self.client = vision.ImageAnnotatorClient()
        self.dataset_dir = dataset_dir
        self.dataset_dir.mkdir(exist_ok=True)
        self.results_file = self.dataset_dir / "test_results.json"
    
    def download_image(self, url: str, filename: str) -> Path:
        """Download image to dataset directory"""
        filepath = self.dataset_dir / filename
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            return filepath
        
        raise Exception(f"Failed to download: {response.status_code}")
    
    def detect_labels(self, image_path: Path, max_results: int = 10) -> Tuple[List[str], List[float]]:
        """Detect labels using Google Cloud Vision API"""
        with open(image_path, 'rb') as f:
            content = f.read()
        
        image = vision.Image(content=content)
        response = self.client.label_detection(image=image, max_results=max_results)
        
        if response.error.message:
            raise Exception(f"API Error: {response.error.message}")
        
        labels = [label.description.lower() for label in response.label_annotations]
        scores = [label.score for label in response.label_annotations]
        
        return labels, scores
    
    def evaluate_detection(self, expected: List[str], detected: List[str]) -> bool:
        """Check if any expected ingredient is detected with fuzzy matching"""
        expected_lower = [e.lower() for e in expected]
        detected_lower = [d.lower() for d in detected]
        
        for exp in expected_lower:
            # Remove common plural/singular variations
            exp_base = exp.rstrip('s') if exp.endswith('s') and len(exp) > 3 else exp
            
            for det in detected_lower:
                det_base = det.rstrip('s') if det.endswith('s') and len(det) > 3 else det
                
                # Check for exact match, base match, or substring match
                if (exp == det or 
                    exp_base == det_base or 
                    exp in det or 
                    det in exp or
                    exp_base in det_base or
                    det_base in exp_base):
                    return True
        
        return False
    
    def run_test(self, test_images: List[TestImage]) -> Dict:
        """Run complete test suite"""
        results = {'single': [], 'multi': []}
        
        print("\n" + "="*60)
        print("TESTING PHASE")
        print("="*60)
        
        for idx, test_img in enumerate(test_images, 1):
            print(f"\n[{idx}/{len(test_images)}] Testing: {', '.join(test_img.expected_ingredients)}")
            
            try:
                filename = f"{test_img.image_type}_{idx}.jpg"
                filepath = self.download_image(test_img.url, filename)
                test_img.downloaded_path = str(filepath)
                
                labels, scores = self.detect_labels(filepath)
                test_img.identified_labels = labels
                test_img.confidence_scores = scores
                
                test_img.is_correct = self.evaluate_detection(
                    test_img.expected_ingredients, 
                    labels
                )
                
                print(f"  Detected: {', '.join(labels[:5])}")
                print(f"  Result: {'✅ PASS' if test_img.is_correct else '❌ FAIL'}")
                
                results[test_img.image_type].append(test_img)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        return results
    
    def generate_report(self, results: Dict) -> None:
        """Generate detailed test report"""
        stats = {}
        
        for img_type in ['single', 'multi']:
            total = len(results[img_type])
            correct = sum(1 for img in results[img_type] if img.is_correct)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            stats[img_type] = {
                'total': total,
                'correct': correct,
                'accuracy': round(accuracy, 2),
                'pass': accuracy >= 70
            }
        
        report = {
            'summary': {
                'single_ingredient': stats['single'],
                'multi_ingredient': stats['multi'],
                'overall_pass': stats['single']['pass'] and stats['multi']['pass']
            },
            'detailed_results': {
                'single': [asdict(img) for img in results['single']],
                'multi': [asdict(img) for img in results['multi']]
            }
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._print_summary(stats, report['summary']['overall_pass'])
    
    def _print_summary(self, stats: Dict, overall_pass: bool) -> None:
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        for label, key in [("Single-Ingredient", 'single'), ("Multi-Ingredient", 'multi')]:
            s = stats[key]
            print(f"\n{label} Images:")
            print(f"  Total: {s['total']}")
            print(f"  Correct: {s['correct']}")
            print(f"  Accuracy: {s['accuracy']:.2f}%")
            print(f"  Status: {'✅ PASS' if s['pass'] else '❌ FAIL'}")
        
        print(f"\nOverall Test: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        print(f"\nDetailed results saved to: {self.results_file}")


def main():
    """Main execution flow"""
    
    # RAW household ingredients (uncooked, packaged, raw form)
    SINGLE_INGREDIENTS = [
        ["rice"], ["salt"], ["sugar"], ["flour"], ["cooking oil"],
        ["onion"], ["garlic"], ["ginger"], ["tomato"], ["potato"],
        ["carrot"], ["banana"], ["apple"], ["orange"], ["eggs"],
        ["milk"], ["butter"], ["cheese"], ["chicken"], ["beef"],
        ["fish"], ["bread"], ["pasta"], ["beans"], ["lentils"],
        ["groundnut"], ["yam"], ["plantain"], ["cassava"], ["pepper"],
        ["cumin"], ["cinnamon"], ["turmeric"], ["curry powder"],
        ["bouillon cube"], ["soy sauce"], ["vinegar"], ["honey"],
        ["peanut butter"], ["jam"], ["coffee"], ["tea"], ["cornmeal"],
        ["oats"], ["corn"], ["peas"], ["spinach"], ["cabbage"],
        ["cucumber"], ["bell pepper"]
    ]
    
    MULTI_INGREDIENTS = [
        ["onion", "tomato", "pepper"],
        ["rice", "beans"],
        ["chicken", "vegetables"],
        ["eggs", "milk", "butter"],
        ["pasta", "tomato", "garlic"],
        ["potato", "carrot", "onion"],
        ["bread", "butter", "jam"],
        ["flour", "sugar", "eggs"],
        ["beef", "onion", "pepper"],
        ["fish", "lemon"],
        ["banana", "apple", "orange"],
        ["coffee", "milk", "sugar"],
        ["plantain", "groundnut"],
        ["yam", "pepper", "onion"],
        ["cassava", "palm oil"],
        ["spinach", "garlic", "onion"],
        ["lentils", "rice", "spices"],
        ["chicken", "curry", "rice"],
        ["beans", "plantain"],
        ["corn", "butter"],
        ["cabbage", "carrot"],
        ["peanut butter", "bread"],
        ["oats", "milk", "banana"],
        ["cheese", "bread", "tomato"],
        ["cucumber", "tomato", "onion"],
        ["bell pepper", "onion", "tomato"],
        ["ginger", "garlic", "onion"],
        ["soy sauce", "garlic", "ginger"],
        ["honey", "lemon"],
        ["turmeric", "ginger"],
        ["cinnamon", "sugar"],
        ["cumin", "coriander"],
        ["pasta", "cheese"],
        ["potato", "butter"],
        ["carrot", "ginger"],
        ["apple", "cinnamon"],
        ["orange", "honey"],
        ["fish", "tomato", "onion"],
        ["beef", "potato", "carrot"],
        ["chicken", "rice", "vegetables"],
        ["eggs", "cheese", "bread"],
        ["banana", "peanut butter"],
        ["tea", "milk", "sugar"],
        ["cornmeal", "milk"],
        ["beans", "rice", "plantain"],
        ["yam", "palm oil"],
        ["cassava", "groundnut"],
        ["plantain", "eggs"],
        ["spinach", "tomato"],
        ["lentils", "onion", "garlic"]
    ]
    
    print("🍽️  Food Recognition Test System")
    print("="*60)
    print("\n⚡ Fast Mode: Auto-select top images + Batch review")
    
    mode = input("\nSelect mode:\n1. Curate dataset (auto-select + review)\n2. Run test (test existing dataset)\n3. Full workflow\n\nChoice (1/2/3): ").strip()
    
    if mode in ['1', '3']:
        curator = ImageCurator()
        
        # Prepare ingredients list
        ingredients_list = [(ing, 'single') for ing in SINGLE_INGREDIENTS[:50]]
        ingredients_list += [(ing, 'multi') for ing in MULTI_INGREDIENTS[:50]]
        
        # Auto-select images
        test_images = curator.batch_auto_select(ingredients_list)
        
        if not test_images:
            print("❌ No images could be fetched. Check API keys and internet connection.")
            return
        
        # Batch review
        test_images = curator.review_selections(test_images)
        
        # Save dataset
        dataset_file = Path("curated_dataset.json")
        with open(dataset_file, 'w') as f:
            json.dump([asdict(img) for img in test_images], f, indent=2)
        
        print(f"✅ Dataset saved to {dataset_file}")
        
        if mode == '1':
            return
    
    if mode in ['2', '3']:
        tester = VisionAPITester()
        
        dataset_file = Path("curated_dataset.json")
        if not dataset_file.exists():
            print("❌ No curated dataset found. Run mode 1 first.")
            return
        
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            test_images = [TestImage(**item) for item in data]
        
        results = tester.run_test(test_images)
        tester.generate_report(results)


if __name__ == "__main__":
    main()