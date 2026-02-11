#!/usr/bin/env python3
"""
Letter Extractor for Vintage Book Scans (with Lasso Masking)
Extracts individual characters from JPG scans with pixel-perfect cropping
"""

from PIL import Image, ImageDraw
import numpy as np
import os
from collections import defaultdict

class LetterExtractor:
    def __init__(self, output_dir="extracted_chars"):
        self.output_dir = output_dir
        self.char_counter = defaultdict(int)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/preview", exist_ok=True)

    def preprocess_image(self, image_path, threshold=180):
        """Load and preprocess the scanned page"""
        img = Image.open(image_path)
        gray = img.convert('L')
        binary = gray.point(lambda x: 0 if x < threshold else 255, '1')
        return img, gray, binary

    def find_character_regions_with_masks(self, binary_img, original_img, 
                                          min_width=5, min_height=10,
                                          max_width=100, max_height=100):
        """Find characters and extract their exact pixel masks (lasso-style)"""
        arr = np.array(binary_img)
        arr = 1 - arr  # Invert: text=1, background=0

        height, width = arr.shape
        visited = np.zeros_like(arr, dtype=bool)
        regions = []

        def flood_fill_with_mask(start_y, start_x):
            """Return exact pixel locations for this character"""
            stack = [(start_y, start_x)]
            pixels = []

            while stack:
                y, x = stack.pop()

                if (y < 0 or y >= height or x < 0 or x >= width or 
                    visited[y, x] or arr[y, x] == 0):
                    continue

                visited[y, x] = True
                pixels.append((y, x))

                # 8-connected neighbors
                stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1),
                             (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)])

            return pixels

        # Find all connected components with their pixel masks
        for y in range(height):
            for x in range(width):
                if arr[y, x] == 1 and not visited[y, x]:
                    pixels = flood_fill_with_mask(y, x)

                    if len(pixels) > 0:
                        ys = [p[0] for p in pixels]
                        xs = [p[1] for p in pixels]

                        min_y, max_y = min(ys), max(ys)
                        min_x, max_x = min(xs), max(xs)

                        w = max_x - min_x + 1
                        h = max_y - min_y + 1

                        if (min_width < w < max_width and 
                            min_height < h < max_height):

                            # Create pixel mask for this character
                            mask = np.zeros((h, w), dtype=np.uint8)
                            for py, px in pixels:
                                mask[py - min_y, px - min_x] = 1

                            regions.append({
                                'bbox': (min_x, min_y, w, h),
                                'position': (min_x, min_y),
                                'mask': mask
                            })

        regions.sort(key=lambda r: (r['position'][1] // 20, r['position'][0]))
        return regions

    def extract_characters(self, image_path, threshold=180, padding=2, **kwargs):
        """Extract characters with masked (lasso-style) cropping"""
        img, gray, binary = self.preprocess_image(image_path, threshold)
        regions = self.find_character_regions_with_masks(binary, img, **kwargs)

        characters = []
        for region in regions:
            x, y, w, h = region['bbox']
            mask = region['mask']

            # Crop with minimal padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.width, x + w + padding)
            y2 = min(img.height, y + h + padding)

            # Extract the character region
            char_img = img.crop((x1, y1, x2, y2))

            # Create padded mask
            padded_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            mask_y_offset = y - y1
            mask_x_offset = x - x1
            padded_mask[mask_y_offset:mask_y_offset+h, 
                       mask_x_offset:mask_x_offset+w] = mask * 255

            mask_img = Image.fromarray(padded_mask, mode='L')

            # Create RGBA with transparency for non-character pixels
            char_rgba = char_img.convert('RGBA')
            char_array = np.array(char_rgba)
            mask_array = np.array(mask_img)

            # Apply mask: transparent where not part of character
            char_array[:, :, 3] = mask_array

            masked_char = Image.fromarray(char_array, 'RGBA')

            characters.append({
                'image': masked_char,
                'bbox': (x, y, w, h),
                'position': (x, y),
                'mask': mask
            })

        return characters, img

    def save_character(self, char_data, index, label=None):
        """Save masked character image as PNG with transparency"""
        char_img = char_data['image']

        if label:
            filename = f"{label}_{self.char_counter[label]:04d}.png"
            self.char_counter[label] += 1
        else:
            filename = f"char_{index:04d}.png"

        filepath = os.path.join(self.output_dir, filename)
        char_img.save(filepath)
        return filepath

    def visualize_detection(self, image_path, output_path=None, **kwargs):
        """Create preview showing detected regions"""
        characters, img = self.extract_characters(image_path, **kwargs)

        if output_path is None:
            output_path = os.path.join(self.output_dir, "preview", "detection.jpg")

        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)

        for char in characters:
            x, y, w, h = char['bbox']
            draw.rectangle([x, y, x + w, y + h], outline="lime", width=1)

        vis_img.save(output_path)
        print(f"✓ Detection preview: {output_path}")
        print(f"✓ Detected {len(characters)} characters")

        return len(characters), output_path

    def extract_and_save_all(self, image_path, **kwargs):
        """Extract all characters and save with masking"""
        characters, _ = self.extract_characters(image_path, **kwargs)

        saved_files = []
        for i, char_data in enumerate(characters):
            filepath = self.save_character(char_data, i)
            saved_files.append(filepath)

        print(f"\n✓ Extracted {len(saved_files)} masked characters")
        print(f"✓ Saved to: {self.output_dir}/")

        return saved_files

    def process_batch(self, image_paths, **kwargs):
        """Process multiple scanned pages"""
        all_files = []

        for i, path in enumerate(image_paths, 1):
            print(f"\nProcessing page {i}/{len(image_paths)}: {path}")
            files = self.extract_and_save_all(path, **kwargs)
            all_files.extend(files)

        print(f"\n=== COMPLETE ===")
        print(f"Total characters extracted: {len(all_files)}")
        return all_files


if __name__ == "__main__":
    print("Letter Extractor - Lasso Masking Edition")
    print("=" * 50)

    extractor = LetterExtractor(output_dir="extracted_chars")

    # Example usage
    test_image = "018.jpg"

    if os.path.exists(test_image):
        print(f"\nProcessing: {test_image}\n")

        # First show detection preview
        extractor.visualize_detection(test_image, threshold=180, padding=2)

        # Then extract all characters (UNCOMMENTED NOW!)
        print("\nExtracting individual characters...")
        extractor.extract_and_save_all(test_image, threshold=180, padding=2)

        print("\n" + "="*50)
        print("DONE! Check the extracted_chars/ folder")
        print("="*50)
    else:
        print("\nPlace your JPG scans here and run:")
        print("  python letter_extractor.py")
