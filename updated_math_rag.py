import os
import fitz  # PyMuPDF
import base64
import chromadb
import hashlib
import re
from chromadb.utils import embedding_functions
from openai import OpenAI

class MathRAG:
    def __init__(self, folder_path, db_path):
        self.folder_path = folder_path
        
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")

        self.client = OpenAI(api_key=self.api_key)

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-3-small"
        )
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="math_notes_hybrid", # New name for the hybrid approach
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_file_hash(self, filepath):
        """Calculates MD5 hash to detect file changes."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def load_and_index_pdfs(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            return

        local_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        
        # Get existing files and hashes
        existing_data = self.collection.get(include=['metadatas'])
        indexed_map = {} 
        if existing_data['metadatas']:
            for meta in existing_data['metadatas']:
                indexed_map[meta['file']] = meta.get('hash', '')

        print(f"Checking {len(local_files)} files...")
        
        for filename in local_files:
            path = os.path.join(self.folder_path, filename)
            current_hash = self._get_file_hash(path)
            
            if filename not in indexed_map or indexed_map[filename] != current_hash:
                print(f"Indexing (Text-Mode): {filename}")
                if filename in indexed_map:
                    self.collection.delete(where={"file": filename})
                
                self._process_single_pdf(filename, current_hash)
            else:
                print(f"Skipping {filename} (No changes)")

    def _process_single_pdf(self, filename, file_hash):
        path = os.path.join(self.folder_path, filename)
        try:
            doc = fitz.open(path)
            documents, metadatas, ids = [], [], []

            for page_num, page in enumerate(doc):
                # --- THE CHEAP PART ---
                # We use standard text extraction. It's free.
                text = page.get_text()
                clean_text = " ".join(text.split())
                
                if len(clean_text) < 20: 
                    continue

                documents.append(clean_text)
                metadatas.append({
                    "file": filename, 
                    "page": page_num, 
                    "path": path,
                    "hash": file_hash
                })
                ids.append(f"{filename}_p{page_num}")

            if documents:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                print(f"   -> Indexed {len(documents)} pages.")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def _pdf_page_to_base64_image(self, pdf_path, page_num):
        try:
            doc = fitz.open(pdf_path)
            if page_num < 0 or page_num >= len(doc): return None
            
            # Lower resolution (1,1) is enough for GPT-4o-mini and saves tokens
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
            return base64.b64encode(pix.tobytes("png")).decode('utf-8')
        except:
            return None

    def query(self, user_question):
            # 1. Retrieve the TOP 3 text matches
            results = self.collection.query(
                query_texts=[user_question],
                n_results=3 
            )

            relevant_images = []
            context_desc = []
            
            # This set prevents processing the same page twice
            unique_pages_map = {} # Key: (filename, page_num), Value: (path, distance_score)

            if results['ids']:
                # Loop through the top 3 results
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    current_page = meta['page']
                    filename = meta['file']
                    path = meta['path']
                    
                    # 2. For EACH match, get Prev (-1), Current (0), and Next (+1)
                    # This ensures we get the full context around every potential hit.
                    pages_to_fetch = [current_page - 1, current_page, current_page + 1]
                    
                    for p_num in pages_to_fetch:
                        key = (filename, p_num)
                        
                        # Store if we haven't seen it, or if it's the exact match (priority)
                        if key not in unique_pages_map:
                            unique_pages_map[key] = path

            # 3. Sort the pages so the AI reads them in logical order (File A: Pg 1,2,3... File B: Pg 10,11,12)
            sorted_keys = sorted(unique_pages_map.keys(), key=lambda x: (x[0], x[1]))

            # 4. Fetch Images (Limit to max 9 images to prevent token overflow)
            # We take the first 9 sorted pages.
            MAX_IMAGES = 9
            final_keys = sorted_keys[:MAX_IMAGES]

            for filename, p_num in final_keys:
                pdf_path = unique_pages_map[(filename, p_num)]
                
                img_b64 = self._pdf_page_to_base64_image(pdf_path, p_num)
                if img_b64:
                    relevant_images.append(img_b64)
                    context_desc.append(f"From {filename}, Page {p_num + 1}")

            return self._ask_vision_model(user_question, relevant_images, context_desc)

    def _safe_latex_format(self, text):
        text = re.sub(r'\\\[(.*?)\\\]', r'$$ \1 $$', text, flags=re.DOTALL)
        text = re.sub(r'\\\((.*?)\\\)', r'$ \1 $', text, flags=re.DOTALL)
        return text

    def _ask_vision_model(self, question, images, context_desc):
        if not images:
            return "I couldn't find any relevant notes.", [], []

        messages = [
            {
                "role": "system", 
                "content": """You are an expert math tutor.
                
                FORMATTING RULES:
                1. Use LaTeX for ALL math.
                2. Inline math: $...$
                3. Block math: $$...$$
                4. Do NOT use `\\[` or `\\(`.
                
                INSTRUCTIONS:
                1. Analyze the provided images first.
                2. ALWAYS start your response with a section called "**🔍 Analysis of Notes**".
                   - In this section, identify the SINGLE most relevant formula, definition, or theorem found in the images.
                   - Quote which file/page it came from.
                   - If the images are irrelevant to the user's question, clearly state: "The retrieved notes discuss [Topic X], which is not directly related to your question."
                
                3. Then, provide the **Answer**:
                   - **Scenario A:** If the answer IS in the notes, explain it step-by-step using the notes.
                   - **Scenario B:** If the answer is NOT in the notes, solve the problem using your own expert knowledge, but append this warning at the end:
                     "**Note:** This solution was derived from general mathematical principles because the specific answer was not found in the retrieved documents."
                """
            },
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        
        for i, img in enumerate(images):
            messages[1]["content"].append({
                "type": "text", 
                "text": f"Context Image {i+1}: {context_desc[i]}"
            })
            messages[1]["content"].append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{img}"}
            })

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages, 
                max_tokens=1500
            )
            raw_text = response.choices[0].message.content
            final_text = self._safe_latex_format(raw_text)
            return final_text, images, context_desc
            
        except Exception as e:
            return f"Error contacting OpenAI: {e}", [], []