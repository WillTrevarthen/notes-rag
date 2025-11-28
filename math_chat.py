import os
import fitz  # PyMuPDF
import base64
import chromadb
import re
from chromadb.utils import embedding_functions
from openai import OpenAI

class MathRAG:
    def __init__(self, folder_path, db_path):
        self.folder_path = folder_path
        
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your environment variables.")

        self.client = OpenAI(api_key=self.api_key)

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-3-small"
        )
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="math_notes",
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        
    def load_and_index_pdfs(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            return

        local_files = {f for f in os.listdir(self.folder_path) if f.endswith('.pdf')}
        existing_data = self.collection.get(include=['metadatas'])
        indexed_files = {m['file'] for m in existing_data['metadatas']}
        new_files = list(local_files - indexed_files)
        
        if not new_files:
            return

        print(f"Processing {len(new_files)} new PDFs...")
        for filename in new_files:
            self._process_single_pdf(filename)

    def _process_single_pdf(self, filename):
        path = os.path.join(self.folder_path, filename)
        try:
            doc = fitz.open(path)
            documents, metadatas, ids = [], [], []

            for page_num, page in enumerate(doc):
                text = page.get_text()
                clean_text = " ".join(text.split())
                if len(clean_text) < 20: continue

                documents.append(clean_text)
                metadatas.append({"file": filename, "page": page_num, "path": path})
                ids.append(f"{filename}_p{page_num}")

            if documents:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def _pdf_page_to_base64_image(self, pdf_path, page_num):
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
            return base64.b64encode(pix.tobytes("png")).decode('utf-8')
        except:
            return None

    def query(self, user_question):
        results = self.collection.query(
            query_texts=[user_question],
            n_results=3
        )

        relevant_images = []
        context_desc = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                img_b64 = self._pdf_page_to_base64_image(meta['path'], meta['page'])
                if img_b64:
                    relevant_images.append(img_b64)
                    context_desc.append(f"From {meta['file']}, Page {meta['page'] + 1}")

        return self._ask_vision_model(user_question, relevant_images, context_desc)

    def _clean_latex(self, text):
        """
        Advanced cleaning to force Streamlit-compatible LaTeX.
        """
        # 1. Convert \[ ... \] to $$ ... $$
        text = re.sub(r'\\\[(.*?)\\\]', r'$$ \1 $$', text, flags=re.DOTALL)
        
        # 2. Convert \( ... \) to $ ... $
        text = re.sub(r'\\\((.*?)\\\)', r'$ \1 $', text, flags=re.DOTALL)
        
        # 3. HEURISTIC FIX: Convert ( math ) to $ math $
        # We look for parentheses containing typical math chars like _, ^, \, =, or >
        # This fixes ( G = (g_{ij}) ) turning into $ G = (g_{ij}) $
        pattern = r'\(\s*([^)\n]*?[_=^\\].*?)\s*\)'
        text = re.sub(pattern, r'$ \1 $', text)
        
        return text

    def _ask_vision_model(self, question, images, context_desc):
        if not images:
            return "I couldn't find any relevant notes.", [], []

        messages = [
            {
                "role": "system", 
                "content": """You are a math tutor used by a STEM student.
                
                CRITICAL FORMATTING INSTRUCTIONS:
                1. You are generating content for a system that ONLY understands LaTeX wrapped in dollar signs.
                2. DO NOT use parentheses `(...)` to wrap math formulas. 
                3. DO NOT use `\[` or `\(`. 
                4. ALWAYS use `$` for inline math. Example: $x^2$
                5. ALWAYS use `$$` for block math. Example: $$ \int x dx $$
                6. If you write a matrix or an equation, it MUST be inside `$$ $$`.
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
                max_tokens=1500 # Increased tokens for longer math explanations
            )
            raw_text = response.choices[0].message.content
            
            # Run the aggressive cleaner
            final_text = self._clean_latex(raw_text)
            
            return final_text, images, context_desc
            
        except Exception as e:
            return f"Error contacting OpenAI: {e}", [], []