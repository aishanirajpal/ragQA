"""
RAG System Module
Core retrieval-augmented generation system for document QA
"""

import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from document_processor import DocumentProcessor

# Try to import advanced dependencies
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    ADVANCED_MODE = True
except ImportError:
    ADVANCED_MODE = False

# Try to import generative dependencies
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    GENERATIVE_MODE_ENABLED = True
except ImportError:
    GENERATIVE_MODE_ENABLED = False

# Try to import Hugging Face dependencies
try:
    from transformers import pipeline, AutoTokenizer
    import torch
    HF_GENERATIVE_MODE_ENABLED = True
except ImportError:
    HF_GENERATIVE_MODE_ENABLED = False


class RAGSystem:
    """RAG system that works with or without advanced ML dependencies"""
    
    def __init__(self, use_advanced: bool = True, google_api_key: Optional[str] = None):
        """
        Initialize RAG system
        
        Args:
            use_advanced: Whether to use advanced ML models if available
            google_api_key: Optional Google API key for generative answers
        """
        self.document_processor = DocumentProcessor()
        self.documents: Dict[str, Dict] = {} # Still used for basic mode
        
        self.advanced_mode = ADVANCED_MODE and use_advanced
        self.generative_mode = False
        self.qa_chain = None
        self.chroma_collection = None
        self.hf_pipeline = None
        self.generative_source = "none"

        if self.advanced_mode:
            self._setup_advanced_models()
        
        # Prioritize Gemini if key is available
        if google_api_key and GENERATIVE_MODE_ENABLED:
            self._setup_gemini_models(google_api_key)
        # Skip Hugging Face model setup for faster startup - use basic mode instead
        # elif self.advanced_mode and HF_GENERATIVE_MODE_ENABLED:
        #     self._setup_huggingface_models()

        print(f"🤖 RAG System initialized in {'Advanced' if self.advanced_mode else 'Basic'} mode")
        if self.generative_mode:
            print(f"✨ Generative answers enabled with {self.generative_source.upper()}")
    
    def _setup_gemini_models(self, google_api_key: str):
        """Initialize generative models using LangChain and Gemini"""
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                                         convert_system_message_to_human=True)
            
            prompt_template = """
            You are a helpful assistant designed to answer questions from a provided document.
            Your goal is to provide a clear, concise, and well-structured answer based *only* on the information in the context below. Do not use any outside knowledge.

            Follow these instructions carefully:
            1.  **Synthesize, do not copy:** Read the relevant parts of the context and rephrase the information in your own words. Do not copy sentences verbatim.
            2.  **Use bullet points:** If the answer involves a list, steps, or multiple key points, format them using bullet points for readability.
            3.  **Be direct:** Start the answer directly without introductory phrases like "According to the context...".
            4.  **Handle missing information:** If the answer is not available in the context, state clearly: "I couldn't find a definitive answer in the document."

            Context:
            {context}

            Question: {question}

            Helpful Answer:
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            self.qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            self.generative_mode = True
            self.generative_source = "gemini"
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini: {e}")
            print("🔧 Falling back to extractive answers")
            self.generative_mode = False

    def _setup_huggingface_models(self):
        """Initialize a local generative model from Hugging Face"""
        try:
            print("⚙️ No Google API key found. Attempting to load local Hugging Face model...")
            # Use a smaller, well-balanced model. Requires `accelerate` for `device_map`.
            model_name = "google/flan-t5-base"
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=self.hf_tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.generative_mode = True
            self.generative_source = "huggingface"
            print("✅ Hugging Face model (flan-t5-base) loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load Hugging Face model: {e}")
            print("🔧 Falling back to extractive answers.")
            self.generative_mode = False

    def _setup_advanced_models(self):
        """Initialize advanced ML models and ChromaDB if available"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
            
            # Setup ChromaDB
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.chroma_collection = chroma_client.get_or_create_collection(name="rag_documents")
            print(f"✅ ChromaDB collection loaded. Total items: {self.chroma_collection.count()}")

        except Exception as e:
            print(f"⚠️ Failed to load advanced models or ChromaDB: {e}")
            print("🔧 Falling back to basic mode")
            self.advanced_mode = False
    
    def clear_documents(self):
        """Clears all loaded documents and resets the system."""
        self.documents = {} # Clear basic mode docs
        if self.advanced_mode and self.chroma_collection:
            # This is a destructive operation. A real app might archive or use separate collections.
            # For this example, we clear the existing collection by deleting and recreating it.
            try:
                collection_name = self.chroma_collection.name
                chroma_client = chromadb.PersistentClient(path="./chroma_db")
                chroma_client.delete_collection(name=collection_name)
                self.chroma_collection = chroma_client.create_collection(name=collection_name)
                print("🗑️ ChromaDB collection cleared.")
            except Exception as e:
                print(f"⚠️ Error clearing ChromaDB collection: {e}")

        print("🗑️ All documents cleared.")

    def load_document(self, filename: str, file_content: bytes):
        """Load and process a new document, adding it to the collection."""
        # For advanced mode, check if the document already exists in ChromaDB
        if self.advanced_mode and self.chroma_collection:
            existing_docs = self.chroma_collection.get(where={"source": filename})
            if existing_docs and len(existing_docs['ids']) > 0:
                print(f"⚠️ Document '{filename}' is already loaded in ChromaDB. To update, please clear documents first.")
                return

        new_chunks = self.document_processor.process_document(filename, file_content)
        if not new_chunks:
            raise ValueError("No valid chunks extracted from document")

        # Basic mode still uses in-memory dict
        self.documents[filename] = {'chunks': new_chunks}

        if self.advanced_mode and self.embedding_model and self.chroma_collection:
            self._add_document_to_vector_store(filename, new_chunks)

        print(f"✅ Document '{filename}' loaded with {len(new_chunks)} chunks.")

    def load_document_from_path(self, file_path: str):
        """Load document from file path (for testing)"""
        with open(file_path, 'rb') as f:
            content = f.read()
        import os
        self.load_document(os.path.basename(file_path), content)
    
    def _add_document_to_vector_store(self, filename: str, new_chunks: List[str]):
        """Creates embeddings for new chunks and adds them to ChromaDB."""
        if not self.chroma_collection:
            print("⚠️ ChromaDB collection not available. Skipping vector store update.")
            return
        try:
            print(f"🔍 Creating embeddings for {len(new_chunks)} new chunks from '{filename}'...")
            new_embeddings = self.embedding_model.encode(new_chunks).tolist()
            
            # Create unique IDs and metadata for each chunk
            ids = [f"{filename}_{i}" for i in range(len(new_chunks))]
            metadatas = [{"source": filename} for _ in new_chunks]
            
            self.chroma_collection.add(
                embeddings=new_embeddings,
                documents=new_chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ ChromaDB updated. Total items in collection: {self.chroma_collection.count()}")

        except Exception as e:
            print(f"⚠️ Error creating embeddings or updating ChromaDB: {e}")
            print("🔧 Falling back to basic search for all documents")
            self.advanced_mode = False
    
    def _retrieve_chunks_advanced(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve chunks from all documents using ChromaDB."""
        if not self.chroma_collection or not self.embedding_model:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            results = self.chroma_collection.query(
                query_embeddings=query_embedding,
                n_results=k
            )
            
            retrieved_chunks = []
            if results and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    doc_name = results['metadatas'][0][i]['source']
                    chunk_text = results['documents'][0][i]
                    # ChromaDB returns distance, convert to similarity score (0-1)
                    # A simple approach for L2 distance: similarity = 1 - distance
                    # This is not a perfect cosine similarity, but works for ranking
                    similarity = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                    retrieved_chunks.append((doc_name, chunk_text, max(0.0, similarity)))
            
            return retrieved_chunks
        except Exception as e:
            print(f"⚠️ Advanced search failed: {e}")
            return self._retrieve_chunks_basic(query, k)
    
    def _retrieve_chunks_basic(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve chunks from all documents using keyword matching."""
        if not self.documents:
            return []
        
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for doc_name, doc_data in self.documents.items():
            for chunk in doc_data['chunks']:
                chunk_words = set(chunk.lower().split())
                overlap = len(query_words.intersection(chunk_words))
                if overlap > 0:
                    # Calculate relevance score
                    precision = overlap / len(query_words) if len(query_words) > 0 else 0
                    recall = overlap / len(chunk_words) if len(chunk_words) > 0 else 0
                    score = 0.7 * precision + 0.3 * recall
                    scored_chunks.append((doc_name, chunk, score))
        
        # Sort by relevance and return top k
        scored_chunks.sort(key=lambda x: x[2], reverse=True)
        return scored_chunks[:k]
    
    def _generate_answer(self, query: str, context_chunks: List[Tuple[str, str]]) -> str:
        """Generate answer using retrieved context from multiple documents."""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."

        # Route to the correct generative model or fallback
        if self.generative_mode and self.generative_source == "gemini":
            try:
                # Use LangChain for a generative answer, providing source documents in metadata
                langchain_docs = [Document(page_content=chunk, metadata={"source": doc_name}) for doc_name, chunk in context_chunks]
                
                # Format context for the prompt
                context_for_prompt = "\n\n".join([f"--- Start of content from {doc.metadata['source']} ---\n{doc.page_content}\n--- End of content from {doc.metadata['source']} ---" for doc in langchain_docs])

                result = self.qa_chain({"input_documents": langchain_docs, "question": query, "context": context_for_prompt}, return_only_outputs=True)
                return result.get('output_text', "Sorry, I had trouble generating an answer.")
            except Exception as e:
                print(f"⚠️ Gemini generation failed: {e}")
                # Fallback to extractive method if generative fails
                return self._generate_answer_extractive(query, context_chunks)
        
        elif self.generative_mode and self.generative_source == "huggingface" and self.hf_pipeline:
            return self._generate_answer_huggingface(query, context_chunks)
            
        else:
            # Use the original extractive method
            return self._generate_answer_extractive(query, context_chunks)

    def _generate_answer_huggingface(self, query: str, context_chunks: List[Tuple[str, str]]) -> str:
        """Generate an answer using the local Hugging Face model."""
        
        # Dynamically pack context to fit within the model's token limit
        # Flan-T5 has a 512 token limit. We'll aim for ~400 for context to leave space for the answer.
        max_context_tokens = 400
        packed_context = ""
        
        base_prompt_template = f"""
        Answer the question based only on the provided context.

        Instructions:
        - Rephrase the information in your own words. Do not copy the text directly.
        - If the answer has multiple parts, use bullet points.
        - If the context does not contain the answer, say "I couldn't find an answer in the document."

        Question: {query}
        """

        for _, chunk in context_chunks:
            # Check token count before adding the next chunk
            potential_context = packed_context + "\\n" + chunk
            token_count = len(self.hf_tokenizer.encode(potential_context + base_prompt_template))
            
            if token_count <= max_context_tokens:
                packed_context = potential_context
            else:
                # We've added as much context as we can
                break
        
        if not packed_context:
            # This can happen if even the single most relevant chunk is too large.
            # In this case, we truncate the most relevant chunk.
            most_relevant_chunk = context_chunks[0][1]
            truncated_tokens = self.hf_tokenizer.encode(most_relevant_chunk, max_length=max_context_tokens, truncation=True)
            packed_context = self.hf_tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        prompt = f"""
        {base_prompt_template}

        Context:
        {packed_context}

        Answer:
        """
        
        try:
            # Increase max_length for the generated output, not to be confused with input length
            result = self.hf_pipeline(prompt, max_new_tokens=256, clean_up_tokenization_spaces=True)
            return result[0]['generated_text']
        except Exception as e:
            print(f"⚠️ Hugging Face generation failed: {e}")
            return self._generate_answer_extractive(query, context_chunks) # Fallback

    def _generate_answer_extractive(self, query: str, context_chunks: List[Tuple[str, str]]) -> str:
        """Original extractive answer generation, adapted for multiple documents."""
        # Prepend the document name to each chunk for context
        context_parts = [f"From document '{doc_name}': {chunk}" for doc_name, chunk in context_chunks[:3]]
        context = "\n\n".join(context_parts)
        
        # Extract most relevant sentences using keyword overlap
        sentences = re.split(r'[.!?]+', context)
        query_words = set(query.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only substantial sentences
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((sentence, overlap))
        
        # Sort by relevance and combine top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if scored_sentences:
            answer_parts = [sent[0] for sent in scored_sentences[:3]]
            answer = ". ".join(answer_parts)
            if not answer.endswith('.'):
                answer += "."
            return answer
        else:
            # Fallback: return first part of most relevant chunk with its source
            doc_name, chunk = context_chunks[0]
            return f"From '{doc_name}': {chunk[:400]}..." if len(chunk) > 400 else f"From '{doc_name}': {chunk}"
    
    def answer_question(self, query: str) -> Dict[str, any]:
        """
        Answer a question using the loaded document
        
        Args:
            query: Question to answer
            
        Returns:
            Dictionary with answer, sources, and confidence
        """
        current_mode = "basic"
        if self.advanced_mode:
            current_mode = "advanced"
        if self.generative_mode:
            current_mode = f"generative ({self.generative_source})"

        # Check if any documents are loaded, either in ChromaDB or in-memory
        docs_loaded = (self.advanced_mode and self.chroma_collection and self.chroma_collection.count() > 0) or \
                      (not self.advanced_mode and self.documents)
        
        if not docs_loaded:
            return {
                "answer": "❌ No documents loaded. Please upload one or more files to begin.",
                "sources": [],
                "confidence": 0.0,
                "mode": current_mode
            }
        
        # Retrieve relevant chunks
        if self.advanced_mode and self.chroma_collection:
            relevant_chunks = self._retrieve_chunks_advanced(query, k=5)
        else:
            relevant_chunks = self._retrieve_chunks_basic(query, k=5)
        
        if not relevant_chunks:
            return {
                "answer": "❌ No relevant information found in the loaded documents.",
                "sources": [],
                "confidence": 0.0,
                "mode": current_mode
            }
        
        # Extract chunks and calculate confidence
        context_chunks = [(doc_name, chunk) for doc_name, chunk, _ in relevant_chunks]
        avg_confidence = np.mean([score for _, _, score in relevant_chunks]) if relevant_chunks else 0.0
        
        # Generate answer
        answer = self._generate_answer(query, context_chunks)
        
        # Prepare sources with document names
        sources = [{'source': doc_name, 'content': chunk} for doc_name, chunk, _ in relevant_chunks[:3]]
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": float(avg_confidence),
            "mode": current_mode
        }
    
    def get_document_info(self) -> Dict[str, any]:
        """Get information about all loaded documents."""
        if self.advanced_mode and self.chroma_collection:
            try:
                # Get info from ChromaDB
                count = self.chroma_collection.count()
                if count > 0:
                    all_docs = self.chroma_collection.get()
                    doc_names = sorted(list(set(meta['source'] for meta in all_docs['metadatas'])))
                else:
                    doc_names = []
                
                return {
                    "doc_names": doc_names,
                    "total_chunks": count,
                    "mode": f"generative ({self.generative_source})" if self.generative_mode else "advanced",
                    "generative": self.generative_mode,
                    "has_embeddings": count > 0
                }
            except Exception as e:
                print(f"⚠️ Could not get info from ChromaDB: {e}")
                # Fallback to basic if Chroma fails
                self.advanced_mode = False

        # Fallback to basic (in-memory) info
        total_chunks = sum(len(doc.get('chunks', [])) for doc in self.documents.values())
        return {
            "doc_names": list(self.documents.keys()),
            "total_chunks": total_chunks,
            "mode": "basic",
            "generative": self.generative_mode,
            "has_embeddings": False
        }
