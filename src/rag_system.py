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
    import faiss
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
        self.documents: Dict[str, Dict] = {}
        self.index_to_doc_map: List[Tuple[str, str]] = [] # Maps FAISS index to (doc_name, chunk)

        self.advanced_mode = ADVANCED_MODE and use_advanced
        self.generative_mode = False
        self.qa_chain = None

        if self.advanced_mode:
            self._setup_advanced_models()
        
        if google_api_key and GENERATIVE_MODE_ENABLED:
            self._setup_generative_models(google_api_key)

        print(f"🤖 RAG System initialized in {'Advanced' if self.advanced_mode else 'Basic'} mode")
        if self.generative_mode:
            print("✨ Generative answers enabled with Gemini")
    
    def _setup_generative_models(self, google_api_key: str):
        """Initialize generative models using LangChain and Gemini"""
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                                         convert_system_message_to_human=True)
            
            prompt_template = """
            You are a helpful assistant that answers questions based on the context provided.
            Answer the user's question concisely and accurately, using only the information from the context below.
            If the answer is not available in the context, say "I couldn't find a definitive answer in the document."

            Context:
            {context}

            Question: {question}

            Helpful Answer:
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            self.qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            self.generative_mode = True
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini: {e}")
            print("🔧 Falling back to extractive answers")
            self.generative_mode = False

    def _setup_advanced_models(self):
        """Initialize advanced ML models if available"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load embedding model: {e}")
            print("🔧 Falling back to basic mode")
            self.advanced_mode = False
    
    def clear_documents(self):
        """Clears all loaded documents and resets the system."""
        self.documents = {}
        self.index_to_doc_map = []
        self.vector_store = None
        print("🗑️ All documents cleared.")

    def load_document(self, filename: str, file_content: bytes):
        """Load and process a new document, adding it to the collection."""
        if filename in self.documents:
            print(f"⚠️ Document '{filename}' is already loaded. To update, please clear documents first.")
            return

        new_chunks = self.document_processor.process_document(filename, file_content)
        if not new_chunks:
            raise ValueError("No valid chunks extracted from document")

        self.documents[filename] = {'chunks': new_chunks}
        
        # Add chunks to the global map before creating embeddings
        for chunk in new_chunks:
            self.index_to_doc_map.append((filename, chunk))

        if self.advanced_mode and self.embedding_model:
            self._add_document_to_vector_store(new_chunks)

        print(f"✅ Document '{filename}' loaded with {len(new_chunks)} chunks.")

    def load_document_from_path(self, file_path: str):
        """Load document from file path (for testing)"""
        with open(file_path, 'rb') as f:
            content = f.read()
        import os
        self.load_document(os.path.basename(file_path), content)
    
    def _add_document_to_vector_store(self, new_chunks: List[str]):
        """Creates embeddings for new chunks and adds them to the vector store."""
        try:
            print(f"🔍 Creating embeddings for {len(new_chunks)} new chunks...")
            new_embeddings = self.embedding_model.encode(new_chunks)
            faiss.normalize_L2(new_embeddings)

            if self.vector_store is None:
                dimension = new_embeddings.shape[1]
                self.vector_store = faiss.IndexFlatIP(dimension)
            
            self.vector_store.add(new_embeddings.astype('float32'))
            print(f"✅ Vector store updated. Total indexed chunks: {self.vector_store.ntotal}")

        except Exception as e:
            print(f"⚠️ Error creating embeddings: {e}")
            print("🔧 Falling back to basic search for all documents")
            self.advanced_mode = False
            self.vector_store = None # Ensure we don't use a partial index
    
    def _retrieve_chunks_advanced(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve chunks from all documents using semantic embeddings."""
        if not self.vector_store or not self.embedding_model:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.index_to_doc_map):
                    doc_name, chunk_text = self.index_to_doc_map[idx]
                    results.append((doc_name, chunk_text, float(score)))
            
            return results
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

        if self.generative_mode and self.qa_chain:
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
        else:
            # Use the original extractive method
            return self._generate_answer_extractive(query, context_chunks)

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
        current_mode = "generative" if self.generative_mode else ("advanced" if self.advanced_mode else "basic")

        if not self.documents:
            return {
                "answer": "❌ No documents loaded. Please upload one or more files to begin.",
                "sources": [],
                "confidence": 0.0,
                "mode": current_mode
            }
        
        # Retrieve relevant chunks
        if self.advanced_mode and self.vector_store:
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
        total_chunks = sum(len(doc['chunks']) for doc in self.documents.values())
        return {
            "doc_names": list(self.documents.keys()),
            "total_chunks": total_chunks,
            "mode": "advanced" if self.advanced_mode else "basic",
            "generative": self.generative_mode,
            "has_embeddings": self.vector_store is not None and self.vector_store.ntotal > 0
        }
