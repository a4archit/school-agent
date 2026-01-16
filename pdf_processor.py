from typing import List, Dict
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document




class PDFProcessor:
    """Process PDF documents for RAG system with page tracking."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators optimized for document chunking
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    def load_pdf(
            self, 
            pdf_path: str, 
            _from: int = 0, 
            to: int = 0
        ) -> List[Document]:
        """Load PDF and extract pages.

        Args:
            pdf_path (str): Path to the PDF file
            _from (int, optional): Start index. Defaults to 0.
            to (int, optional): End index. Defaults to 0.

        Raises:
            FileNotFoundError: _description_
            ValueError: _description_

        Returns:
            List[Document]: List of Document objects with page content and metadata
        """
        
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF: {pdf_path}")
        
        loader = PyPDFLoader(str(path))
        pages = loader.load()

        if _from or to:
            return pages[_from:to]
        
        return pages
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving page numbers.
        
        Args:
            documents: List of Document objects from PDF pages
            
        Returns:
            List of chunked Document objects with page metadata
        """
        chunked_docs = []
        
        for doc in documents:
            # Get page number from metadata
            page_num = doc.metadata.get('page', 0)
            source = doc.metadata.get('source', '')
            
            # Split the document into chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create new Document objects with metadata
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': source,
                        'page': page_num,
                        'chunk': i,
                        'total_chunks_in_page': len(chunks)
                    }
                )
                chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete pipeline: load PDF and chunk it.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of chunked Document objects ready for vector DB
        """
        # Load pages
        pages = self.load_pdf(pdf_path)
        print(f"Loaded {len(pages)} pages from {pdf_path}")
        
        # Chunk documents
        chunks = self.chunk_documents(pages)
        print(f"Created {len(chunks)} chunks")
        
        return chunks


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = PDFProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Process PDF
    pdf_path = "file.pdf"
    
    try:
        # Get chunked documents
        texts = processor.process_pdf(pdf_path)
        
        # Display sample output
        print("\n--- Sample Chunk ---")
        print(f"Content: {texts[0].page_content[:200]}...")
        print(f"Metadata: {texts[0].metadata}")
        
        # Now you can add to Qdrant
        # qdrant_db.add_documents(texts)
        
    except Exception as e:
        print(f"Error processing PDF: {e}")