
from langchain_core.prompts import PromptTemplate
from openai import AzureOpenAI
from faiss_manager import FaissManager
from pdf_processor import PDFProcessor
from dotenv import load_dotenv
from typing import Optional

from config import Configurations
from utils import * 





load_dotenv()





class SchoolRAG:

    def __init__(
        self, 
        pdf_path: str,
        vector_store_name: Optional[str] = "temp_vec",
        load_pre_stored: bool = False
    ):
        
        self.pdf_path = pdf_path
        self.__vector_store_name = vector_store_name

        # creating faiss manager instance
        self.__fm = FaissManager(
            root_dir="LOCAL"
        )


        # Initialize processor
        self.__processor = PDFProcessor(
            chunk_size=1000,
            chunk_overlap=200
        )

        # llm (client) instance
        self.__client = AzureOpenAI()
    



    def setup(self, load_pre_stored: bool ):
        
        if load_pre_stored:
            return True 
        
        else:
            # Get chunked documents
            documents = self.__processor.process_pdf(self.pdf_path)
            # creating and storing vector store
            self.__fm.create_store(
                name=self.__vector_store_name, 
                overwrite=True, 
                documents=documents
            )




    def _fetch_index_content(self) -> str:
        """ 
        It fetch docs from rag and hits llm then extract a dictionary of chapter 
        names and numbers (index) then save them using pickle
        """
        
        # query = "table of contents, index, index page, indexing page"
        query = "eturn the book’s Table of Contents / Index page: a list of chapter numbers and titles"

        docs = self.__fm.mmr_search(name=self.__vector_store_name, query=query)

        content = ""
        for doc in docs:
            content += doc.page_content

        return content
        


    def save_index_content(self):

        # getting index raw content
        raw_content = self._fetch_index_content()

        # prompt
        
        template = PromptTemplate(
            template="""
You are an expert data extractor.

Goal:
Extract all chapter titles and their corresponding chapter numbers from the provided book content.

OUTPUT REQUIREMENTS:
1. Output MUST be valid JSON.
2. Use ONLY double quotes.
3. Chapter numbers must be JSON string keys (e.g., "1", "2", "3").
4. Chapter titles must be JSON string values.
5. Do NOT add any explanation or commentary.
6. Do NOT add text before or after the JSON.
7. Do NOT add trailing commas.
8. Return ONLY a single JSON object.

VALID OUTPUT EXAMPLE:
{{
  "1": "Introduction to the World",
  "2": "Exploring Electricity"
}}

Raw Content:
{raw_content}

""",
            input_variables=['raw_content']
        )


        # prompt
        prompt = template.invoke({'raw_content': raw_content}).text

        response = self.__client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=8192,
            temperature=0.0
        )

        # saving content
        save_python_dict_string_as_json(
            python_dict_str=response.choices[0].message.content,
            file_path=Configurations.chapters_file_path
        )
        
        return response


    





if __name__ == "__main__":


    rag = SchoolRAG(
        pdf_path="cls_8_science_ncert_top_5_ch.pdf",
        load_pre_stored=True, 
        vector_store_name="cls-viii"
    )


    rag.setup(load_pre_stored=True)


    response = rag.save_index_content()


    print(response.choices[0].message.content)


    opt_str = response.choices[0].message.content





