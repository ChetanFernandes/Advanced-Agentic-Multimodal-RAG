from pathlib import Path
from langchain_core.documents import Document
from src.logger_config import log
import os, time



def get_doc_image_dir(file_name,user_id): 
    doc_name = Path(file_name).stem # give only file name without etension
    base_output_dir = "all_images"
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_output_dir, user_id, timestamp, doc_name + "_images")
    os.makedirs(output_dir,exist_ok=True)
    return output_dir



def extract_text_elements(raw_pdf_elements):
    Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables =  [], [], [],  [], [], [], [], []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Header" in str(type(element)):
             Header.append(str(element))
        elif "unstructured.documents.elements.Footer" in str(type(element)):
             Footer.append(str(element))
        elif "unstructured.documents.elements.Title" in str(type(element)):
             Title.append(str(element))
        elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
             NarrativeText.append(str(element))
        elif "unstructured.documents.elements.Text" in str(type(element)):
             Text.append(str(element))
        elif "unstructured.documents.elements.ListItem" in str(type(element)):
             ListItem.append(str(element))
        elif "unstructured.documents.elements.Image" in str(type(element)):
             Img.append(str(element))
        elif "unstructured.documents.elements.Table" in str(type(element)):
             Tables.append(str(element))
        elif 'unstructured.documents.elements.CompositeElement' in str(type(element)):
             Text.append(str(element))

    category_counts = {}
    for element in raw_pdf_elements:
         category = str(type(element))
         log.info(f"Category -> {category}")
         if category in category_counts:
            category_counts[category] += 1
         else:
            category_counts[category] = 1

    # Unique_categories will have unique elements
    unique_categories = set(category_counts.keys())
    log.info(f"unique_Category_counts{unique_categories}")

    return Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables


class final_doc:
    def __init__(self, Header, Footer, Title, NarrativeText, Text, ListItem, Img, Tables, image_summaries,file_name):
        self.Header = Header
        self.Footer = Footer
        self.Title = Title
        self.NarrativeText = NarrativeText
        self.Text = Text
        self.ListItem = ListItem
        self.Img = Img
        self.Tables = Tables
        self.image_summaries = image_summaries
        self.file_name = file_name

    def safe_text(self, value):
        """
        Converts lists or nested values into a single clean line of text.
        Example: ["A", "B", "C"] -> "A. B. C"
        """
        if not value:
            return ""
        if isinstance(value, list):
            return ". ".join(str(v).strip() for v in value if v)
        return str(value).strip()

    def overall(self):
        """
        Combine all fields in single-line format under each label.
        """
        overall_text = "\n".join([
            f"Header:\n{self.safe_text(self.Header)}\n",
            f"Footer:\n{self.safe_text(self.Footer)}\n",
            f"Title:\n{self.safe_text(self.Title)}\n",
            f"Narrative_Text:\n{self.safe_text(self.NarrativeText)}\n",
            f"Text:\n{self.safe_text(self.Text)}\n",
            f"List Items:\n{self.safe_text(self.ListItem)}\n",
            f"Image Items:\n{self.safe_text(self.Img)}\n",
            f"Table Items:\n{self.safe_text(self.Tables)}\n",
            f"Image summaries:\n{self.safe_text(self.image_summaries)}\n",
        ])

        documents = [
            Document(
                page_content=overall_text,
                metadata={"source": self.file_name, "type": "combined"}
            )
        ]
        return documents



