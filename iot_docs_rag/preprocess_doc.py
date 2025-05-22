import asyncio
import json
import os
from io import BytesIO
from dataclasses import dataclass
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import glob
import fitz as pymupdf  # PyMuPDF
import pymupdf4llm
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
)

# DOCUMENTS_SOURCE_TYPE = "iot_docs"
DOCUMENTS_SOURCE_TYPE = "airplane_docs"

# Define the documents folder path
# DOCUMENTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
DOCUMENTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airplane_documents")


@dataclass
class ProcessedChunk:
    document_name: str
    chapter_name: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


# @dataclass
# class TextChunk:
#     content: str
#     section_name: str
#     chapter_name: str


class TextChunkSection(BaseModel):
    name: str
    level: int
    line_start: int
    line_end: Optional[int] = None
    content: str = ""
    subsections: list["TextChunkSection"] = []
    parent: Optional["TextChunkSection"] = None
    page_num: int = 1
    page_breaks: dict[int, int] = {}
    total_pages: Optional[int] = None

    @property
    def root(self) -> "TextChunkSection":
        """Get the root section to access total_pages."""
        current = self
        while current.parent is not None:
            current = current.parent
        return current


class TextChunkSectionInfo(BaseModel):
    name: str
    level: int


class TextChunk(BaseModel):
    content: str
    line_start: int
    line_end: int
    sections: list[str]
    pages: list[int]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        The extracted text in markdown format
    """
    try:
        doc = pymupdf.open(pdf_path)
        return pymupdf4llm.to_markdown(doc)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF.

    Args:
        data: PDF content as bytes

    Returns:
        The extracted text in markdown format
    """
    try:
        pdf_file = BytesIO(data)
        doc = pymupdf.open(stream=pdf_file)
        return pymupdf4llm.to_markdown(doc)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF bytes: {e}")


# def chunk_text(text: str, chunk_size: int = 5000) -> List[TextChunk]:
#     """Split text into chunks, respecting code blocks, paragraphs, and markdown headers.
#     Returns a list of TextChunk objects containing the content, section name and chapter name."""
#     chunks = []
#     start = 0
#     text_length = len(text)

#     # Extract all section headers and their positions
#     section_markers = []
#     chapter_markers = []
#     lines = text.split('\n')
#     current_pos = 0
#     current_section = "Introduction"  # Default section name
#     current_chapter = "Introduction"  # Default chapter name

#     for line in lines:
#         if line.startswith('#'):
#             # Count the level of the header
#             header_level = 0
#             for char in line:
#                 if char == '#':
#                     header_level += 1
#                 else:
#                     break

#             # Extract the section name (remove # and whitespace)
#             section_name = line[header_level:].strip()
#             if section_name:
#                 # Level 1 headers are chapters
#                 if header_level == 1:
#                     current_chapter = section_name
#                     chapter_markers.append((current_pos, section_name))

#                 # All headers are sections
#                 section_markers.append((current_pos, section_name, header_level, current_chapter))
#                 current_section = section_name

#         current_pos += len(line) + 1  # +1 for the newline

#     # Add an end marker
#     section_markers.append((text_length, "End", 0, current_chapter))

#     # Current section tracking
#     current_section_idx = 0
#     current_section_name = current_section
#     current_chapter_name = current_chapter

#     while start < text_length:
#         # Calculate end position
#         end = min(start + chunk_size, text_length)

#         # Update current section if needed
#         while (current_section_idx < len(section_markers) - 1 and
#                start >= section_markers[current_section_idx][0]):
#             current_section_idx += 1
#             current_section_name = section_markers[current_section_idx - 1][1]
#             current_chapter_name = section_markers[current_section_idx - 1][3]

#         # Check if this chunk would cross a section boundary
#         next_section_start = text_length
#         if current_section_idx < len(section_markers):
#             next_section_start = section_markers[current_section_idx][0]

#         if end > next_section_start and next_section_start > start:
#             end = next_section_start

#         # If we're at the end of the text, just take what's left
#         if end >= text_length:
#             chunk_content = text[start:].strip()
#             if chunk_content:
#                 chunks.append(TextChunk(
#                     content=chunk_content,
#                     section_name=current_section_name,
#                     chapter_name=current_chapter_name
#                 ))
#             break

#         # Try to find a code block boundary first (```)
#         chunk = text[start:end]
#         code_block = chunk.rfind("```")
#         if code_block != -1 and code_block > chunk_size * 0.3:
#             end = start + code_block

#         # If no code block, try to break at a paragraph
#         elif "\n\n" in chunk:
#             # Find the last paragraph break
#             last_break = chunk.rfind("\n\n")
#             if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
#                 end = start + last_break

#         # If no paragraph break, try to break at a sentence
#         elif ". " in chunk:
#             # Find the last sentence break
#             last_period = chunk.rfind(". ")
#             if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
#                 end = start + last_period + 1

#         # Extract chunk and clean it up
#         chunk_content = text[start:end].strip()
#         if chunk_content:
#             chunks.append(TextChunk(
#                 content=chunk_content,
#                 section_name=current_section_name,
#                 chapter_name=current_chapter_name
#             ))

#         # Move start position for next chunk
#         start = end

#     return chunks


class ChunkSplitter:
    def __init__(self, max_chunk_size: int = 2048, overlap_percentage: float = 0.05):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = int(max_chunk_size * overlap_percentage)
        self.unique_sections: list[TextChunkSectionInfo] = []
        self.last_section_numbers = []

    def process(self, text: str) -> list[TextChunk]:
        """Main entry point for processing text into chunks."""
        lines = [line for line in text.replace("\r\n", "\n").split("\n") if line]
        root_section = self._build_section_tree(lines)
        chunks = self._convert_sections_to_chunks(root_section)

        self.post_process()

        return chunks

    def post_process(self):
        self.uniquefy_sections()

    def uniquefy_sections(self):
        seen_names = {}
        for section in self.unique_sections:
            if (
                section.name not in seen_names
                or section.level < seen_names[section.name].level
            ):
                seen_names[section.name] = section

        # Update unique_sections with deduplicated list
        self.unique_sections = list(seen_names.values())
        # Sort by level and then by name for consistent ordering
        self.unique_sections.sort(key=lambda x: (x.level, x.name))

    def get_unique_sections(self) -> list[TextChunkSectionInfo]:
        return self.unique_sections

    def _build_section_tree(self, lines: list[str]) -> TextChunkSection:
        """Build a tree of sections from the text lines."""
        root_section = TextChunkSection(
            name="root",
            level=0,
            line_start=0,
            content="",
            subsections=[],
            parent=None,
            page_num=1,
            page_breaks={},
        )
        current_section = root_section
        section_stack = [root_section]

        # Track current page
        current_page = 1
        page_breaks = {}  # Store all page breaks
        has_sections = False

        for line_num, line in enumerate(lines, 1):
            # Handle page breaks
            if line.startswith("-----"):
                current_page += 1
                page_breaks[line_num] = current_page
                current_section.page_breaks.update({line_num: current_page})
                continue

            # Handle section headers (both # and numbered)
            is_section = False
            level = 0

            if line.startswith("#"):
                is_section = True
                has_sections = True
                level = len(re.match(r"^#+", line).group())
                section_name = line.lstrip("#").strip()

            if is_section:
                # Close previous sections of equal or higher level
                while section_stack and section_stack[-1].level >= level:
                    popped_section = section_stack.pop()
                    popped_section.line_end = line_num - 1

                # Create new section with current page number
                new_section = TextChunkSection(
                    name=section_name,
                    level=level,
                    line_start=line_num,
                    content="",
                    subsections=[],
                    parent=section_stack[-1],
                    page_num=current_page,
                    page_breaks=dict(page_breaks),
                )

                # Add to parent's subsections
                section_stack[-1].subsections.append(new_section)

                # Update stack and current section
                section_stack.append(new_section)
                current_section = new_section

            else:
                # Add content to current section
                current_section.content += line + "\n"

        # Close any remaining open sections
        for section in reversed(section_stack):
            section.line_end = len(lines)

        # Store the total number of pages
        root_section.total_pages = current_page - 1

        # If no sections were found, create a default "Initial Section"
        if not has_sections and lines:
            default_section = TextChunkSection(
                name="Initial Section",
                level=1,
                line_start=1,
                line_end=len(lines),
                content=root_section.content,
                subsections=[],
                parent=root_section,
                page_num=1,
                page_breaks=dict(page_breaks),
            )
            root_section.subsections.append(default_section)

        return root_section

    def _get_pages_for_chunk(
        self,
        chunk_start: int,
        chunk_end: int,
        page_breaks: dict[int, int],
        total_pages: int,
    ) -> list[int]:
        """Determine which pages a chunk spans."""
        pages = set()

        # Find the exact page range for the chunk
        chunk_start_page = 1  # Default to first page
        chunk_end_page = 1  # Default to first page

        # Sort page breaks for consistent processing
        sorted_breaks = sorted(page_breaks.items())

        # Find the starting page
        for line_num, page_num in sorted_breaks:
            if line_num <= chunk_start:
                chunk_start_page = page_num
            else:
                break

        # Find the ending page
        for line_num, page_num in sorted_breaks:
            if line_num <= chunk_end:
                chunk_end_page = page_num
            else:
                break

        # Add all pages in the range
        for page in range(chunk_start_page, chunk_end_page + 1):
            if page <= total_pages:  # Ensure we don't exceed total pages
                pages.add(page)

        return sorted(list(pages))

    def _convert_sections_to_chunks(
        self, root_section: TextChunkSection
    ) -> list[TextChunk]:
        """Convert the section tree into a flat list of chunks."""
        chunks = []
        self._process_section_to_chunks(root_section, [], chunks)
        return chunks

    def _process_section_to_chunks(
        self,
        section: TextChunkSection,
        parent_sections: list[str],
        chunks: list[TextChunk],
    ) -> None:
        """Process a section and its subsections recursively to create chunks."""
        if section.level > 0:  # Skip root section
            # Add section with its level to unique_sections
            self.unique_sections.append(
                TextChunkSectionInfo(name=section.name, level=section.level)
            )

            current_section_path = parent_sections + [section.name]

            if section.content.strip():
                # Only use the current section name for the header
                section_header = f"{section.name}\n"

                # Split content normally - max_chunk_size applies only to content
                content_chunks = self._split_content_into_chunks(section.content)

                for i, content in enumerate(content_chunks):
                    # Add section header to each chunk's content after splitting
                    content_with_header = section_header + content

                    chunk_start = section.line_start + i * (self.max_chunk_size // 100)
                    chunk_end = min(
                        section.line_start + (i + 1) * (self.max_chunk_size // 100)
                        if i < len(content_chunks) - 1
                        else section.line_end,
                        section.line_end,
                    )

                    chunk_pages = self._get_pages_for_chunk(
                        chunk_start,
                        chunk_end,
                        section.page_breaks,
                        section.root.total_pages,
                    )

                    chunk = TextChunk(
                        content=content_with_header,
                        line_start=chunk_start,
                        line_end=chunk_end,
                        sections=current_section_path.copy(),
                        pages=chunk_pages,
                    )
                    chunks.append(chunk)
        else:
            current_section_path = parent_sections

        # Process subsections
        for subsection in section.subsections:
            self._process_section_to_chunks(subsection, current_section_path, chunks)

    def _split_content_into_chunks(self, content: str) -> list[str]:
        """Split content into chunks while preserving word boundaries and adding overlap."""
        chunks = []
        words = content.split()
        current_chunk = []
        current_size = 0
        overlap_words = []
        overlap_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space

            # If we'd exceed max size with this word and we have content
            if current_size + word_size > self.max_chunk_size and current_chunk:
                # Add the current chunk to our results
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap from previous chunk
                overlap_end = len(current_chunk)
                while overlap_size < self.overlap_size and overlap_end > 0:
                    overlap_end -= 1
                    word_to_add = current_chunk[overlap_end]
                    overlap_size += len(word_to_add) + 1
                    overlap_words.insert(0, word_to_add)

                # Start new chunk with overlap words plus current word
                current_chunk = overlap_words + [word]
                current_size = overlap_size + word_size

                # Reset overlap tracking
                overlap_words = []
                overlap_size = 0
            else:
                current_chunk.append(word)
                current_size += word_size

        # Add the last chunk if we have one
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


def chunk_text(text: str) -> List[TextChunk]:
    splitter = ChunkSplitter()
    return splitter.process(text)


async def get_title_and_summary(chunk: str, document_name: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Document: {document_name}\n\nContent:\n{chunk[:1000]}...",
                },
                # Send first 1000 chars for context
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
        }


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def process_chunk(
    chunk: TextChunk, chunk_number: int, document_name: str
) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk.content, document_name)

    # Get embedding
    embedding = await get_embedding(chunk.content)

    section_name = chunk.sections[-1] if len(chunk.sections) > 0 else ""

    # Create metadata
    metadata = {
        "source": DOCUMENTS_SOURCE_TYPE,
        "chunk_size": len(chunk.content),
        "section_name": section_name,
    }

    return ProcessedChunk(
        document_name=document_name,
        chapter_name=section_name,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk.content,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding,
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "document_name": chunk.document_name,
            "chapter_name": chunk.chapter_name,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
        }

        result = supabase.table(DOCUMENTS_SOURCE_TYPE).insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.chapter_name}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None


async def process_and_store_document(document_name: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    # print(markdown)
    # Process chunks in parallel
    tasks = [
        process_chunk(
            chunk,
            i,
            document_name,
        )
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Store chunks in parallel
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)


async def process_pdf_document(pdf_path: str):
    """Process a PDF document and store its chunks."""
    try:
        # Extract filename as document name
        document_name = os.path.basename(pdf_path)

        print(f"Processing PDF: {document_name}")

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)

        # Process and store the document
        await process_and_store_document(document_name, pdf_text)

        print(f"Successfully processed PDF: {document_name}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")


async def main():
    """Process all PDF files in the documents folder."""
    # Check if documents folder exists
    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"Documents folder not found: {DOCUMENTS_FOLDER}")
        return

    # Get all PDF files in the documents folder
    # pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.pdf"))
    pdf_files = [
        "/Users/sdkappasrl/Desktop/UserFolder/git/cole_medin_agentic_rag/iot_docs_rag/airplane_documents/TAR ROMA sentenza 23262_2024.pdf"
        ]

    if not pdf_files:
        print(f"No PDF files found in {DOCUMENTS_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Process PDF files in batches of 5 in parallel
    batch_size = 5
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        print(f"Processing batch of {len(batch)} files (batch {i//batch_size + 1}/{(len(pdf_files) + batch_size - 1)//batch_size})")
        tasks = [process_pdf_document(pdf_file) for pdf_file in batch]
        await asyncio.gather(*tasks)

    print("Completed processing all PDF files")


if __name__ == "__main__":
    asyncio.run(main())
