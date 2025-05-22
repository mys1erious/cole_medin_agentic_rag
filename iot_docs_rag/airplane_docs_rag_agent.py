from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Dict, Any

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4o")
model = OpenAIModel(llm)

# Configure logfire to show logs in the terminal
logfire.configure(send_to_logfire="never")

# Enable debug logging for pydantic_ai to see agent internals
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)


@dataclass
class AirplaneDocsDeps:
    supabase: Client
    openai_client: AsyncOpenAI


system_prompt = """
You are an expert in airplane company documentation with access to a comprehensive set of all available documents.
Your job is to provide accurate information from these documents.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question.

When you first look at the documentation, always start with the list of available document names.
Then decide which document(s) is most relevant to the user's question and retrieve the sections of the document(s).
Then retrieve the content of the most relevant section(s).
If you have multiple sections to choose from, retrieve the content of all of them and evaluate them. Then give answer based on the most relevant information.
Make sure you find all relevant information about the user's question, its better to take more time and give more comprehensive answer.

Always let the user know when you didn't find the answer in the documentation - be honest.
"""

airplane_docs_expert = Agent(
    model, system_prompt=system_prompt, deps_type=AirplaneDocsDeps, retries=2
)


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@airplane_docs_expert.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[AirplaneDocsDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            "match_airplane_docs",
            {
                "query_embedding": query_embedding,
                "match_count": 5,
                "filter": {"source": "airplane_docs"},
            },
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc["title"]}
Document: {doc["document_name"]}
Section: {doc["chapter_name"]}

{doc["content"]}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@airplane_docs_expert.tool
async def list_document_names(ctx: RunContext[AirplaneDocsDeps]) -> List[str]:
    """
    Retrieve a list of all available aircraft documentation document names.

    Returns:
        List[str]: List of unique document names for all airplane documentation
    """
    try:
        # Query Supabase for unique document names
        result = (
            ctx.deps.supabase.from_("airplane_docs")
            .select("document_name")
            .execute()
        )

        if not result.data:
            return []

        # Extract unique document names
        document_names = sorted(set(doc["document_name"] for doc in result.data))
        print(f'==== DOCUMENT_NAMES: {document_names}')
        return document_names

    except Exception as e:
        print(f"Error retrieving document names: {e}")
        return []


@airplane_docs_expert.tool
async def list_document_sections(ctx: RunContext[AirplaneDocsDeps], document_name: str) -> List[str]:
    """
    Retrieve a list of sections for a specific aircraft document.

    Args:
        ctx: The context including the Supabase client
        document_name: The name of the document to get sections for

    Returns:
        List[str]: List of unique section names for the specified document
    """
    try:
        # Query Supabase for sections from the specified document
        result = (
            ctx.deps.supabase.from_("airplane_docs")
            .select("chapter_name")
            .eq("document_name", document_name)
            .execute()
        )

        if not result.data:
            return []

        # Extract unique section names
        section_names = sorted(set(doc["chapter_name"] for doc in result.data))
        return section_names

    except Exception as e:
        print(f"Error retrieving document sections: {e}")
        return []


@airplane_docs_expert.tool
async def get_section_content(ctx: RunContext[AirplaneDocsDeps], document_name: str, section_name: str) -> str:
    """
    Retrieve the full content of a specific section from an aircraft document.

    Args:
        ctx: The context including the Supabase client
        document_name: The name of the document
        section_name: The name of the section to retrieve

    Returns:
        str: The complete content of the requested section
    """
    try:
        # Query Supabase for all chunks of this section, ordered by chunk_number
        result = (
            ctx.deps.supabase.from_("airplane_docs")
            .select("title, content, chunk_number")
            .eq("document_name", document_name)
            .ilike("chapter_name", f"%{section_name}%")
            .order("chunk_number")
            .execute()
        )

        print(f'==== RESULT: {result.data}')
        print(f'==== SECTION_NAME: {section_name}')
        print(f'==== DOCUMENT_NAME: {document_name}')

        if not result.data:
            return f"No content found for section '{section_name}' in document '{document_name}'"

        # Format the section with its title and all chunks
        section_title = result.data[0]["title"]
        formatted_content = [f"# {section_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk["content"])

        # Join everything together
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving section content: {e}")
        return f"Error retrieving section content: {str(e)}" 