import asyncio
import os
from dotenv import load_dotenv
import logfire
from openai import AsyncOpenAI
from supabase import create_client, Client

from iot_docs_rag_agent import iot_docs_expert, IoTDocsDeps

load_dotenv()

async def debug_tool_part():
    """Debug script to print the structure of ToolCallPart"""
    print("Starting debug script")
    
    # Setup clients
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    # Prepare dependencies
    deps = IoTDocsDeps(supabase=supabase, openai_client=openai_client)
    
    # Run the agent with a simple query
    print("Running agent...")
    async with iot_docs_expert.run_stream(
        "What is MQTT?",
        deps=deps,
    ) as result:
        # Collect final text
        final_text = ""
        async for chunk in result.stream_text(delta=True):
            final_text += chunk
        
        print("\nFinal response:", final_text[:100], "...\n")
        
        # Debug the tool call parts
        print("Debugging message parts:")
        for i, msg in enumerate(result.new_messages()):
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if part.part_kind == "tool-call":
                        print(f"\nTool Call Part (Step {i}):")
                        print(f"  part_kind: {part.part_kind}")
                        # Print all attributes of the part
                        for attr_name in dir(part):
                            if not attr_name.startswith('_'):  # Skip private attributes
                                try:
                                    attr_value = getattr(part, attr_name)
                                    print(f"  {attr_name}: {attr_value}")
                                except Exception as e:
                                    print(f"  {attr_name}: <error: {e}>")

if __name__ == "__main__":
    asyncio.run(debug_tool_part()) 