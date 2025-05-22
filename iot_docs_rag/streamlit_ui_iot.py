from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from iot_docs_rag_agent import iot_docs_expert, IoTDocsDeps

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="never")


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)
    # tool-call
    elif part.part_kind == "tool-call":
        with st.chat_message("assistant", avatar="🔧"):
            st.markdown(f"**Tool Call**: `{part.tool_name}`")
            if hasattr(part, "args") and part.args:
                st.code(part.args, language="json")
    # tool-result
    elif part.part_kind == "tool-result":
        with st.chat_message("assistant", avatar="📊"):
            st.markdown("**Tool Result**:")
            st.code(str(part.content), language="json")


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = IoTDocsDeps(supabase=supabase, openai_client=openai_client)
    
    # Add a checkbox to toggle viewing reasoning steps
    show_reasoning = st.sidebar.checkbox("Show Agent Reasoning", value=False)
    
    # Log the start of agent execution
    logfire.info("Starting agent execution", user_input=user_input)
    
    # Run the agent in a stream
    async with iot_docs_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[
            :-1
        ],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Log all reasoning steps for terminal visibility
        for i, msg in enumerate(result.new_messages()):
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    # Log different part types to the terminal
                    if part.part_kind == "tool-call":
                        tool_args = getattr(part, "args", {})
                        logfire.debug(
                            "Tool call",
                            step=i,
                            tool_name=getattr(part, "tool_name", "unknown"),
                            tool_args=tool_args
                        )
                    elif part.part_kind == "tool-result":
                        logfire.debug(
                            "Tool result",
                            step=i,
                            content=part.content
                        )
                    elif part.part_kind == "text":
                        logfire.debug(
                            "Agent reasoning", 
                            step=i, 
                            content=part.content[:100] + "..." if len(part.content) > 100 else part.content
                        )
                        
        if show_reasoning:
            # Display all the reasoning steps if checkbox is enabled
            st.subheader("Agent Reasoning Steps:")
            for msg in result.new_messages():
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        display_message_part(part)
        
        # Add new messages to conversation history
        filtered_messages = [
            msg
            for msg in result.new_messages()
            if not (
                hasattr(msg, "parts")
                and any(part.part_kind == "user-prompt" for part in msg.parts)
            )
        ]
        
        # Only add to session state if not showing reasoning (to avoid duplicates)
        if not show_reasoning:
            st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("IoT Documentation RAG")
    st.write(
        "Ask any question about IoT concepts, protocols, hardware specifications, and implementation details."
    )

    # Add sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.write("Toggle options to control the agent's behavior and visualization.")
    
    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about IoT?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
