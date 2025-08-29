import logging
import os
from typing import Dict, Any, Optional, List, Union
from httpx import ConnectError

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.pregel import Pregel

# Set up logging
logger = logging.getLogger(__name__)

def get_langfuse_callback() -> Optional[CallbackHandler]:
    """Get a configured Langfuse callback handler if available.
    
    Returns:
        A Langfuse callback handler or None if initialization fails
    """
    try:
        # Initialize Langfuse client from environment variables
        langfuse = get_client()
        
        # Verify authentication
        if langfuse.auth_check():
            logger.info("Langfuse client is authenticated and ready!")
            return CallbackHandler()
        else:
            logger.warning("Langfuse authentication failed. Please check your credentials.")
            return None
    except ConnectError as e:
        logger.warning(f"Connection error initializing Langfuse: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error initializing Langfuse: {e}")
        return None

class LangfuseSupervisorTracer:
    """Add Langfuse tracing to LangGraph Supervisor workflows."""

    def __init__(
        self, 
        public_key: Optional[str] = None, 
        secret_key: Optional[str] = None, 
        host: Optional[str] = None
    ):
        """Initialize the Langfuse tracer.
        
        Args:
            public_key: Langfuse public key (can be None if set in env vars)
            secret_key: Langfuse secret key (can be None if set in env vars)
            host: Langfuse host URL (can be None if set in env vars)
        """
        # Set environment variables if provided
        if public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        if secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        if host:
            os.environ["LANGFUSE_HOST"] = host
        
        # Get the callback handler
        self.callback_handler = get_langfuse_callback()
        
        # Log status
        if self.callback_handler is None:
            logger.warning("Failed to initialize Langfuse callback handler")
        else:
            logger.info("Successfully initialized Langfuse callback handler")

    def trace_workflow(self, workflow: Pregel) -> Pregel:
        """Add Langfuse tracing to a compiled LangGraph workflow.
        
        Args:
            workflow: The compiled LangGraph workflow to trace
            
        Returns:
            The same workflow with tracing enabled
        """
        # If no callback handler is available, return the original workflow
        if self.callback_handler is None:
            logger.warning("Langfuse callback handler not available. Returning original workflow without tracing.")
            return workflow

        original_invoke = workflow.invoke
        original_ainvoke = workflow.ainvoke

        def traced_invoke(state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Wrap the workflow invoke method with Langfuse tracing."""
            config = config or {}

            # Add Langfuse callbacks
            config_callbacks = config.get("callbacks", [])
            config["callbacks"] = [*config_callbacks, self.callback_handler]

            # Run the workflow
            result = original_invoke(state, config)
            return result

        async def traced_ainvoke(state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Wrap the workflow ainvoke method with Langfuse tracing."""
            config = config or {}

            # Add Langfuse callbacks
            config_callbacks = config.get("callbacks", [])
            config["callbacks"] = [*config_callbacks, self.callback_handler]

            # Run the workflow
            result = await original_ainvoke(state, config)
            return result

        # Replace the invoke methods with traced versions
        workflow.invoke = traced_invoke
        workflow.ainvoke = traced_ainvoke

        return workflow