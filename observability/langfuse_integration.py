"""
LangFuse integration for LLM and agent observability.
"""

import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from config.settings import get_settings, Settings
            
logger = logging.getLogger(__name__)

settings: Settings = get_settings()

class LangFuseTracer:
    """
    LangFuse tracer for capturing LLM and agent invocations.
    """
    
    def __init__(self):
        """Initialize LangFuse tracer."""
        self.enabled = False
        self.langfuse = None
        
        try:
            # Initialize LangFuse client
            public_key = settings.LANGFUSE_PUBLIC_KEY
            secret_key = settings.LANGFUSE_SECRET_KEY
            host = settings.LANGFUSE_HOST
            
            if public_key and secret_key:
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                
                # Create callback handler for LangChain integration
                self.callback_handler = CallbackHandler(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                
                self.enabled = True
                logger.info(f"LangFuse integration enabled (host: {host})")
            else:
                logger.warning("LangFuse credentials not found, tracing disabled")
        except ImportError:
            logger.warning("LangFuse not installed, tracing disabled")
        except Exception as e:
            logger.error(f"Failed to initialize LangFuse: {e}")
    
    def get_callback_handler(self):
        """
        Get LangFuse callback handler for LangChain integration.
        
        Returns:
            CallbackHandler instance or None if not enabled
        """
        if self.enabled:
            return self.callback_handler
        return None
    
    def trace_llm_call(
        self,
        name: str,
        model: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Trace an LLM call.
        
        Args:
            name: Name of the LLM call
            model: Model name (e.g., "gpt-4o-mini")
            input_data: Input to the LLM
            output_data: Output from the LLM
            metadata: Additional metadata
            user_id: User ID
            session_id: Session ID
        """
        if not self.enabled:
            return
        
        try:
            self.langfuse.generation(
                name=name,
                model=model,
                input=input_data,
                output=output_data,
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
                start_time=datetime.now(),
                end_time=datetime.now()
            )
        except Exception as e:
            logger.error(f"Failed to trace LLM call: {e}")
    
    def trace_agent_execution(
        self,
        name: str,
        agent_type: str,
        input_data: Any,
        output_data: Any,
        tools_used: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Trace an agent execution.
        
        Args:
            name: Name of the agent
            agent_type: Type of agent (e.g., "worker", "supervisor")
            input_data: Input to the agent
            output_data: Output from the agent
            tools_used: List of tools used by the agent
            metadata: Additional metadata
            user_id: User ID
            session_id: Session ID
        """
        if not self.enabled:
            return
        
        try:
            trace_metadata = metadata or {}
            trace_metadata.update({
                "agent_type": agent_type,
                "tools_used": tools_used or []
            })
            
            self.langfuse.trace(
                name=name,
                input=input_data,
                output=output_data,
                metadata=trace_metadata,
                user_id=user_id,
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Failed to trace agent execution: {e}")
    
    def trace_tool_execution(
        self,
        name: str,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Trace a tool execution.
        
        Args:
            name: Name of the trace
            tool_name: Name of the tool
            input_data: Input to the tool
            output_data: Output from the tool
            success: Whether the tool execution was successful
            error: Error message if failed
            metadata: Additional metadata
            user_id: User ID
            session_id: Session ID
        """
        if not self.enabled:
            return
        
        try:
            trace_metadata = metadata or {}
            trace_metadata.update({
                "tool_name": tool_name,
                "success": success,
                "error": error
            })
            
            self.langfuse.span(
                name=name,
                input=input_data,
                output=output_data,
                metadata=trace_metadata,
                user_id=user_id,
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Failed to trace tool execution: {e}")
    
    def trace_supervisor_routing(
        self,
        supervisor_name: str,
        input_query: str,
        routing_decision: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Trace a supervisor routing decision.
        
        Args:
            supervisor_name: Name of the supervisor
            input_query: Input query
            routing_decision: Routing decision (next worker/team)
            reasoning: Reasoning for the decision
            metadata: Additional metadata
            user_id: User ID
            session_id: Session ID
        """
        if not self.enabled:
            return
        
        try:
            trace_metadata = metadata or {}
            trace_metadata.update({
                "supervisor": supervisor_name,
                "routing_decision": routing_decision,
                "reasoning": reasoning
            })
            
            self.langfuse.trace(
                name=f"{supervisor_name}_routing",
                input=input_query,
                output=routing_decision,
                metadata=trace_metadata,
                user_id=user_id,
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Failed to trace supervisor routing: {e}")
    
    def flush(self):
        """Flush pending traces to LangFuse."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush LangFuse traces: {e}")


# Global tracer instance
_langfuse_tracer = None

def get_langfuse_tracer() -> LangFuseTracer:
    """Get or create the global LangFuse tracer instance."""
    global _langfuse_tracer
    if _langfuse_tracer is None:
        _langfuse_tracer = LangFuseTracer()
    return _langfuse_tracer


def trace_llm(func: Callable) -> Callable:
    """
    Decorator to trace LLM calls.
    
    Usage:
        @trace_llm
        def my_llm_function(input_text):
            return llm.invoke(input_text)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_langfuse_tracer()
        
        if not tracer.enabled:
            return func(*args, **kwargs)
        
        # Extract metadata from kwargs if available
        user_id = kwargs.get('user_id')
        session_id = kwargs.get('session_id')
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Trace the call
        tracer.trace_llm_call(
            name=func.__name__,
            model=kwargs.get('model', 'unknown'),
            input_data=args[0] if args else kwargs.get('input'),
            output_data=result,
            user_id=user_id,
            session_id=session_id
        )
        
        return result
    
    return wrapper


def trace_agent(func: Callable) -> Callable:
    """
    Decorator to trace agent executions.
    
    Usage:
        @trace_agent
        def my_agent_function(query, user_id, session_id):
            return agent.execute(query)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_langfuse_tracer()
        
        if not tracer.enabled:
            return func(*args, **kwargs)
        
        # Extract metadata
        user_id = kwargs.get('user_id')
        session_id = kwargs.get('session_id')
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Trace the execution
        tracer.trace_agent_execution(
            name=func.__name__,
            agent_type=kwargs.get('agent_type', 'unknown'),
            input_data=args[0] if args else kwargs.get('query'),
            output_data=result,
            user_id=user_id,
            session_id=session_id
        )
        
        return result
    
    return wrapper


def trace_tool(func: Callable) -> Callable:
    """
    Decorator to trace tool executions.
    
    Usage:
        @trace_tool
        def my_tool_function(input_data, user_id, session_id):
            return tool.execute(input_data)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_langfuse_tracer()
        
        if not tracer.enabled:
            return func(*args, **kwargs)
        
        # Extract metadata
        user_id = kwargs.get('user_id')
        session_id = kwargs.get('session_id')
        tool_name = func.__name__
        
        success = True
        error = None
        result = None
        
        try:
            # Execute function
            result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            # Trace the execution
            tracer.trace_tool_execution(
                name=f"tool_{tool_name}",
                tool_name=tool_name,
                input_data=args[0] if args else kwargs.get('input'),
                output_data=result,
                success=success,
                error=error,
                user_id=user_id,
                session_id=session_id
            )
        
        return result
    
    return wrapper
