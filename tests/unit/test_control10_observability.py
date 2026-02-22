"""
Unit tests for LangFuse integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from observability.langfuse_integration import (
    LangFuseTracer,
    get_langfuse_tracer,
    trace_llm,
    trace_agent,
    trace_tool
)


class TestLangFuseTracerInitialization:
    """Test LangFuse tracer initialization."""
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_public_key",
        "LANGFUSE_SECRET_KEY": "test_secret_key",
        "LANGFUSE_HOST": "https://test.langfuse.com"
    })
    @patch('observability.langfuse_integration.Langfuse')
    def test_tracer_initialization_with_credentials(self, mock_langfuse):
        """Test tracer initializes with credentials."""
        tracer = LangFuseTracer()
        
        assert tracer.enabled is True
        mock_langfuse.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_tracer_initialization_without_credentials(self):
        """Test tracer initialization without credentials."""
        tracer = LangFuseTracer()
        
        assert tracer.enabled is False
    
    @patch('observability.langfuse_integration.Langfuse', side_effect=ImportError)
    def test_tracer_initialization_without_langfuse(self, mock_langfuse):
        """Test tracer initialization without LangFuse installed."""
        tracer = LangFuseTracer()
        
        assert tracer.enabled is False


class TestLangFuseCallbackHandler:
    """Test LangFuse callback handler."""
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret"
    })
    @patch('observability.langfuse_integration.Langfuse')
    @patch('observability.langfuse_integration.CallbackHandler')
    def test_get_callback_handler_when_enabled(self, mock_callback, mock_langfuse):
        """Test getting callback handler when enabled."""
        tracer = LangFuseTracer()
        handler = tracer.get_callback_handler()
        
        assert handler is not None
    
    def test_get_callback_handler_when_disabled(self):
        """Test getting callback handler when disabled."""
        tracer = LangFuseTracer()
        tracer.enabled = False
        
        handler = tracer.get_callback_handler()
        
        assert handler is None


class TestLLMTracing:
    """Test LLM call tracing."""
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret"
    })
    @patch('observability.langfuse_integration.Langfuse')
    def test_trace_llm_call(self, mock_langfuse):
        """Test tracing an LLM call."""
        tracer = LangFuseTracer()
        tracer.langfuse = MagicMock()
        
        tracer.trace_llm_call(
            name="test_llm",
            model="gpt-4o-mini",
            input_data="Test input",
            output_data="Test output",
            user_id="user123",
            session_id="session123"
        )
        
        tracer.langfuse.generation.assert_called_once()
    
    def test_trace_llm_call_when_disabled(self):
        """Test tracing LLM call when disabled."""
        tracer = LangFuseTracer()
        tracer.enabled = False
        
        # Should not raise error
        tracer.trace_llm_call(
            name="test_llm",
            model="gpt-4o-mini",
            input_data="Test input",
            output_data="Test output"
        )


class TestAgentTracing:
    """Test agent execution tracing."""
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret"
    })
    @patch('observability.langfuse_integration.Langfuse')
    def test_trace_agent_execution(self, mock_langfuse):
        """Test tracing an agent execution."""
        tracer = LangFuseTracer()
        tracer.langfuse = MagicMock()
        
        tracer.trace_agent_execution(
            name="test_agent",
            agent_type="worker",
            input_data="Test query",
            output_data="Test result",
            tools_used=["tool1", "tool2"],
            user_id="user123",
            session_id="session123"
        )
        
        tracer.langfuse.trace.assert_called_once()
    
    def test_trace_agent_execution_when_disabled(self):
        """Test tracing agent execution when disabled."""
        tracer = LangFuseTracer()
        tracer.enabled = False
        
        # Should not raise error
        tracer.trace_agent_execution(
            name="test_agent",
            agent_type="worker",
            input_data="Test query",
            output_data="Test result"
        )


class TestToolTracing:
    """Test tool execution tracing."""
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret"
    })
    @patch('observability.langfuse_integration.Langfuse')
    def test_trace_tool_execution_success(self, mock_langfuse):
        """Test tracing a successful tool execution."""
        tracer = LangFuseTracer()
        tracer.langfuse = MagicMock()
        
        tracer.trace_tool_execution(
            name="test_tool_trace",
            tool_name="member_lookup",
            input_data="M12345",
            output_data={"member": "data"},
            success=True,
            user_id="user123",
            session_id="session123"
        )
        
        tracer.langfuse.span.assert_called_once()
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret"
    })
    @patch('observability.langfuse_integration.Langfuse')
    def test_trace_tool_execution_failure(self, mock_langfuse):
        """Test tracing a failed tool execution."""
        tracer = LangFuseTracer()
        tracer.langfuse = MagicMock()
        
        tracer.trace_tool_execution(
            name="test_tool_trace",
            tool_name="member_lookup",
            input_data="M12345",
            output_data=None,
            success=False,
            error="Member not found",
            user_id="user123",
            session_id="session123"
        )
        
        tracer.langfuse.span.assert_called_once()


class TestSupervisorRoutingTracing:
    """Test supervisor routing tracing."""
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret"
    })
    @patch('observability.langfuse_integration.Langfuse')
    def test_trace_supervisor_routing(self, mock_langfuse):
        """Test tracing a supervisor routing decision."""
        tracer = LangFuseTracer()
        tracer.langfuse = MagicMock()
        
        tracer.trace_supervisor_routing(
            supervisor_name="member_services_supervisor",
            input_query="Look up member M12345",
            routing_decision="member_lookup",
            reasoning="Query asks for member lookup",
            user_id="user123",
            session_id="session123"
        )
        
        tracer.langfuse.trace.assert_called_once()


class TestTracingDecorators:
    """Test tracing decorators."""
    
    @patch('observability.langfuse_integration.get_langfuse_tracer')
    def test_trace_llm_decorator(self, mock_get_tracer):
        """Test LLM tracing decorator."""
        mock_tracer = MagicMock()
        mock_tracer.enabled = True
        mock_get_tracer.return_value = mock_tracer
        
        @trace_llm
        def test_llm_function(input_text, model="gpt-4o-mini"):
            return "output"
        
        result = test_llm_function("test input", model="gpt-4o-mini")
        
        assert result == "output"
        mock_tracer.trace_llm_call.assert_called_once()
    
    @patch('observability.langfuse_integration.get_langfuse_tracer')
    def test_trace_agent_decorator(self, mock_get_tracer):
        """Test agent tracing decorator."""
        mock_tracer = MagicMock()
        mock_tracer.enabled = True
        mock_get_tracer.return_value = mock_tracer
        
        @trace_agent
        def test_agent_function(query, user_id, session_id):
            return "result"
        
        result = test_agent_function("test query", user_id="user123", session_id="session123")
        
        assert result == "result"
        mock_tracer.trace_agent_execution.assert_called_once()
    
    @patch('observability.langfuse_integration.get_langfuse_tracer')
    def test_trace_tool_decorator_success(self, mock_get_tracer):
        """Test tool tracing decorator on success."""
        mock_tracer = MagicMock()
        mock_tracer.enabled = True
        mock_get_tracer.return_value = mock_tracer
        
        @trace_tool
        def test_tool_function(input_data, user_id, session_id):
            return {"result": "data"}
        
        result = test_tool_function("test input", user_id="user123", session_id="session123")
        
        assert result == {"result": "data"}
        mock_tracer.trace_tool_execution.assert_called_once()
    
    @patch('observability.langfuse_integration.get_langfuse_tracer')
    def test_trace_tool_decorator_failure(self, mock_get_tracer):
        """Test tool tracing decorator on failure."""
        mock_tracer = MagicMock()
        mock_tracer.enabled = True
        mock_get_tracer.return_value = mock_tracer
        
        @trace_tool
        def test_tool_function(input_data, user_id, session_id):
            raise Exception("Tool failed")
        
        with pytest.raises(Exception):
            test_tool_function("test input", user_id="user123", session_id="session123")
        
        # Should still trace the failure
        mock_tracer.trace_tool_execution.assert_called_once()


class TestTracerSingleton:
    """Test tracer singleton pattern."""
    
    def test_get_langfuse_tracer_singleton(self):
        """Test get_langfuse_tracer returns same instance."""
        tracer1 = get_langfuse_tracer()
        tracer2 = get_langfuse_tracer()
        
        assert tracer1 is tracer2


class TestTracerFlush:
    """Test tracer flush functionality."""
    
    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret"
    })
    @patch('observability.langfuse_integration.Langfuse')
    def test_flush_when_enabled(self, mock_langfuse):
        """Test flushing traces when enabled."""
        tracer = LangFuseTracer()
        tracer.langfuse = MagicMock()
        
        tracer.flush()
        
        tracer.langfuse.flush.assert_called_once()
    
    def test_flush_when_disabled(self):
        """Test flushing traces when disabled."""
        tracer = LangFuseTracer()
        tracer.enabled = False
        
        # Should not raise error
        tracer.flush()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
