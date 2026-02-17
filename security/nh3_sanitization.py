"""
NH3 Library Integration for XSS Prevention (Control 3: Input Sanitization)

This module uses the NH3 library (Mozilla's ammonia port to Python) for HTML sanitization
and XSS prevention. NH3 is faster and more secure than alternatives like bleach.

Control 3: Input Sanitization
- XSS prevention using NH3
- HTML sanitization
- Safe output encoding
"""

import logging
from typing import Optional, Set
import nh3

logger = logging.getLogger(__name__)


class NH3Sanitizer:
    """
    NH3-based HTML sanitizer for XSS prevention.
    
    Uses Mozilla's NH3 library (Rust-based ammonia port) for fast and secure
    HTML sanitization.
    """
    
    def __init__(
        self,
        allowed_tags: Optional[Set[str]] = None,
        allowed_attributes: Optional[dict] = None
    ):
        """
        Initialize NH3 sanitizer.
        
        Args:
            allowed_tags: Set of allowed HTML tags (None = default safe tags)
            allowed_attributes: Dict of tag -> allowed attributes (None = default safe attributes)
        """
        # Default safe tags (conservative list)
        self.allowed_tags = allowed_tags or {
            'p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li',
            'blockquote', 'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        }
        
        # Default safe attributes per tag
        self.allowed_attributes = allowed_attributes or {
            'a': {'href', 'title'},
            'blockquote': {'cite'},
            'code': {'class'},
        }
        
        logger.info(f"NH3 Sanitizer initialized with {len(self.allowed_tags)} allowed tags")
    
    def sanitize(self, html: str) -> str:
        """
        Sanitize HTML input to prevent XSS attacks.
        
        Args:
            html: HTML string to sanitize
        
        Returns:
            Sanitized HTML string
        """
        try:
            # Use NH3 to clean HTML
            # NH3 automatically removes dangerous tags, attributes, and JavaScript
            cleaned = nh3.clean(
                html,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                link_rel="noopener noreferrer"  # Security for external links
            )
            
            logger.debug(f"Sanitized HTML: {len(html)} -> {len(cleaned)} chars")
            
            return cleaned
        
        except Exception as e:
            logger.error(f"NH3 sanitization failed: {e}")
            # On error, strip all HTML as fallback
            return self.strip_all_html(html)
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize plain text by escaping HTML entities.
        
        Args:
            text: Plain text to sanitize
        
        Returns:
            Escaped text safe for HTML output
        """
        try:
            # NH3 can also be used to escape plain text
            # This converts < > & " ' to HTML entities
            escaped = nh3.clean(text, tags=set())  # No tags allowed
            
            return escaped
        
        except Exception as e:
            logger.error(f"Text sanitization failed: {e}")
            return text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
    
    def strip_all_html(self, html: str) -> str:
        """
        Strip all HTML tags from input.
        
        Args:
            html: HTML string
        
        Returns:
            Plain text with all HTML removed
        """
        try:
            # NH3 with empty tag set strips all HTML
            stripped = nh3.clean(html, tags=set())
            
            return stripped
        
        except Exception as e:
            logger.error(f"HTML stripping failed: {e}")
            return html
    
    def is_safe(self, html: str) -> bool:
        """
        Check if HTML is safe (no dangerous content).
        
        Args:
            html: HTML string to check
        
        Returns:
            True if safe, False if dangerous content detected
        """
        try:
            cleaned = self.sanitize(html)
            
            # If cleaned version is identical, input was safe
            return cleaned == html
        
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return False


# Singleton instance
_nh3_sanitizer: Optional[NH3Sanitizer] = None


def get_nh3_sanitizer() -> NH3Sanitizer:
    """Get or create singleton NH3 sanitizer instance."""
    global _nh3_sanitizer
    if _nh3_sanitizer is None:
        _nh3_sanitizer = NH3Sanitizer()
    return _nh3_sanitizer


def sanitize_html(html: str) -> str:
    """
    Convenience function to sanitize HTML.
    
    Args:
        html: HTML string to sanitize
    
    Returns:
        Sanitized HTML
    """
    sanitizer = get_nh3_sanitizer()
    return sanitizer.sanitize(html)


def sanitize_text(text: str) -> str:
    """
    Convenience function to sanitize plain text.
    
    Args:
        text: Plain text to sanitize
    
    Returns:
        Escaped text safe for HTML output
    """
    sanitizer = get_nh3_sanitizer()
    return sanitizer.sanitize_text(text)


def strip_html(html: str) -> str:
    """
    Convenience function to strip all HTML.
    
    Args:
        html: HTML string
    
    Returns:
        Plain text with HTML removed
    """
    sanitizer = get_nh3_sanitizer()
    return sanitizer.strip_all_html(html)
